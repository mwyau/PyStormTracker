from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models import constants as model_constants
from ..models.time import TimeInput, TimeRange, is_missing_time, select_time_range
from ..models.tracker import RawDetectionStep
from ..preprocessing.refinement import subgrid_refine as refine_center
from .kernels import (
    _numba_extrema_filter,
    _numba_get_centers,
    _numba_laplace_masked,
    _numba_remove_dup,
)


class SimpleDetector:
    """
    A meteorological feature detector that treats fields as 2D images.
    Uses xarray for robust coordinate handling and lazy-loading.
    """

    _ds_cache: ClassVar[dict[Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        pathname: str | Path | None,
        variable_name: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = (
            Path(pathname)
            if pathname is not None
            and not (isinstance(pathname, str) and "://" in pathname)
            else pathname
        )
        self.requested_variable_name = variable_name
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps

        self._loader = DataLoader(self.pathname, engine=engine)
        self._data: xr.DataArray | None = None
        self.variable_name = variable_name  # Updated after open

    def _ensure_open(self) -> None:
        """Ensures the xarray dataset is open and basic variables are mapped."""
        if self._data is None:
            ds = self._loader.ensure_open()

            # Identify the actual variable name using mapping aliases
            actual_var = None
            possible_names = DataLoader.VAR_MAPPING.get(
                self.requested_variable_name, [self.requested_variable_name]
            )
            for name in possible_names:
                if name in ds.data_vars:
                    actual_var = name
                    break

            if actual_var is None:
                if self.requested_variable_name in ds.data_vars:
                    actual_var = self.requested_variable_name
                else:
                    raise KeyError(
                        f"Variable '{self.requested_variable_name}' not found. "
                        f"Available: {list(ds.data_vars.keys())}"
                    )

            self.variable_name = actual_var
            self._data = ds[self.variable_name]

    @property
    def lat(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, lat_name, _ = self._loader.get_coords()
        return np.asarray(ds[lat_name].values)

    @property
    def lon(self) -> NDArray[np.float64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        _, _, lon_name = self._loader.get_coords()
        return np.asarray(ds[lon_name].values)

    def get_var(
        self, frame: int | tuple[int, int] | None = None
    ) -> NDArray[np.float64] | None:
        self._ensure_open()
        assert self._data is not None

        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            # Handle NaT bounds with explicit types
            # xarray .sel() accepts DataArray or slice
            if not is_missing_time(start) and not is_missing_time(end):
                data_range = self._data.sel({time_dim: slice(start, end)})
            elif not is_missing_time(start):
                data_range = self._data.sel({time_dim: slice(start, None)})
            elif not is_missing_time(end):
                data_range = self._data.sel({time_dim: slice(None, end)})
            else:
                data_range = self._data
        else:
            data_range = self._data

        match frame:
            case int(idx):
                data = data_range.isel({time_dim: idx})
                return np.asarray(data.values).reshape((data.shape[-2], data.shape[-1]))

            case (int(s_off), int(e_off)):
                data = data_range.isel({time_dim: slice(s_off, e_off)})
                return np.asarray(
                    data.values.reshape((data.shape[0], data.shape[-2], data.shape[-1]))
                )

            case None:
                return np.asarray(
                    data_range.values.reshape(
                        (
                            data_range.shape[0],
                            data_range.shape[-2],
                            data_range.shape[-1],
                        )
                    )
                )

            case _:
                raise TypeError("frame must be an int, tuple[int, int], or None")

    def get_time(self) -> np.ndarray | None:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            time_coord = ds[time_dim]
            if not is_missing_time(start) and not is_missing_time(end):
                times = time_coord.sel({time_dim: slice(start, end)})
            elif not is_missing_time(start):
                times = time_coord.sel({time_dim: slice(start, None)})
            elif not is_missing_time(end):
                times = time_coord.sel({time_dim: slice(None, end)})
            else:
                times = time_coord
        else:
            times = ds[time_dim]
        return np.asarray(times.values)

    def get_xarray(
        self,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
    ) -> xr.DataArray:
        """Returns the requested data range as an xarray DataArray."""
        self._ensure_open()
        assert self._data is not None
        effective_start = (
            start_time
            if start_time is not None
            else self.time_range.start
            if self.time_range is not None
            else None
        )
        effective_end = (
            end_time
            if end_time is not None
            else self.time_range.end
            if self.time_range is not None
            else None
        )
        selected = select_time_range(
            self._data,
            start_time=effective_start,
            end_time=effective_end,
        )
        assert isinstance(selected, xr.DataArray)
        return selected

    @classmethod
    def from_xarray(
        cls, data: xr.DataArray, variable_name: str | None = None
    ) -> SimpleDetector:
        """Creates a detector from an existing xarray DataArray."""
        obj = cls.__new__(cls)
        obj.requested_variable_name = variable_name or (
            str(data.name) if data.name else "var"
        )
        obj.variable_name = obj.requested_variable_name
        obj._data = data
        obj._loader = DataLoader(data)
        obj.pathname = None
        obj.time_range = None
        obj.global_start_idx = 0
        obj.global_total_steps = None
        return obj

    def split(self, num: int) -> list[SimpleDetector]:
        self._ensure_open()
        time_name, _, _ = self._loader.get_coords()
        time_coord = self._loader.ensure_open()[time_name]

        # Determine total length based on active time range
        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            if not is_missing_time(start) and not is_missing_time(end):
                active_times = time_coord.sel({time_name: slice(start, end)})
            elif not is_missing_time(start):
                active_times = time_coord.sel({time_name: slice(start, None)})
            elif not is_missing_time(end):
                active_times = time_coord.sel({time_name: slice(None, end)})
            else:
                active_times = time_coord
        else:
            active_times = time_coord

        time_values = np.asarray(active_times.values)
        total_len = len(time_values)

        chunk_size = total_len // num
        remainder = total_len % num

        detectors: list[SimpleDetector] = []
        for i in range(num):
            s_idx = i * chunk_size + min(i, remainder)
            e_idx = (i + 1) * chunk_size + min(i + 1, remainder)

            if s_idx >= e_idx:
                continue

            if self.pathname is None and self._data is not None:
                # Preserve in-memory data for split detectors
                new_obj = SimpleDetector.from_xarray(self._data)
                new_obj.time_range = TimeRange(
                    start=time_values[s_idx], end=time_values[e_idx - 1]
                )
                new_obj.global_start_idx = s_idx
                new_obj.global_total_steps = total_len
                detectors.append(new_obj)
            else:
                detectors.append(
                    SimpleDetector(
                        self.pathname,
                        self.requested_variable_name,
                        time_range=TimeRange(
                            start=time_values[s_idx], end=time_values[e_idx - 1]
                        ),
                        global_start_idx=s_idx,
                        global_total_steps=total_len,
                        engine=self._loader.engine,
                    )
                )
        return detectors

    def detect(
        self,
        size: int = 5,
        threshold: float | None = None,
        minmaxmode: Literal["min", "max"] = "min",
        subgrid_refine: bool = False,
    ) -> list[RawDetectionStep]:
        if size % 2 != 1:
            raise ValueError("size must be an odd number")

        # Set variable specific thresholds if not provided
        if threshold is None:
            if self.requested_variable_name == "vo":
                threshold = model_constants.DEFAULT_VO_THRESHOLD
            else:
                threshold = model_constants.DEFAULT_MSL_THRESHOLD

        time_array = self.get_time()
        lat, lon = self.lat, self.lon
        _, _, lon_name = self._loader.get_coords()
        periodic_x = lon_name != "x" and self._loader.is_global_longitude()
        assert time_array is not None
        num_steps = len(time_array)

        # Optimization: Read the entire time range for this worker in one go
        full_var = self.get_var()
        assert full_var is not None

        raw_results: list[RawDetectionStep] = []
        is_min = minmaxmode == "min"

        for it, t in enumerate(time_array):
            if (it + 1) % 10 == 0 or it == 0 or it == num_steps - 1:
                if self.global_total_steps:
                    s_idx = self.global_start_idx + it + 1
                    g_steps = self.global_total_steps
                    print(f"  Step {it + 1}/{num_steps} (Global: {s_idx}/{g_steps})")
                else:
                    print(f"  Step {it + 1}/{num_steps}")

            frame = full_var[it, :, :]

            fill = np.inf if is_min else -np.inf
            filled_frame = np.where(np.isnan(frame), fill, frame)

            extrema = _numba_extrema_filter(
                filled_frame, size, threshold, is_min, periodic_x
            )

            if np.isnan(frame).any():
                extrema[np.isnan(frame)] = 0

            laplacian = _numba_laplace_masked(filled_frame, extrema, is_min, periodic_x)
            extrema = _numba_remove_dup(laplacian, size=5, periodic_x=periodic_x)

            # Extract raw data using Numba
            r_idx, c_idx, vals = _numba_get_centers(extrema, frame)
            time_val = t

            if subgrid_refine:
                refined_lats = np.empty(len(r_idx), dtype=np.float64)
                refined_lons = np.empty(len(r_idx), dtype=np.float64)
                refined_vals = np.empty(len(r_idx), dtype=np.float64)
                for i in range(len(r_idx)):
                    refined_lats[i], refined_lons[i], refined_vals[i] = refine_center(
                        frame,
                        r_idx[i],
                        c_idx[i],
                        lat,
                        lon,
                        periodic_x=periodic_x,
                    )
                raw_results.append(
                    RawDetectionStep(
                        time_val,
                        refined_lats,
                        refined_lons,
                        refined_vals,
                    )
                )
            else:
                raw_results.append(
                    RawDetectionStep(time_val, lat[r_idx], lon[c_idx], vals)
                )

        return raw_results
