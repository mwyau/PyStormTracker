from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.loader import DataLoader
from ..models.tracks import TimeRange
from .kernels import (
    _numba_extrema_filter,
    _numba_get_centers,
    _numba_laplace_masked,
    _numba_remove_dup,
)

# Type alias for a single time step's raw detection arrays
RawDetectionStep: TypeAlias = tuple[
    np.datetime64,
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
]


class SimpleDetector:
    """
    A meteorological feature detector that treats fields as 2D images.
    Uses xarray for robust coordinate handling and lazy-loading.
    """

    # Common variable name aliases
    VAR_MAPPING: dict[str, list[str]] = {
        "msl": ["msl", "slp", "prmsl", "mean_sea_level_pressure"],
        "vo": ["vo", "rv", "relative_vorticity", "vorticity"],
    }

    def __init__(
        self,
        pathname: str | Path,
        varname: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = Path(pathname)
        self.requested_varname = varname
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps
        
        self._loader = DataLoader(self.pathname, engine=engine)
        self._data: xr.DataArray | None = None
        self.varname = varname  # Updated after open

    def _ensure_open(self) -> None:
        """Ensures the xarray dataset is open and basic variables are mapped."""
        if self._data is None:
            ds = self._loader.ensure_open()
            
            # Identify the actual variable name using mapping aliases
            actual_var = None
            possible_names = self.VAR_MAPPING.get(
                self.requested_varname, [self.requested_varname]
            )
            for name in possible_names:
                if name in ds.data_vars:
                    actual_var = name
                    break

            if actual_var is None:
                if self.requested_varname in ds.data_vars:
                    actual_var = self.requested_varname
                else:
                    raise KeyError(
                        f"Variable '{self.requested_varname}' not found. "
                        f"Available: {list(ds.data_vars.keys())}"
                    )

            self.varname = actual_var
            self._data = ds[self.varname]

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
            sel_dict = {}
            if not np.isnat(start) and not np.isnat(end):
                sel_dict[time_dim] = slice(start, end)
            elif not np.isnat(start):
                sel_dict[time_dim] = self._data[time_dim] >= start
            elif not np.isnat(end):
                sel_dict[time_dim] = self._data[time_dim] <= end
            
            data_range = self._data.sel(sel_dict) if sel_dict else self._data
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

    def get_time(self) -> NDArray[np.datetime64] | None:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()
        
        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            sel_dict = {}
            if not np.isnat(start) and not np.isnat(end):
                sel_dict[time_dim] = slice(start, end)
            elif not np.isnat(start):
                sel_dict[time_dim] = ds[time_dim] >= start
            elif not np.isnat(end):
                sel_dict[time_dim] = ds[time_dim] <= end
                
            times = ds[time_dim].sel(sel_dict) if sel_dict else ds[time_dim]
        else:
            times = ds[time_dim]
        return np.asarray(times.values).astype("datetime64[s]")

    def split(self, num: int) -> list[SimpleDetector]:
        self._ensure_open()
        time_name, _, _ = self._loader.get_coords()
        time_coord = self._loader.ensure_open()[time_name]
        
        # Determine total length based on active time range
        if self.time_range:
            start, end = self.time_range.start, self.time_range.end
            sel_dict = {}
            if not np.isnat(start) and not np.isnat(end):
                sel_dict[time_name] = slice(start, end)
            elif not np.isnat(start):
                sel_dict[time_name] = time_coord >= start
            elif not np.isnat(end):
                sel_dict[time_name] = time_coord <= end
            active_times = time_coord.sel(sel_dict) if sel_dict else time_coord
        else:
            active_times = time_coord

        time_values = np.asarray(active_times.values).astype("datetime64[s]")
        total_len = len(time_values)

        chunk_size = total_len // num
        remainder = total_len % num

        detectors: list[SimpleDetector] = []
        for i in range(num):
            s_idx = i * chunk_size + min(i, remainder)
            e_idx = (i + 1) * chunk_size + min(i + 1, remainder)

            if s_idx >= e_idx:
                continue

            detectors.append(
                SimpleDetector(
                    self.pathname,
                    self.requested_varname,
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
        threshold: float = 0.0,
        time_chunk_size: int = 360,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[RawDetectionStep]:
        if size % 2 != 1:
            raise ValueError("size must be an odd number")

        time_array = self.get_time()
        lat, lon = self.lat, self.lon
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

            extrema = _numba_extrema_filter(filled_frame, size, threshold, is_min)

            if np.isnan(frame).any():
                extrema[np.isnan(frame)] = 0

            laplacian = _numba_laplace_masked(filled_frame, extrema)
            extrema = _numba_remove_dup(laplacian, size=5)

            # Extract raw data using Numba
            r_idx, c_idx, vals = _numba_get_centers(extrema, frame)
            time_val = t.astype("datetime64[s]")

            raw_results.append((time_val, lat[r_idx], lon[c_idx], {self.varname: vals}))

        return raw_results
