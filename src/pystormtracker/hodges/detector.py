from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.loader import DataLoader
from ..models import constants as model_constants
from ..models.tracker import RawDetectionStep
from ..models.tracks import TimeRange
from .kernels import (
    _numba_ccl,
    _numba_get_centers,
    _numba_object_extrema,
    subgrid_refine,
)


class HodgesDetector:
    """
    Feature detector based on the Hodges (TRACK) logic.
    Identifies local extrema (min/max) within thresholded objects.
    """

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
        self.varname = varname

    def _ensure_open(self) -> None:
        if self._data is None:
            ds = self._loader.ensure_open()
            actual_var = None
            possible_names = DataLoader.VAR_MAPPING.get(
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
                    raise KeyError(f"Variable '{self.requested_varname}' not found.")
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

    def get_var(self, frame_idx: int | None = None) -> NDArray[np.float64]:
        self._ensure_open()
        assert self._data is not None
        time_dim, _, _ = self._loader.get_coords()

        if self.time_range:
            data = self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            data = self._data

        if frame_idx is not None:
            data = data.isel({time_dim: frame_idx})
            # Ensure 2D spatial frame
            return np.asarray(data.values).reshape((data.shape[-2], data.shape[-1]))

        # Ensure (time, lat, lon)
        return np.asarray(data.values).reshape(
            (data.shape[0], data.shape[-2], data.shape[-1])
        )

    def get_time(self) -> NDArray[np.datetime64]:
        self._ensure_open()
        ds = self._loader.ensure_open()
        time_dim, _, _ = self._loader.get_coords()
        if self.time_range:
            times = ds[time_dim].sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            times = ds[time_dim]
        return np.asarray(times.values).astype("datetime64[s]")

    def get_xarray(
        self,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
    ) -> xr.DataArray:
        """Returns the requested data range as an xarray DataArray."""
        self._ensure_open()
        assert self._data is not None
        time_dim, _, _ = self._loader.get_coords()

        if start_time and end_time:
            return self._data.sel({time_dim: slice(start_time, end_time)})
        elif self.time_range:
            return self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
        return self._data

    @classmethod
    def from_xarray(cls, data: xr.DataArray) -> HodgesDetector:
        """Creates a detector from an existing xarray DataArray."""
        obj = cls.__new__(cls)
        obj.requested_varname = str(data.name) if data.name else "var"
        obj.varname = obj.requested_varname
        obj._data = data
        obj._loader = DataLoader(pathname="in-memory", data=data)
        obj.pathname = Path("in-memory")
        obj.time_range = None
        obj.global_start_idx = 0
        obj.global_total_steps = None
        return obj

    def detect(
        self,
        size: int = 5,
        threshold: float | None = None,
        minmaxmode: Literal["min", "max"] = "min",
        min_points: int = 1,
    ) -> list[RawDetectionStep]:
        """
        Runs the feature detection on the selected time steps.

        Args:
            size: Diameter of local search window for extrema.
            threshold: Intensity threshold for objects.
            minmaxmode: Whether to search for local minima or maxima.
            min_points: Minimum number of grid points in an object to be processed.
        """
        if threshold is None:
            # Standard thresholds based on Hodges (1994, 1995, 1999)
            # Vo: typically 1.0e-5 for weak systems, 3.0e-5 for more intense.
            # MSL: Usually local minima search with no strict global threshold
            if self.requested_varname == "vo":
                threshold = model_constants.DEFAULT_VO_THRESHOLD
            else:
                threshold = model_constants.DEFAULT_MSL_THRESHOLD

        times = self.get_time()
        lat, lon = self.lat, self.lon
        full_var = self.get_var()
        is_min = minmaxmode == "min"
        num_steps = len(times)

        raw_results: list[RawDetectionStep] = []
        for it, t in enumerate(times):
            if (it + 1) % 10 == 0 or it == 0 or it == num_steps - 1:
                if self.global_total_steps:
                    s_idx = self.global_start_idx + it + 1
                    g_steps = self.global_total_steps
                    print(f"  Step {it + 1}/{num_steps} (Global: {s_idx}/{g_steps})")
                else:
                    print(f"  Step {it + 1}/{num_steps}")

            frame = full_var[it]

            # 1. Threshold and Segment (CCL)
            binary_mask = (
                (frame <= threshold) if is_min else (frame >= threshold)
            ).astype(np.float64)
            labeled_mask, num_objects = _numba_ccl(binary_mask)

            # 2. Find Extrema within objects
            extrema = _numba_object_extrema(
                frame, labeled_mask, num_objects, size, is_min, min_points
            )

            # 3. Extract and Refine
            r_idx, c_idx, _ = _numba_get_centers(extrema, frame)

            refined_lats = np.zeros(len(r_idx))
            refined_lons = np.zeros(len(r_idx))
            refined_vals = np.zeros(len(r_idx))

            for i in range(len(r_idx)):
                rlat, rlon, rval = subgrid_refine(frame, r_idx[i], c_idx[i], lat, lon)
                refined_lats[i] = rlat
                refined_lons[i] = rlon
                refined_vals[i] = rval

            raw_results.append(
                (t, refined_lats, refined_lons, {self.varname: refined_vals})
            )

        return raw_results
