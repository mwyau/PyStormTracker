from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.loader import DataLoader
from ..models.tracks import TimeRange
from .kernels import _numba_get_centers, _numba_hodges_extrema, subgrid_refine

RawDetectionStep: TypeAlias = tuple[
    np.datetime64,
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
]


class HodgesDetector:
    """
    Feature detector based on the Hodges (TRACK) logic.
    Identifies local extrema (min/max) and applies thresholding.
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

        return np.asarray(data.values)

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

    def detect(
        self,
        size: int = 5,
        threshold: float | None = None,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[RawDetectionStep]:
        """
        Runs the feature detection on the selected time steps.
        """
        if threshold is None:
            threshold = 1.0e-4 if self.requested_varname == "vo" else 0.0

        times = self.get_time()
        lat, lon = self.lat, self.lon
        full_var = self.get_var()
        is_min = minmaxmode == "min"

        raw_results: list[RawDetectionStep] = []
        for it, t in enumerate(times):
            frame = full_var[it]
            extrema = _numba_hodges_extrema(frame, size, threshold, is_min)
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
