from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

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

    _ds_cache: ClassVar[dict[Path, xr.Dataset]] = {}
    _ds_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        pathname: str | Path,
        varname: str,
        time_range: TimeRange | None = None,
        global_start_idx: int = 0,
        global_total_steps: int | None = None,
    ) -> None:
        self.pathname = Path(pathname)
        self.varname = varname
        self.time_range = time_range
        self.global_start_idx = global_start_idx
        self.global_total_steps = global_total_steps

        self._ds: xr.Dataset | None = None
        self._data: xr.DataArray | None = None
        self._lat_name: str = "latitude"
        self._lon_name: str = "longitude"
        self._time_name: str = "time"

    def _ensure_open(self) -> None:
        """Ensures the xarray dataset is open and basic variables are mapped."""
        if self._ds is None:
            with self._ds_lock:
                if self.pathname not in self._ds_cache:
                    # open_dataset is lazy by default
                    # chunks={} enables dask-backed arrays for better threading
                    # Use h5netcdf engine for better thread safety if available
                    try:
                        self._ds_cache[self.pathname] = xr.open_dataset(
                            self.pathname, engine="h5netcdf", chunks={}
                        )
                    except Exception:
                        self._ds_cache[self.pathname] = xr.open_dataset(
                            self.pathname, chunks={}
                        )
                self._ds = self._ds_cache[self.pathname]
            self._data = self._ds[self.varname]

            # Identify coordinate names
            self._lat_name = "latitude" if "latitude" in self._ds.coords else "lat"
            self._lon_name = "longitude" if "longitude" in self._ds.coords else "lon"
            self._time_name = "time" if "time" in self._ds.coords else "valid_time"

    @property
    def lat(self) -> NDArray[np.float64]:
        self._ensure_open()
        assert self._ds is not None
        return np.asarray(self._ds[self._lat_name].values)

    @property
    def lon(self) -> NDArray[np.float64]:
        self._ensure_open()
        assert self._ds is not None
        return np.asarray(self._ds[self._lon_name].values)

    def get_var(
        self, frame: int | tuple[int, int] | None = None
    ) -> NDArray[np.float64] | None:
        self._ensure_open()
        assert self._data is not None

        # Base data for the configured time range
        time_dim = self._data.dims[0]
        if self.time_range:
            data_range = self._data.sel(
                {time_dim: slice(self.time_range.start, self.time_range.end)}
            )
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
        assert self._ds is not None
        if self.time_range:
            times = self._ds[self._time_name].sel(
                {self._time_name: slice(self.time_range.start, self.time_range.end)}
            )
        else:
            times = self._ds[self._time_name]
        return np.asarray(times.values)

    def get_lat(self) -> NDArray[np.float64] | None:
        return self.lat

    def get_lon(self) -> NDArray[np.float64] | None:
        return self.lon

    def split(self, num: int) -> list[SimpleDetector]:
        if self._ds is not None:
            raise RuntimeError("Cannot split after file has been opened.")

        # Open lazily using chunks={} to only read metadata
        with xr.open_dataset(self.pathname, chunks={}) as ds:
            time_name = "time" if "time" in ds.coords else "valid_time"
            time_coord = ds[time_name]
            total_len = len(time_coord)

            # Slicing if there's already a time range
            if self.time_range:
                # Find indices for the given time range
                # Use sel with nearest to be robust, but indices are preferred
                times_subset = time_coord.sel(
                    {time_name: slice(self.time_range.start, self.time_range.end)}
                )
                total_len = len(times_subset)
                # To be truly lazy, we can use the relative indices if we knew them
                # but we need some values here.
                # Let's get values only for the subset if it's already a slice.
                time_values = times_subset.values
            else:
                # No subset yet, get metadata only
                time_values = None

        chunk_size = total_len // num
        remainder = total_len % num

        detectors: list[SimpleDetector] = []
        for i in range(num):
            s_idx = i * chunk_size + min(i, remainder)
            e_idx = (i + 1) * chunk_size + min(i + 1, remainder)

            if s_idx >= e_idx:
                continue

            # Determine TimeRange for this chunk
            if time_values is not None:
                # Use already loaded subset
                s_time = time_values[s_idx]
                e_time = time_values[e_idx - 1]
            else:
                # Load only the first and last time for this chunk lazily
                with xr.open_dataset(self.pathname, chunks={}) as ds:
                    time_coord = ds[time_name]
                    s_time = time_coord[s_idx].values
                    e_time = time_coord[e_idx - 1].values

            detectors.append(
                SimpleDetector(
                    self.pathname,
                    self.varname,
                    time_range=TimeRange(start=s_time, end=e_time),
                    global_start_idx=s_idx,
                    global_total_steps=total_len,
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
        # This significantly improves I/O performance
        full_var = self.get_var()
        assert full_var is not None

        raw_results: list[RawDetectionStep] = []
        is_min = minmaxmode == "min"

        for it, t in enumerate(time_array):
            # Print progress using global indices if available
            if (it + 1) % 10 == 0 or it == 0 or it == num_steps - 1:
                if self.global_total_steps:
                    s_idx = self.global_start_idx + 1
                    e_idx = self.global_start_idx + num_steps
                    idx_range = f"{s_idx}-{e_idx}"
                    step_prog = f"Step {it + 1}/{num_steps}"
                    g_idx = self.global_start_idx + it + 1
                    glob_prog = f"Global: {g_idx}/{self.global_total_steps}"
                    print(f"  [{idx_range}] {step_prog} ({glob_prog})")
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
