from __future__ import annotations

import threading
from pathlib import Path
from typing import ClassVar, Literal

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.center import Center, DetectedCenters
from ..models.grid import Grid
from ..models.time import TimeRange


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_extrema_filter(
    data: NDArray[np.float64], size: int, threshold: float, is_min: bool
) -> NDArray[np.float64]:
    rows, cols = data.shape
    out = np.zeros_like(data)
    half_size = size // 2

    for r in range(half_size, rows - half_size):
        for c in range(cols):
            center_val = data[r, c]
            if np.isnan(center_val) or np.isinf(center_val):
                continue

            is_extrema = True
            for i in range(-half_size, half_size + 1):
                rr = r + i
                for j in range(-half_size, half_size + 1):
                    cc = (c + j) % cols
                    if is_min:
                        if data[rr, cc] < center_val:
                            is_extrema = False
                            break
                    else:
                        if data[rr, cc] > center_val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break

            if is_extrema:
                if threshold == 0.0:
                    out[r, c] = 1.0
                else:
                    window = np.empty(size * size, dtype=data.dtype)
                    idx = 0
                    for i in range(-half_size, half_size + 1):
                        rr = r + i
                        for j in range(-half_size, half_size + 1):
                            cc = (c + j) % cols
                            window[idx] = data[rr, cc]
                            idx += 1
                    window.sort()
                    if is_min:
                        if window[8] - center_val > threshold:
                            out[r, c] = 1.0
                    else:
                        if window[-9] - center_val < -threshold:
                            out[r, c] = 1.0

    return out


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_laplace_masked(
    data: NDArray[np.float64], mask: NDArray[np.float64]
) -> NDArray[np.float64]:
    rows, cols = data.shape
    out = np.zeros_like(data)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] != 0:
                up = data[(r - 1) % rows, c]
                down = data[(r + 1) % rows, c]
                left = data[r, (c - 1) % cols]
                right = data[r, (c + 1) % cols]
                center = data[r, c]
                out[r, c] = (up + down + left + right - 4.0 * center) * mask[r, c]
    return out


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_remove_dup(laplacian: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    rows, cols = laplacian.shape
    out = np.zeros_like(laplacian)
    half_size = size // 2

    for r in range(rows):
        for c in range(cols):
            center_val = laplacian[r, c]
            if center_val != 0:
                is_max = True
                for i in range(-half_size, half_size + 1):
                    rr = (r + i) % rows
                    for j in range(-half_size, half_size + 1):
                        cc = (c + j) % cols
                        if laplacian[rr, cc] > center_val:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max:
                    out[r, c] = 1.0
    return out


@nb.njit(nogil=True, cache=True)
def _numba_get_centers(
    extrema: NDArray[np.float64], frame: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    rows, cols = extrema.shape
    # Pre-allocate a large enough buffer to avoid double pass
    # In practice, centers are sparse.
    max_centers = 10000  # Reasonable upper bound for a single frame
    r_idx_tmp = np.empty(max_centers, dtype=np.int64)
    c_idx_tmp = np.empty(max_centers, dtype=np.int64)
    vals_tmp = np.empty(max_centers, dtype=np.float64)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if extrema[r, c] != 0:
                if count < max_centers:
                    r_idx_tmp[count] = r
                    c_idx_tmp[count] = c
                    vals_tmp[count] = frame[r, c]
                    count += 1
                else:
                    # Fallback or error if too many centers
                    # For now, just stop at max_centers
                    break
        if count >= max_centers:
            break

    return r_idx_tmp[:count].copy(), c_idx_tmp[:count].copy(), vals_tmp[:count].copy()


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
                    # chunks={} enables dask-backed arrays which handles threading better
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

    def split(self, num: int) -> list[Grid]:
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
                # Use sel with nearest to be robust, but indices are preferred for lazy loading
                times_subset = time_coord.sel(
                    {time_name: slice(self.time_range.start, self.time_range.end)}
                )
                total_len = len(times_subset)
                # To be truly lazy, we can use the relative indices if we knew them, 
                # but we need some values here.
                # Let's get the values only for the subset if it's already a slice.
                time_values = times_subset.values
            else:
                # No subset yet, get metadata only
                time_values = None

        chunk_size = total_len // num
        remainder = total_len % num

        grids: list[Grid] = []
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

            grids.append(
                SimpleDetector(
                    self.pathname,
                    self.varname,
                    time_range=TimeRange(start=s_time, end=e_time),
                    global_start_idx=s_idx,
                    global_total_steps=total_len,
                )
            )
        return grids

    def detect_raw(
        self,
        size: int = 5,
        threshold: float = 0.0,
        time_chunk_size: int = 360,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[tuple[np.datetime64, NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]]:
        if size % 2 != 1:
            raise ValueError("size must be an odd number")

        time = self.get_time()
        assert time is not None
        num_steps = len(time)
        
        # Optimization: Read the entire time range for this worker in one go
        # This significantly improves I/O performance by making a single contiguous-ish read
        full_var = self.get_var()
        assert full_var is not None

        raw_results = []
        is_min = minmaxmode == "min"

        for it, t in enumerate(time):
            # Print progress using global indices if available
            if (it + 1) % 10 == 0 or it == 0 or it == num_steps - 1:
                if self.global_total_steps:
                    print(
                        f"  [{self.global_start_idx + 1}-{self.global_start_idx + num_steps}] "
                        f"Step {it + 1}/{num_steps} (Global: {self.global_start_idx + it + 1}/{self.global_total_steps})"
                    )
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
            
            raw_results.append((time_val, r_idx, c_idx, vals))
        
        return raw_results

    def detect(
        self,
        size: int = 5,
        threshold: float = 0.0,
        time_chunk_size: int = 360,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> DetectedCenters:
        raw_results = self.detect_raw(
            size=size,
            threshold=threshold,
            time_chunk_size=time_chunk_size,
            minmaxmode=minmaxmode,
        )

        lat, lon = self.lat, self.lon
        centers = []
        for time_val, r_idx, c_idx, vals in raw_results:
            center_list = [
                Center(time_val, float(lat[r]), float(lon[c]), float(val))
                for r, c, val in zip(r_idx, c_idx, vals, strict=False)
            ]
            centers.append(center_list)
        return centers
