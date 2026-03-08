from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from scipy.ndimage import generic_filter, laplace

from ..models.center import Center
from ..models.grid import Grid
from ..models.time import TimeRange


class SimpleDetector:
    """
    A meteorological feature detector that treats fields as 2D images.
    Uses xarray for robust coordinate handling and lazy-loading.
    """

    def __init__(
        self,
        pathname: str | Path,
        varname: str,
        time_range: TimeRange | None = None,
    ) -> None:
        self.pathname = Path(pathname)
        self.varname = varname
        self.time_range = time_range

        self._ds: xr.Dataset | None = None
        self._data: xr.DataArray | None = None

    def _ensure_open(self) -> None:
        """Ensures the xarray dataset is open and basic variables are mapped."""
        if self._ds is None:
            # open_dataset is lazy by default
            self._ds = xr.open_dataset(self.pathname, mask_and_scale=False)
            self._data = self._ds[self.varname]

            # Map coordinates to standard names if needed
            if "latitude" not in self._ds.coords and "lat" in self._ds.coords:
                self._ds = self._ds.rename({"lat": "latitude"})
            if "longitude" not in self._ds.coords and "lon" in self._ds.coords:
                self._ds = self._ds.rename({"lon": "longitude"})
            if "valid_time" in self._ds.coords and "time" not in self._ds.coords:
                self._ds = self._ds.rename({"valid_time": "time"})

    @property
    def lat(self) -> np.ndarray:
        self._ensure_open()
        assert self._ds is not None
        return np.asarray(self._ds.latitude.values)

    @property
    def lon(self) -> np.ndarray:
        self._ensure_open()
        assert self._ds is not None
        return np.asarray(self._ds.longitude.values)

    def get_var(self, chart: int | tuple[int, int] | None = None) -> np.ndarray | None:
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

        match chart:
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
                raise TypeError("chart must be an int, tuple[int, int], or None")

    def get_time(self) -> np.ndarray | None:
        self._ensure_open()
        assert self._ds is not None
        if self.time_range:
            times = self._ds.time.sel(
                time=slice(self.time_range.start, self.time_range.end)
            )
        else:
            times = self._ds.time
        return np.asarray(times.values)

    def get_time_obj(self) -> object | None:
        self._ensure_open()
        assert self._ds is not None
        return self._ds.time  # type: ignore[no-any-return]

    def get_lat(self) -> np.ndarray | None:
        return self.lat

    def get_lon(self) -> np.ndarray | None:
        return self.lon

    def split(self, num: int) -> list[Grid]:
        if self._ds is not None:
            raise RuntimeError("Cannot split after file has been opened.")

        with xr.open_dataset(self.pathname) as ds:
            if self.time_range:
                times = ds.time.sel(
                    time=slice(self.time_range.start, self.time_range.end)
                )
            else:
                times = ds.time

            time_values = times.values
            total_len = len(time_values)

        chunk_size = total_len // num
        remainder = total_len % num

        grids: list[Grid] = []
        for i in range(num):
            s_idx = i * chunk_size + min(i, remainder)
            e_idx = (i + 1) * chunk_size + min(i + 1, remainder)

            # Use actual time values for the sub-ranges
            grids.append(
                SimpleDetector(
                    self.pathname,
                    self.varname,
                    time_range=TimeRange(
                        start=time_values[s_idx], end=time_values[e_idx - 1]
                    ),
                )
            )
        return grids

    def _local_extrema_func(
        self,
        buffer: np.ndarray,
        size: int,
        threshold: float,
        minmaxmode: Literal["min", "max"],
    ) -> bool:
        center_val = buffer[(size * size) // 2]

        if np.isnan(center_val) or np.isinf(center_val):
            return False

        if minmaxmode == "min":
            if center_val == buffer.min():
                if threshold == 0.0:
                    return True
                # Quick check: 9th smallest value must be > center + threshold
                return bool(np.partition(buffer, 8)[8] - center_val > threshold)
            return False
        else:
            if center_val == buffer.max():
                if threshold == 0.0:
                    return True
                # Quick check: 9th largest value must be < center - threshold
                # Partition at index 8 of negated buffer finds 9th largest
                ninth_largest = -np.partition(-buffer, 8)[8]
                return bool(ninth_largest - center_val < -1.0 * threshold)
            return False

    def _local_extrema_filter(
        self,
        input_arr: np.ndarray,
        size: int,
        threshold: float = 0.0,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> np.ndarray:
        if size % 2 != 1:
            raise ValueError("size must be an odd number")

        output = generic_filter(
            input_arr,
            self._local_extrema_func,
            size=size,
            mode="wrap",
            extra_keywords={
                "size": size,
                "threshold": threshold,
                "minmaxmode": minmaxmode,
            },
        )
        half_size = size // 2
        output[:half_size, :] = 0.0
        output[-half_size:, :] = 0.0
        return np.asarray(output)

    def _remove_dup_laplace(
        self, data: np.ndarray, mask: np.ndarray, size: int = 5
    ) -> np.ndarray:
        laplacian = np.multiply(laplace(data, mode="wrap"), mask)
        return np.asarray(
            generic_filter(
                laplacian,
                lambda b: bool(b[len(b) // 2] and b[len(b) // 2] == b.max()),
                size=size,
                mode="wrap",
            )
        )

    def detect(
        self,
        size: int = 5,
        threshold: float = 0.0,
        time_chunk_size: int = 360,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[list[Center]]:
        time = self.get_time()
        lat, lon = self.lat, self.lon
        assert time is not None

        centers = []
        num_steps = len(time)

        for it, t in enumerate(time):
            ibuffer = it % time_chunk_size
            if ibuffer == 0:
                var = self.get_var(chart=(it, min(it + time_chunk_size, num_steps)))

            assert var is not None
            chart = var[ibuffer, :, :]

            # Xarray doesn't use masked arrays by default, but values can be NaN
            fill = np.inf if minmaxmode == "min" else -np.inf
            filled_chart = np.where(np.isnan(chart), fill, chart)

            extrema = self._local_extrema_filter(
                filled_chart, size, threshold=threshold, minmaxmode=minmaxmode
            )

            # Ensure we don't detect centers on originally masked pixels
            if np.ma.is_masked(chart):
                extrema[chart.mask] = 0  # type: ignore[attr-defined]
            elif np.isnan(chart).any():
                extrema[np.isnan(chart)] = 0

            extrema = self._remove_dup_laplace(filled_chart, extrema, size=5)

            # Convert datetime64 to seconds since 1970
            time_val = t.astype("datetime64[s]")

            center_list = [
                Center(time_val, float(lat[i]), float(lon[j]), float(chart[i, j]))
                for i, j in np.transpose(extrema.nonzero())
            ]
            print(f"Step {it + 1}/{num_steps}: Found {len(center_list)} centers")
            centers.append(center_list)
        return centers
