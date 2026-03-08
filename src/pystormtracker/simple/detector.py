from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import netCDF4
import numpy as np
from scipy.ndimage import generic_filter, laplace

from ..models.center import Center
from ..models.grid import Grid
from ..models.time import TimeRange


class SimpleDetector:
    """
    A meteorological feature detector that treats fields as 2D images.
    Uses lazy-loading to minimize memory and I/O overhead.
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

        self._dataset: netCDF4.Dataset | None = None
        self._nc_var: Any = None
        self._cache: dict[str, np.ndarray] = {}

    def _ensure_open(self) -> None:
        """Ensures the NetCDF file is open and basic variables are mapped."""
        if self._dataset is None:
            self._dataset = netCDF4.Dataset(self.pathname, "r")
            self._nc_var = self._dataset.variables[self.varname]
            self._nc_var.set_auto_maskandscale(False)

    def _get_nc_var(self, name: str) -> Any:  # noqa: ANN401
        self._ensure_open()
        assert self._dataset is not None
        return self._dataset.variables[name]

    @property
    def lat(self) -> np.ndarray:
        if "lat" not in self._cache:
            names = ["latitude", "lat"]
            self._ensure_open()
            assert self._dataset is not None
            for name in names:
                if name in self._dataset.variables:
                    self._cache["lat"] = np.asarray(self._dataset.variables[name][:])
                    return self._cache["lat"]
            raise KeyError(f"Neither {names} found in {self.pathname}")
        return self._cache["lat"]

    @property
    def lon(self) -> np.ndarray:
        if "lon" not in self._cache:
            names = ["longitude", "lon"]
            self._ensure_open()
            assert self._dataset is not None
            for name in names:
                if name in self._dataset.variables:
                    self._cache["lon"] = np.asarray(self._dataset.variables[name][:])
                    return self._cache["lon"]
            raise KeyError(f"Neither {names} found in {self.pathname}")
        return self._cache["lon"]

    def get_var(self, chart: int | tuple[int, int] | None = None) -> np.ndarray | None:
        self._ensure_open()

        # SimpleDetector uses start/end as indices for temporal slicing
        tstart = int(self.time_range.start) if self.time_range else 0
        tend = int(self.time_range.end) if self.time_range else self._nc_var.shape[0]

        match chart:
            case int(idx):
                data = self._nc_var[tstart + idx, ...]
                return np.asarray(data.reshape((data.shape[-2], data.shape[-1])))

            case (int(s_off), int(e_off)):
                s, e = tstart + s_off, tstart + e_off
                data = self._nc_var[s:e, ...]
                return np.asarray(
                    data.reshape((data.shape[0], data.shape[-2], data.shape[-1]))
                )

            case None:
                data = self._nc_var[tstart:tend, ...]
                return np.asarray(
                    data.reshape((data.shape[0], data.shape[-2], data.shape[-1]))
                )

            case _:
                raise TypeError("chart must be an int, tuple[int, int], or None")

    def get_time(self) -> np.ndarray | None:
        if "time" not in self._cache:
            name = "time" if "time" in self._get_nc_vars_list() else "valid_time"
            nc_time = self._get_nc_var(name)
            tstart = int(self.time_range.start) if self.time_range else 0
            tend = int(self.time_range.end) if self.time_range else nc_time.shape[0]
            self._cache["time"] = np.asarray(nc_time[tstart:tend])
        return self._cache["time"]

    def _get_nc_vars_list(self) -> list[str]:
        self._ensure_open()
        assert self._dataset is not None
        return list(self._dataset.variables.keys())

    def get_time_obj(self) -> object | None:
        name = "time" if "time" in self._get_nc_vars_list() else "valid_time"
        return self._get_nc_var(name)  # type: ignore[no-any-return]

    def get_lat(self) -> np.ndarray | None:
        return self.lat

    def get_lon(self) -> np.ndarray | None:
        return self.lon

    def split(self, num: int) -> list[Grid]:
        if self._dataset is not None:
            raise RuntimeError("Cannot split after file has been opened.")

        with netCDF4.Dataset(self.pathname, "r") as f:
            time_dim = "time" if "time" in f.dimensions else "valid_time"
            total_len = f.dimensions[time_dim].size

        tstart = int(self.time_range.start) if self.time_range else 0
        tend = int(self.time_range.end) if self.time_range else total_len
        time_len = tend - tstart

        chunk_size = time_len // num
        remainder = time_len % num

        grids: list[Grid] = []
        for i in range(num):
            s = tstart + i * chunk_size + min(i, remainder)
            e = tstart + (i + 1) * chunk_size + min(i + 1, remainder)
            grids.append(
                SimpleDetector(
                    self.pathname,
                    self.varname,
                    time_range=TimeRange(start=float(s), end=float(e)),
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
        half_size = size // 2
        search_window = buffer.reshape((size, size))
        center_val = search_window[half_size, half_size]

        if np.ma.is_masked(center_val):
            return False

        if threshold == 0.0:
            limit = search_window.min() if minmaxmode == "min" else search_window.max()
            return bool(center_val == limit)

        if minmaxmode == "min" and center_val == search_window.min():
            return bool(sorted(buffer)[8] - center_val > threshold)
        if minmaxmode == "max" and center_val == search_window.max():
            return bool(sorted(buffer)[0] - center_val < -1 * threshold)
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
        chart_buffer: int = 400,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[list[Center]]:
        time = self.get_time()
        lat, lon = self.lat, self.lon
        assert time is not None

        centers = []
        num_steps = len(time)

        for it, t in enumerate(time):
            ibuffer = it % chart_buffer
            if ibuffer == 0:
                var = self.get_var(chart=(it, min(it + chart_buffer, num_steps)))

            assert var is not None
            chart = var[ibuffer, :, :]
            fill = np.inf if minmaxmode == "min" else -np.inf
            filled_chart = np.ma.filled(chart, fill_value=fill)

            extrema = self._local_extrema_filter(
                filled_chart, size, threshold=threshold, minmaxmode=minmaxmode
            )
            if np.ma.is_masked(chart):
                extrema[chart.mask] = 0  # type: ignore

            extrema = self._remove_dup_laplace(filled_chart, extrema, size=5)
            center_list = [
                Center(t, float(lat[i]), float(lon[j]), chart[i, j])
                for i, j in np.transpose(extrema.nonzero())
            ]
            print(f"Step {it + 1}/{num_steps}: Found {len(center_list)} centers")
            centers.append(center_list)
        return centers
