from typing import Any, Literal

import netCDF4
import numpy as np
from scipy.ndimage.filters import generic_filter, laplace

from ..models.center import Center
from ..models.grid import Grid


class SimpleDetector(Grid):
    def __init__(
        self, pathname: str, varname: str, trange: tuple[int, int] | None = None
    ) -> None:

        self.pathname = pathname
        self.varname = varname
        self.trange = trange

        self._open_file: bool = False
        self._var: Any = None
        self._time: Any = None
        self._lat: Any = None
        self._lon: Any = None

        self.f: Any = None
        self.time: Any = None
        self.lat: Any = None
        self.lon: Any = None

    def _init(self) -> None:
        if self._open_file is False:
            self._open_file = True
            self.f = netCDF4.Dataset(self.pathname, "r")

            # Dimension of var is time, lat, lon
            self._var = self.f.variables[self.varname]
            # Disable auto mask and scale as it may mask valid SLP values in some files
            self._var.set_auto_maskandscale(False)
            
            self._time = self.f.variables["time"]

            if "latitude" in self.f.variables:
                self._lat = self.f.variables["latitude"]
            elif "lat" in self.f.variables:
                self._lat = self.f.variables["lat"]
            else:
                raise KeyError(
                    "Neither 'latitude' nor 'lat' found in NetCDF variables."
                )

            if "longitude" in self.f.variables:
                self._lon = self.f.variables["longitude"]
            elif "lon" in self.f.variables:
                self._lon = self.f.variables["lon"]
            else:
                raise KeyError(
                    "Neither 'longitude' nor 'lon' found in NetCDF variables."
                )

            self.time = None
            self.lat = None
            self.lon = None

    def get_var(self, chart: int | tuple[int, int] | None = None) -> Any:

        if self.trange is not None:
            if self.trange[0] >= self.trange[1]:
                return None

        if chart is not None:
            if isinstance(chart, tuple):
                if (
                    len(chart) != 2
                    or not isinstance(chart[0], int)
                    or not isinstance(chart[1], int)
                ):
                    raise TypeError("chart must be a tuple of two integers")
            elif not isinstance(chart, int):
                raise TypeError("chart must be an integer or tuple")

            if self.trange is not None:
                if isinstance(chart, int):
                    if chart < 0 or chart >= self.trange[1] - self.trange[0]:
                        raise IndexError("chart is out of bound of trange")
                if isinstance(chart, tuple):
                    if chart[0] == chart[1]:
                        return None
                    if chart[0] > chart[1]:
                        raise IndexError("chart[1] must be larger than chart[0]")
                    if chart[0] < 0 or chart[0] > self.trange[1] - self.trange[0]:
                        raise IndexError("chart[0] is out of bound of trange")
                    if chart[1] < 0 or chart[1] > self.trange[1] - self.trange[0]:
                        raise IndexError("chart[1] is out of bound of trange")

        self._init()

        if isinstance(chart, int):
            if self.trange is None:
                return self._var[chart, :, :]
            else:
                return self._var[self.trange[0] + chart, :, :]
        elif isinstance(chart, tuple):
            if self.trange is None:
                return self._var[chart[0] : chart[1], :, :]
            else:
                return self._var[
                    self.trange[0] + chart[0] : self.trange[0] + chart[1], :, :
                ]
        else:
            return self._var[:]

    def get_time(self) -> Any:

        if self.trange is not None:
            if self.trange[0] >= self.trange[1]:
                return None

        self._init()
        if self.time is None:
            if self.trange is None:
                self.time = self._time[:]
            else:
                self.time = self._time[self.trange[0] : self.trange[1]]
        return self.time

    def get_lat(self) -> Any:

        self._init()
        if self.lat is None:
            self.lat = self._lat[:]
        return self.lat

    def get_lon(self) -> Any:

        self._init()
        if self.lon is None:
            self.lon = self._lon[:]
        return self.lon

    def split(self, num: int) -> list["Grid"]:

        if not isinstance(num, int):
            raise TypeError("number to split must be an integer")

        if self._open_file is False:
            if self.trange is not None:
                time_len = self.trange[1] - self.trange[0]
                tstart = self.trange[0]
            else:
                f = netCDF4.Dataset(self.pathname, "r")
                time_len = f.dimensions["time"].size
                f.close()
                tstart = 0

            chunk_size = time_len // num
            remainder = time_len % num

            tranges = [
                (
                    tstart + i * chunk_size + remainder * i // num,
                    tstart + (i + 1) * chunk_size + remainder * (i + 1) // num,
                )
                for i in range(num)
            ]

            return [
                SimpleDetector(self.pathname, self.varname, trange=it) for it in tranges
            ]

        else:
            raise RuntimeError(
                "SimpleDetector must not be initialized before running split()"
            )

    def _local_extrema_func(
        self,
        buffer: np.ndarray,
        size: int,
        threshold: float,
        minmaxmode: Literal["min", "max"],
    ) -> bool:

        half_size = size // 2

        search_window = buffer.reshape((size, size))
        origin = (half_size, half_size)
        
        center_val = search_window[origin]
        
        # If the center value is masked, it cannot be an extrema
        if np.ma.is_masked(center_val):
            return False

        if threshold == 0.0:
            if minmaxmode == "min":
                return bool(center_val == search_window.min())
            elif minmaxmode == "max":
                return bool(center_val == search_window.max())
        elif center_val == search_window.min():
            if minmaxmode == "min":
                # At least 8 of values in buffer should be larger than threshold
                return bool(sorted(buffer)[8] - center_val > threshold)
        elif center_val == search_window.max():
            if minmaxmode == "max":
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

        half_size = size // 2

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

        # Mask the extreme latitudes
        output[:half_size, :] = 0.0
        output[-half_size:, :] = 0.0

        return np.asarray(output)

    def _local_max_laplace(self, buffer: np.ndarray, size: int) -> bool:
        origin = (size * size) // 2
        return bool(buffer[origin] and buffer[origin] == buffer.max())

    def _remove_dup_laplace(
        self, data: np.ndarray, mask: np.ndarray, size: int = 5
    ) -> np.ndarray:
        laplacian = np.multiply(laplace(data, mode="wrap"), mask)

        return np.asarray(
            generic_filter(
                laplacian,
                self._local_max_laplace,
                size=size,
                mode="wrap",
                extra_keywords={"size": size},
            )
        )

    def detect(
        self,
        size: int = 5,
        threshold: float = 0.0,
        chart_buffer: int = 400,
        minmaxmode: Literal["min", "max"] = "min",
    ) -> list[list[Center]]:
        """Returns a list of list of Center's"""

        if self.trange is not None and self.trange[0] >= self.trange[1]:
            return []

        time = self.get_time()
        lat = self.get_lat()
        lon = self.get_lon()

        centers = []

        var: Any = None

        num_steps = len(time)
        for it, t in enumerate(time):
            ibuffer = it % chart_buffer
            if ibuffer == 0:
                var = self.get_var(chart=(it, min(it + chart_buffer, num_steps)))

            if var is not None:
                chart = var[ibuffer, :, :]
                
                # Fill masked values so they aren't detected as extrema
                if minmaxmode == "min":
                    filled_chart = np.ma.filled(chart, fill_value=np.inf)
                else:
                    filled_chart = np.ma.filled(chart, fill_value=-np.inf)

                extrema = self._local_extrema_filter(
                    filled_chart, size, threshold=threshold, minmaxmode=minmaxmode
                )
                
                # Ensure we don't detect centers on originally masked pixels
                if np.ma.is_masked(chart):
                    extrema[chart.mask] = 0

                extrema = self._remove_dup_laplace(filled_chart, extrema, size=5)

                center_list = [
                    Center(t, float(lat[i]), float(lon[j]), chart[i, j])
                    for i, j in np.transpose(extrema.nonzero())
                ]
                print(f"Step {it + 1}/{num_steps}: Found {len(center_list)} centers")
                centers.append(center_list)

        return centers
