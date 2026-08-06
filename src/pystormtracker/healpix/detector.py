from __future__ import annotations

from typing import Any, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import DataLoader
from ..models import constants as model_constants
from ..models.time import TimeInput, TimeRange, is_missing_time, select_time_range
from ..models.tracker import FeaturePointMethod, RawDetectionStep
from ..models.units import ResolvedDetectionMode
from .kernels import (
    _build_healpix_neighbor_table,
    _numba_get_healpix_centers,
    _numba_healpix_ccl,
    _numba_healpix_object_extrema,
    interpolate_quadratic_healpix_feature_point,
)


class HealpixDetector:
    """
    Feature detector for 1D un-nested HEALPix arrays.
    Calculates spatial neighbors on-the-fly or uses pre-built tables.
    """

    def __init__(
        self,
        pathname: str | None,
        variable_name: str,
        time_range: TimeRange | None = None,
        engine: str | None = None,
    ) -> None:
        self.pathname = pathname
        self.requested_variable_name = variable_name
        self.time_range = time_range

        self._loader = DataLoader(pathname, engine=engine) if pathname else None
        self._data: xr.DataArray | None = None
        self.variable_name = variable_name

        self._nside: int | None = None
        self._neighbor_table: NDArray[np.int64] | None = None
        self._lat: NDArray[np.float64] | None = None
        self._lon: NDArray[np.float64] | None = None

    @classmethod
    def from_xarray(
        cls, data: xr.DataArray, variable_name: str | None = None
    ) -> HealpixDetector:
        obj = cls.__new__(cls)
        obj.requested_variable_name = variable_name or (
            str(data.name) if data.name else "var"
        )
        obj.variable_name = obj.requested_variable_name
        obj._data = data
        obj._loader = DataLoader(data)
        obj.pathname = None
        obj.time_range = None
        obj._nside = None
        obj._neighbor_table = None
        obj._lat = None
        obj._lon = None
        obj._ensure_open()
        return obj

    def _ensure_open(self) -> None:
        if self._data is None and self.pathname is not None:
            self._loader = DataLoader(self.pathname)
            ds = self._loader.ensure_open()
            actual_var = None
            possible_names = DataLoader.VAR_MAPPING.get(
                self.requested_variable_name, [self.requested_variable_name]
            )
            for name in possible_names:
                if name in ds.data_vars:
                    actual_var = name
                    break
            if actual_var is None:
                raise KeyError(f"Variable {self.requested_variable_name!r} not found.")

            self.variable_name = actual_var
            self._data = ds[actual_var]

        if self._nside is None and self._data is not None:
            meta: dict[str, Any] = (
                dict(self._loader.get_grid_metadata(self.variable_name))
                if self._loader
                else {}
            )
            nside_meta = meta.get("nside")
            cell_dim = str(meta.get("cell_dim", "cell"))

            if nside_meta is not None:
                self._nside = int(str(nside_meta))
            else:
                cell_count = self._data.sizes.get("cell", self._data.shape[-1])
                self._nside = int(np.sqrt(cell_count / 12))

            if "lat" in self._data.coords and "lon" in self._data.coords:
                self._lat = np.asarray(
                    self._data.coords["lat"].values, dtype=np.float64
                )
                self._lon = np.asarray(
                    self._data.coords["lon"].values, dtype=np.float64
                )
            elif cell_dim in self._data.coords and hasattr(self._data[cell_dim], "lat"):
                self._lat = np.asarray(
                    self._data[cell_dim].lat.values, dtype=np.float64
                )
                self._lon = np.asarray(
                    self._data[cell_dim].lon.values, dtype=np.float64
                )
            else:
                import ducc0.healpix  # type: ignore[import-not-found] # ty: ignore[unresolved-import]

                hp_base = ducc0.healpix.Healpix_Base(self._nside, "RING")
                all_pix = np.arange(12 * self._nside * self._nside, dtype=np.int64)
                ang = hp_base.pix2ang(all_pix)
                self._lat = 90.0 - np.degrees(ang[:, 0])
                self._lon = np.degrees(ang[:, 1])

            self._neighbor_table = _build_healpix_neighbor_table(self._nside)

    def get_var(self, time_index: int | None = None) -> xr.DataArray | None:
        self._ensure_open()
        if self._data is None:
            return None

        if time_index is not None:
            return self._data.isel(time=time_index)

        start_time = self.time_range.start if self.time_range else None
        end_time = self.time_range.end if self.time_range else None
        if not is_missing_time(start_time) or not is_missing_time(end_time):
            return cast(
                xr.DataArray,
                select_time_range(self._data, start_time=start_time, end_time=end_time),
            )

        return self._data

    def get_time(
        self, start: TimeInput | None = None, end: TimeInput | None = None
    ) -> NDArray[np.datetime64] | None:
        self._ensure_open()
        if self._data is None:
            return None

        time_dim = DataLoader(self._data).get_coords()[0]
        if time_dim not in self._data.coords:
            return None

        ds = self._data.to_dataset()
        effective_start = (
            start
            if start is not None
            else self.time_range.start
            if self.time_range is not None
            else None
        )
        effective_end = (
            end
            if end is not None
            else self.time_range.end
            if self.time_range is not None
            else None
        )

        if not is_missing_time(effective_start) or not is_missing_time(effective_end):
            time_coord = ds[time_dim]
            if not is_missing_time(effective_start) and not is_missing_time(
                effective_end
            ):
                times = time_coord.sel(
                    {time_dim: slice(effective_start, effective_end)}
                )
            elif not is_missing_time(effective_start):
                times = time_coord.sel({time_dim: slice(effective_start, None)})
            elif not is_missing_time(effective_end):
                times = time_coord.sel({time_dim: slice(None, effective_end)})
            else:
                times = time_coord
        else:
            times = ds[time_dim]

        return np.asarray(times.values)

    def detect(
        self,
        intensity_threshold: float | None = None,
        detection_mode: ResolvedDetectionMode = "min",
        min_grid_points: int = 1,
        feature_point_method: FeaturePointMethod = "quadratic",
    ) -> list[RawDetectionStep]:
        if min_grid_points <= 0:
            raise ValueError("min_grid_points must be positive")

        use_quadratic = feature_point_method == "quadratic"

        self._ensure_open()
        times = self.get_time()
        if times is None:
            return []

        if intensity_threshold is None:
            if self.requested_variable_name == "vo":
                intensity_threshold = model_constants.DEFAULT_VO_THRESHOLD
            else:
                intensity_threshold = model_constants.DEFAULT_MSL_THRESHOLD

        raw_steps: list[RawDetectionStep] = []
        is_min = detection_mode == "min"

        for i in range(len(times)):
            current_time = times[i]
            frame = self.get_var(i)
            if frame is None:
                continue

            assert self._neighbor_table is not None
            labels, num_objects = _numba_healpix_ccl(
                frame.values, self._neighbor_table, intensity_threshold, is_min
            )

            extrema = _numba_healpix_object_extrema(
                frame.values,
                self._neighbor_table,
                labels,
                num_objects,
                is_min,
                min_grid_points,
            )

            p_idx, _ = _numba_get_healpix_centers(extrema, frame.values)

            assert self._lat is not None
            assert self._lon is not None
            assert self._nside is not None

            if use_quadratic and len(p_idx) > 0:
                refined_lats = np.empty(len(p_idx), dtype=np.float64)
                refined_lons = np.empty(len(p_idx), dtype=np.float64)
                refined_vals = np.empty(len(p_idx), dtype=np.float64)
                for idx_i, cell in enumerate(p_idx):
                    (
                        refined_lats[idx_i],
                        refined_lons[idx_i],
                        refined_vals[idx_i],
                    ) = interpolate_quadratic_healpix_feature_point(
                        frame.values,
                        cell,
                        self._neighbor_table,
                        self._lat,
                        self._lon,
                    )
            else:
                refined_lats = self._lat[p_idx]
                refined_lons = self._lon[p_idx]
                refined_vals = frame.values[p_idx]

            raw_step = RawDetectionStep(
                current_time,
                refined_lats,
                refined_lons,
                refined_vals,
            )
            raw_steps.append(raw_step)

        return raw_steps
