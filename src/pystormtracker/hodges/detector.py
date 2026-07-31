from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.interpolate import RectSphereBivariateSpline

from ..io.data_loader import DataLoader
from ..models import constants as model_constants
from ..models.tracker import RawDetectionStep
from ..models.tracks import TimeRange
from .kernels import (
    _numba_ccl,
    _numba_get_centers,
    _numba_object_extrema,
    _numba_object_properties,
)
from .kernels import subgrid_refine as refine_center


class HodgesDetector:
    """
    Feature detector based on the Hodges (TRACK) logic.
    Identifies local extrema (min/max) within thresholded objects.
    """

    def __init__(
        self,
        pathname: str | Path | None,
        varname: str,
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
        obj._loader = DataLoader(data)
        obj.pathname = None
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
        subgrid_refine: bool = True,
    ) -> list[RawDetectionStep]:
        """
        Runs the feature detection on the selected time steps.

        Args:
            size: Diameter of local search window for extrema.
            threshold: Intensity threshold for objects.
            minmaxmode: Whether to search for local minima or maxima.
            min_points: Minimum number of grid points in an object to be processed.
            subgrid_refine: Whether to refine centers with a local quadratic fit.
        """
        if threshold is None:
            if self.requested_varname == "vo":
                threshold = model_constants.DEFAULT_VO_THRESHOLD
            else:
                threshold = model_constants.DEFAULT_MSL_THRESHOLD

        times = self.get_time()
        lat, lon = self.lat, self.lon
        _, _, lon_name = self._loader.get_coords()
        projected_xy = lon_name == "x"
        periodic_x = not projected_xy and self._loader.is_global_longitude()
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

            # 1. Threshold and Segment using Connected Component Labeling (CCL).
            # This partitions the grid into discrete storm 'objects' based on intensity.
            binary_mask = (
                (frame <= threshold) if is_min else (frame >= threshold)
            ).astype(np.float64)
            labeled_mask, num_objects = _numba_ccl(binary_mask, periodic_x=periodic_x)

            # 2. Find local extrema (min/max) within each identified object.
            # This ensures we don't pick multiple points from the same noise spike.
            extrema = _numba_object_extrema(
                frame,
                labeled_mask,
                num_objects,
                size,
                is_min,
                min_points,
                periodic_x=periodic_x,
            )

            # 3. Compute physical object properties (Size in km^2, Ellipse fitting).
            # Uses a reverse flood-fill logic optimized with Numba.
            props = _numba_object_properties(
                frame,
                labeled_mask,
                num_objects,
                lat,
                lon,
                threshold,
                is_min,
                spherical_coords=not projected_xy,
            )
            raw_areas, fitted_areas, majors, minors, orientations = props

            # 4. Extract centers and perform sub-grid refinement.
            r_idx, c_idx, raw_vals = _numba_get_centers(extrema, frame)

            n_feats = len(r_idx)
            refined_lats = np.zeros(n_feats)
            refined_lons = np.zeros(n_feats)
            quad_vals = np.zeros(n_feats)
            bspline_vals = np.zeros(n_feats)
            f_raw_size = np.zeros(n_feats)
            f_fit_size = np.zeros(n_feats)
            f_major = np.zeros(n_feats)
            f_minor = np.zeros(n_feats)
            f_orient = np.zeros(n_feats)

            # Pre-calculate global spherical spline for the whole frame
            try:
                if not periodic_x:
                    raise ValueError("spherical spline requires longitude coordinates")
                theta_global = np.deg2rad(90.0 - lat)
                phi_global = np.deg2rad(lon)
                # RectSphereBivariateSpline needs theta strictly increasing
                idx_tg = np.argsort(theta_global)
                theta_sorted = theta_global[idx_tg]
                frame_sorted = frame[idx_tg, :]
                global_spline = RectSphereBivariateSpline(
                    theta_sorted, phi_global, frame_sorted
                )
            except Exception:
                global_spline = None

            for i in range(n_feats):
                if subgrid_refine:
                    rlat, rlon, rval = refine_center(
                        frame,
                        r_idx[i],
                        c_idx[i],
                        lat,
                        lon,
                        periodic_x=periodic_x,
                    )
                else:
                    rlat = lat[r_idx[i]]
                    rlon = lon[c_idx[i]]
                    rval = raw_vals[i]
                refined_lats[i] = rlat
                refined_lons[i] = rlon
                quad_vals[i] = rval

                # B-spline fit (Global Spherical Spline)
                if subgrid_refine and global_spline is not None:
                    try:
                        # evaluate at sub-grid center (convert to colatitude/rad)
                        bspline_vals[i] = float(
                            global_spline(np.deg2rad(90.0 - rlat), np.deg2rad(rlon))[
                                0, 0
                            ]
                        )
                    except Exception:
                        bspline_vals[i] = rval
                else:
                    bspline_vals[i] = rval

                # Attach object properties
                obj_id = labeled_mask[r_idx[i], c_idx[i]]
                f_raw_size[i] = raw_areas[obj_id]
                f_fit_size[i] = fitted_areas[obj_id]
                f_major[i] = majors[obj_id]
                f_minor[i] = minors[obj_id]
                f_orient[i] = orientations[obj_id]

            raw_results.append(
                (
                    t,
                    refined_lats,
                    refined_lons,
                    {
                        self.varname: quad_vals,
                        "raw_val": raw_vals,
                        "quad_val": quad_vals,
                        "bspline_val": bspline_vals,
                        "raw_size_km2": f_raw_size,
                        "fitted_size_km2": f_fit_size,
                        "major_axis_km": f_major,
                        "minor_axis_km": f_minor,
                        "orientation_deg": f_orient,
                    },
                )
            )

        return raw_results
