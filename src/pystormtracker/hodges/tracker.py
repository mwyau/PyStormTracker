from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..io.data_loader import normalize_tracking_data
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.time import TimeInput
from ..models.tracker import (
    FeaturePointMethod,
    RawDetectionStep,
    Tracker,
    TrackingInput,
)
from ..models.tracks import (
    ProcessingStep,
    Tracks,
)
from ..models.units import (
    DetectionMode,
    ResolvedDetectionMode,
    normalize_variable_units,
    resolve_mode,
)
from ..preprocessing.tracking import (
    Projection,
    preprocess_tracking_data,
    resolve_filter_bounds,
)
from . import constants
from .detector import HodgesDetector
from .linker import HodgesLinker

if TYPE_CHECKING:
    from ..models.geo import MapExtent


class HodgesTracker(Tracker):
    """
    A tracker implementing the Hodges (TRACK) algorithm with adaptive constraints.
    """

    def __init__(
        self,
        *,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        max_iterations: int = constants.MAX_ITERATIONS_DEFAULT,
        min_lifetime_steps: int = constants.LIFETIME_DEFAULT,
        max_missing_steps: int = constants.MISSING_DEFAULT,
        min_grid_points: int = constants.MIN_POINTS_DEFAULT,
        dmax_zones: NDArray[np.float64] | None = None,
        adaptive_smoothness: NDArray[np.float64] | None = None,
        projection: Projection = "global",
        stereo_grid_spacing_km: float = 100.0,
        extent: MapExtent | None = None,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        search_window_size: int = 5,
        feature_point_method: FeaturePointMethod = "quadratic",
        chunk_size: int | None = None,
    ) -> None:
        """
        Initialize the Hodges Tracker.
        """
        if w1 < 0.0 or w2 < 0.0:
            raise ValueError("w1 and w2 must be nonnegative")
        if dmax <= 0.0:
            raise ValueError("dmax must be positive")
        if phimax < 0.0:
            raise ValueError("phimax must be nonnegative")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if min_lifetime_steps <= 0:
            raise ValueError("min_lifetime_steps must be positive")
        if max_missing_steps < 0:
            raise ValueError("max_missing_steps must be nonnegative")
        if search_window_size <= 0 or search_window_size % 2 == 0:
            raise ValueError("search_window_size must be a positive odd integer")
        if min_grid_points <= 0:
            raise ValueError("min_grid_points must be positive")
        if stereo_grid_spacing_km <= 0.0:
            raise ValueError(
                "stereo_grid_spacing_km must be positive stereographic grid spacing "
                "in kilometres"
            )
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")
        if projection not in ("global", "nh_stereo", "sh_stereo", "healpix"):
            raise ValueError(f"unsupported projection {projection!r}")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if feature_point_method not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_point_method {feature_point_method!r}; "
                "expected 'grid' or 'quadratic'"
            )
        resolve_filter_bounds(filter_lmin, filter_lmax)

        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.max_iterations = max_iterations
        self.min_lifetime_steps = min_lifetime_steps
        self.max_missing_steps = max_missing_steps
        self.projection = projection
        self.stereo_grid_spacing_km = stereo_grid_spacing_km
        self.extent = extent
        self.filter_lmin = filter_lmin
        self.filter_lmax = filter_lmax
        self.taper_points = taper_points
        self.search_window_size = search_window_size
        self.min_grid_points = min_grid_points
        self.feature_point_method = feature_point_method
        self.chunk_size = chunk_size

        if dmax_zones is None:
            self.dmax_zones = constants.TRACK_ZONES
        else:
            self.dmax_zones = dmax_zones

        if adaptive_smoothness is None:
            if self.phimax > 0:
                self.adaptive_smoothness = constants.ADAPT_PARAMS
            else:
                self.adaptive_smoothness = np.zeros((2, 0), dtype=np.float64)
        else:
            self.adaptive_smoothness = adaptive_smoothness

    def _preprocess_standard_track(
        self,
        data: xr.DataArray,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        projection: Projection = "global",
        nside: int | None = None,
        stereo_grid_spacing_km: float | None = 100.0,
        extent: MapExtent | None = None,
        filter_type: Literal["sht", "dct", "auto"] = "auto",
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            filter_lmin=filter_lmin,
            filter_lmax=filter_lmax,
            taper_points=taper_points,
            projection=projection,
            nside=nside,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=extent,
            filter_type=filter_type,
        )

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        detection_mode: DetectionMode = "auto",
        intensity_threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        import timeit

        resolved_mode = resolve_mode(variable, detection_mode)
        t_total_start = timeit.default_timer()

        # 1. Load and optionally filter data
        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            data,
            variable,
            start_time=start_time,
            end_time=end_time,
            engine=engine,
        )
        data_xr, intensity_threshold, stored_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=intensity_threshold,
        )

        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self._preprocess_standard_track(
            data_xr,
            filter_lmin=self.filter_lmin,
            filter_lmax=self.filter_lmax,
            taper_points=self.taper_points,
            projection=self.projection,
            stereo_grid_spacing_km=self.stereo_grid_spacing_km,
            extent=self.extent,
        )
        t1 = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t1 - t0:.4f}s")

        if self.chunk_size is None:
            tracks = self._track_single_chunk_from_data(
                data_xr,
                primary_var=variable,
                mode=resolved_mode,
                bounds=bounds,
                threshold=intensity_threshold,
                unit=stored_unit,
                processing=processing,
            )
        else:
            from ..io.data_loader import DataLoader

            time_dim = DataLoader(data_xr).get_coords()[0]
            n_steps = data_xr.sizes[time_dim]
            detections: list[RawDetectionStep] = []

            for start_idx in range(0, n_steps, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, n_steps)
                chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})
                detections.extend(
                    self._detect_single_chunk_from_data(
                        chunk_data,
                        primary_var=variable,
                        mode=resolved_mode,
                        threshold=intensity_threshold,
                    )
                )
            tracks = self._link_detections(
                detections,
                primary_var=variable,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
            )

        t_total_end = timeit.default_timer()
        print(f"Tracking time: {t_total_end - t_total_start:.4f}s")
        return tracks

    def _track_single_chunk_from_data(
        self,
        data: xr.DataArray,
        *,
        primary_var: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None,
        unit: str | None,
        processing: tuple[ProcessingStep, ...],
        threshold: float | None = None,
    ) -> Tracks:
        detections = self._detect_single_chunk_from_data(
            data,
            primary_var=primary_var,
            mode=mode,
            threshold=threshold,
        )
        return self._link_detections(
            detections,
            primary_var=primary_var,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
        )

    def _detect_single_chunk_from_data(
        self,
        data: xr.DataArray,
        *,
        primary_var: str,
        mode: ResolvedDetectionMode,
        threshold: float | None = None,
    ) -> list[RawDetectionStep]:
        import timeit

        t_detect_start = timeit.default_timer()
        detector = HodgesDetector.from_xarray(data, variable_name=primary_var)

        detections = detector.detect(
            search_window_size=self.search_window_size,
            intensity_threshold=threshold,
            detection_mode=mode,
            min_grid_points=self.min_grid_points,
            feature_point_method=self.feature_point_method,
        )

        projection = data.attrs.get("projection", "global")
        if projection in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if projection == "nh_stereo" else -1
            converted_detections: list[RawDetectionStep] = []
            for dt, lats, lons, values in detections:
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    lat, lon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = lat
                    new_lons[i] = lon
                converted_detections.append(
                    RawDetectionStep(dt, new_lats, new_lons, values)
                )
            detections = converted_detections

        t_detect_end = timeit.default_timer()
        print(f"    [Serial] Detection time: {t_detect_end - t_detect_start:.4f}s")

        return detections

    def _link_detections(
        self,
        detections: list[RawDetectionStep],
        *,
        primary_var: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None = None,
        unit: str | None = None,
        processing: tuple[ProcessingStep, ...] = (),
    ) -> Tracks:
        import timeit

        t_link_start = timeit.default_timer()
        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            max_iterations=self.max_iterations,
            max_missing_steps=self.max_missing_steps,
            dmax_zones=self.dmax_zones,
            adaptive_smoothness=self.adaptive_smoothness,
        )

        tracks = linker.link(
            detections,
            primary_var=primary_var,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
        )
        t_link_end = timeit.default_timer()
        print(f"    [Serial] Linking time: {t_link_end - t_link_start:.4f}s")

        valid_indices = np.asarray(
            [
                index
                for index, tr in enumerate(tracks)
                if len(tr) >= self.min_lifetime_steps
            ],
            dtype=np.int64,
        )
        return tracks.subset(valid_indices)
