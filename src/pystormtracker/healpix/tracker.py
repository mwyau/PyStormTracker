from __future__ import annotations

import timeit

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..hodges import constants
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
    preprocess_tracking_data,
    resolve_filter_bounds,
)
from .detector import HealpixDetector


def _detect_and_gather(
    detector: HealpixDetector,
    intensity_threshold: float | None,
    detection_mode: ResolvedDetectionMode,
    min_grid_points: int,
    feature_point_method: FeaturePointMethod = "quadratic",
) -> list[RawDetectionStep]:
    """Worker task: Detects centers on HEALPix and returns raw results."""
    return detector.detect(
        intensity_threshold=intensity_threshold,
        detection_mode=detection_mode,
        min_grid_points=min_grid_points,
        feature_point_method=feature_point_method,
    )


class HealpixTracker(Tracker):
    """
    A tracker specifically designed for 1D HEALPix grids.
    """

    def __init__(
        self,
        *,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        min_lifetime_steps: int = constants.LIFETIME_DEFAULT,
        max_missing_steps: int = constants.MISSING_DEFAULT,
        min_grid_points: int = constants.MIN_POINTS_DEFAULT,
        dmax_zones: NDArray[np.float64] | None = None,
        adaptive_smoothness: NDArray[np.float64] | None = None,
        nside: int | None = None,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        feature_point_method: FeaturePointMethod = "quadratic",
    ) -> None:
        if w1 < 0.0 or w2 < 0.0:
            raise ValueError("w1 and w2 must be nonnegative")
        if dmax <= 0.0:
            raise ValueError("dmax must be positive")
        if phimax < 0.0:
            raise ValueError("phimax must be nonnegative")
        if min_lifetime_steps <= 0:
            raise ValueError("min_lifetime_steps must be positive")
        if max_missing_steps < 0:
            raise ValueError("max_missing_steps must be nonnegative")
        if min_grid_points <= 0:
            raise ValueError("min_grid_points must be positive")
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")
        if nside is not None and (nside <= 0 or (nside & (nside - 1)) != 0):
            raise ValueError("nside must be a positive power of two")
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
        self.min_lifetime_steps = min_lifetime_steps
        self.max_missing_steps = max_missing_steps
        self.nside = nside
        self.filter_lmin = filter_lmin
        self.filter_lmax = filter_lmax
        self.taper_points = taper_points
        self.min_grid_points = min_grid_points
        self.feature_point_method = feature_point_method

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

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        filter_lmin: int | None = None,
        filter_lmax: int | None = None,
        taper_points: int = 0,
        nside: int | None = None,
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            filter_lmin=filter_lmin,
            filter_lmax=filter_lmax,
            taper_points=taper_points,
            projection="healpix",
            nside=nside,
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
        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable, detection_mode)

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

        bounds: SpatialBounds | None = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self.preprocess_standard_track(
            data_xr,
            filter_lmin=self.filter_lmin,
            filter_lmax=self.filter_lmax,
            taper_points=self.taper_points,
            nside=self.nside,
        )

        detector = HealpixDetector.from_xarray(data_xr, variable_name=variable)
        raw_steps = _detect_and_gather(
            detector,
            intensity_threshold=intensity_threshold,
            detection_mode=resolved_mode,
            min_grid_points=self.min_grid_points,
            feature_point_method=self.feature_point_method,
        )
        from ..hodges.linker import HodgesLinker

        linker = HodgesLinker(
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            max_missing_steps=self.max_missing_steps,
            dmax_zones=self.dmax_zones,
            adaptive_smoothness=self.adaptive_smoothness,
        )
        tracks = linker.link(
            raw_steps,
            primary_var=variable,
            mode=resolved_mode,
            bounds=bounds,
            unit=stored_unit,
            processing=processing,
        )

        t_end = timeit.default_timer()
        print(f"Total HEALPix tracking time: {t_end - t0:.4f}s")

        return tracks
