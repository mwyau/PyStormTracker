from __future__ import annotations

import logging
import timeit
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..backends import (
    Backend,
    configure_sht_threads,
    extract_dask_frame_delayed_blocks,
    local_dask_executor,
    resolve_frame_workers,
    resolve_mge_workers,
    resolve_sht_threads,
    validate_execution_parameters,
)
from ..io.data_loader import DataLoader, normalize_tracking_data
from ..models.geo import Projection, SpatialBounds, spatial_bounds_from_xarray
from ..models.time import TimeInput, select_time_range
from ..models.tracker import Tracker, TrackingInput
from ..models.tracks import (
    DetectionMode,
    ProcessingStep,
    ResolvedDetectionMode,
    Tracks,
    TracksMetadata,
)
from ..models.units import normalize_variable_units, resolve_mode
from ..preprocessing.tracking import (
    preprocess_tracking_data,
    resolve_filter_bounds,
)
from . import constants
from .detections import HodgesCenterFrame
from .detector import (
    DEFAULT_SEARCH_WINDOW_SIZE,
    HodgesDetector,
    HodgesFeatureRefinement,
    detect_hodges_frame,
)
from .linker import HodgesLinker
from .progress import HodgesDaskProgress, hodges_dask_progress_enabled
from .segments import DEFAULT_SEGMENT_FRAMES, merge_segments, plan_tracking_segments

if TYPE_CHECKING:
    from ..models.geo import MapExtent


LOGGER = logging.getLogger(__name__)

__all__ = ["HodgesFeatureRefinement", "HodgesTracker"]


def _mark_hodges_prepared_frame(frame: object) -> object:
    """Mark one lazy preprocessed frame without changing its data."""
    return frame


def _detect_hodges_frame_task(
    frame_arr: NDArray[np.float64],
    time_val: TimeInput,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    intensity_threshold: float,
    mode: ResolvedDetectionMode,
    search_window_size: int,
    min_grid_points: int,
    feature_refinement: HodgesFeatureRefinement,
    group_adjacent_extrema: bool,
    exclude_boundary_extrema: bool,
    bspline_smoothing: float,
    track_smoopy_optimization_scale: float,
    bspline_max_iterations: int,
    bspline_gradient_tolerance: float,
    periodic_x: bool,
    projected_xy: bool,
    projection: Projection,
) -> HodgesCenterFrame:
    """Worker task: Detect and refine features for a single frame."""
    frame_arr_2d = np.asarray(frame_arr, dtype=np.float64).squeeze()

    step, _ = detect_hodges_frame(
        frame_arr_2d,
        time_val,
        lat,
        lon,
        intensity_threshold=intensity_threshold,
        mode=mode,
        search_window_size=search_window_size,
        min_grid_points=min_grid_points,
        feature_refinement=feature_refinement,
        group_adjacent_extrema=group_adjacent_extrema,
        exclude_boundary_extrema=exclude_boundary_extrema,
        bspline_smoothing=bspline_smoothing,
        track_smoopy_optimization_scale=track_smoopy_optimization_scale,
        bspline_max_iterations=bspline_max_iterations,
        bspline_gradient_tolerance=bspline_gradient_tolerance,
        periodic_x=periodic_x,
        projected_xy=projected_xy,
    )

    if projection in ("nh_stereo", "sh_stereo"):
        from ..models.geo import stereo_to_latlon

        hemi = 1 if projection == "nh_stereo" else -1
        lats = step.latitudes
        lons = step.longitudes
        new_lats = np.zeros_like(lats)
        new_lons = np.zeros_like(lons)
        for i in range(len(lats)):
            plat, plon = stereo_to_latlon(lons[i], lats[i], hemi)
            new_lats[i] = plat
            new_lons[i] = plon
        step = HodgesCenterFrame(
            step.time,
            new_lats,
            new_lons,
            step.values,
            step.diagnostics,
            step.diagnostic_units,
        )

    return step


def _link_hodges_segment_task(
    detections: list[HodgesCenterFrame],
    *,
    primary_variable: str,
    mode: ResolvedDetectionMode,
    bounds: SpatialBounds | None,
    unit: str | None,
    processing: tuple[ProcessingStep, ...],
    w1: float,
    w2: float,
    dmax: float,
    phimax: float,
    mge_max_iterations: int,
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
    missing_frame_parameters: NDArray[np.float64],
    time_step_ms: int | None,
) -> Tracks:
    """Worker task: Link detections within a temporal segment using MGE."""
    linker = HodgesLinker(
        w1=w1,
        w2=w2,
        dmax=dmax,
        phimax=phimax,
        mge_max_iterations=mge_max_iterations,
        dmax_zones=dmax_zones,
        adaptive_smoothness=adaptive_smoothness,
        missing_frame_parameters=missing_frame_parameters,
        time_step_ms=time_step_ms,
    )
    return linker.link(
        detections,
        primary_variable=primary_variable,
        mode=mode,
        bounds=bounds,
        unit=unit,
        processing=processing,
    )


def _track_hodges_segment_task(
    data: xr.DataArray,
    *,
    primary_variable: str,
    mode: ResolvedDetectionMode,
    bounds: SpatialBounds | None,
    unit: str | None,
    processing: tuple[ProcessingStep, ...],
    threshold: float | None,
    search_window_size: int,
    min_grid_points: int,
    feature_refinement: HodgesFeatureRefinement,
    group_adjacent_extrema: bool,
    exclude_boundary_extrema: bool,
    bspline_smoothing: float,
    track_smoopy_optimization_scale: float,
    bspline_max_iterations: int,
    bspline_gradient_tolerance: float,
    w1: float,
    w2: float,
    dmax: float,
    phimax: float,
    mge_max_iterations: int,
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
    missing_frame_parameters: NDArray[np.float64],
    time_step_ms: int | None,
) -> Tracks:
    """Worker task: Detect and link one independent Hodges segment (unfiltered)."""
    detector = HodgesDetector.from_xarray(data, variable_name=primary_variable)
    detections = detector.detect(
        search_window_size=search_window_size,
        intensity_threshold=threshold,
        detection_mode=mode,
        min_grid_points=min_grid_points,
        feature_refinement=feature_refinement,
        group_adjacent_extrema=group_adjacent_extrema,
        exclude_boundary_extrema=exclude_boundary_extrema,
        bspline_smoothing=bspline_smoothing,
        track_smoopy_optimization_scale=track_smoopy_optimization_scale,
        bspline_max_iterations=bspline_max_iterations,
        bspline_gradient_tolerance=bspline_gradient_tolerance,
    )
    detections = [
        step if isinstance(step, HodgesCenterFrame) else HodgesCenterFrame(*step)
        for step in detections
    ]

    projection = data.attrs.get("projection", "global")
    if projection in ("nh_stereo", "sh_stereo"):
        from ..models.geo import stereo_to_latlon

        hemi = 1 if projection == "nh_stereo" else -1
        converted_detections: list[HodgesCenterFrame] = []
        for step in detections:
            lats = step.latitudes
            lons = step.longitudes
            new_lats = np.zeros_like(lats)
            new_lons = np.zeros_like(lons)
            for i in range(len(lats)):
                plat, plon = stereo_to_latlon(lons[i], lats[i], hemi)
                new_lats[i] = plat
                new_lons[i] = plon
            converted_detections.append(
                HodgesCenterFrame(
                    step.time,
                    new_lats,
                    new_lons,
                    step.values,
                    step.diagnostics,
                    step.diagnostic_units,
                )
            )
        detections = converted_detections

    return _link_hodges_segment_task(
        detections,
        primary_variable=primary_variable,
        mode=mode,
        bounds=bounds,
        unit=unit,
        processing=processing,
        w1=w1,
        w2=w2,
        dmax=dmax,
        phimax=phimax,
        mge_max_iterations=mge_max_iterations,
        dmax_zones=dmax_zones,
        adaptive_smoothness=adaptive_smoothness,
        missing_frame_parameters=missing_frame_parameters,
        time_step_ms=time_step_ms,
    )


class HodgesTracker(Tracker):
    """Track features with the Hodges method and TRACK 1.5.4 semantics.

    Hodges (1994, 1995) establishes the feature-tracking methodology, while
    Hodges (1999) establishes the regional upper-bound displacement and
    adaptive track-smoothness lineage.  Exact implementation behavior is
    attributed to the immutable TRACK 1.5.4 source map, including the
    detector, SMOOPY/GDFP, and MGE source paths; the PST packed-data and
    execution layers remain engineering adaptations.

    References:
        https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2
        https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2
        https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2
        https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/mge_tracks.c
    """

    def __init__(
        self,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        spectral_taper: float = constants.SPECTRAL_TAPER_DEFAULT,
        min_object_grid_points: int = constants.MIN_OBJECT_GRID_POINTS_DEFAULT,
        feature_refinement: HodgesFeatureRefinement = "bspline",
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        mge_max_iterations: int = constants.MGE_MAX_ITERATIONS_DEFAULT,
        min_track_points: int = constants.MIN_TRACK_POINTS_DEFAULT,
        dmax_zones: NDArray[np.float64] | None = None,
        adaptive_smoothness: NDArray[np.float64] | None = None,
        missing_frame_parameters: NDArray[np.float64] | None = None,
        segment_frames: int | None = DEFAULT_SEGMENT_FRAMES,
        projection: Projection = "global",
        stereo_grid_spacing_km: float = 100.0,
        extent: MapExtent | None = None,
        group_adjacent_extrema: bool = False,
        exclude_boundary_extrema: bool = False,
        track_smoopy_optimization_scale: float = (
            constants.TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT
        ),
        backend: Backend = "dask",
        frame_workers: int | None = None,
        sht_threads: int | None = None,
        mge_workers: int | None = None,
        **kwargs: object,
    ) -> None:
        """
        Initialize the Hodges Tracker.

        ``segment_frames`` is the scientific MGE temporal segment length.

        ``missing_frame_parameters`` provides TRACK-style ``(dmax, phimax)``
        rows selected by the count of known missing source input frames.
        """
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                f"HodgesTracker() got unexpected keyword argument(s): {unexpected}"
            )

        validate_execution_parameters(
            backend,
            segment_frames=segment_frames,
            frame_workers=frame_workers,
            sht_threads=sht_threads,
            mge_workers=mge_workers,
        )

        if w1 < 0.0 or w2 < 0.0:
            raise ValueError("w1 and w2 must be nonnegative")
        if dmax <= 0.0:
            raise ValueError("dmax must be positive")
        if phimax < 0.0:
            raise ValueError("phimax must be nonnegative")
        if mge_max_iterations <= 0:
            raise ValueError("mge_max_iterations must be positive")
        if min_track_points <= 0:
            raise ValueError("min_track_points must be positive")
        if min_object_grid_points <= 0:
            raise ValueError("min_object_grid_points must be positive")
        if track_smoopy_optimization_scale <= 0.0 or not np.isfinite(
            track_smoopy_optimization_scale
        ):
            raise ValueError(
                "track_smoopy_optimization_scale must be finite and positive"
            )
        if stereo_grid_spacing_km <= 0.0:
            raise ValueError(
                "stereo_grid_spacing_km must be positive stereographic grid spacing "
                "in kilometres"
            )
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")
        if not 0.0 < spectral_taper <= 1.0:
            raise ValueError("spectral_taper must be in the interval (0, 1]")
        if projection not in ("global", "nh_stereo", "sh_stereo"):
            raise ValueError(f"unsupported projection {projection!r}")
        if feature_refinement not in (
            "grid",
            "quadratic",
            "spherical_quadratic",
            "bspline",
            "spherical_bspline",
        ):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; "
                "expected 'grid', 'quadratic', 'spherical_quadratic', "
                "'bspline', or 'spherical_bspline'"
            )
        if projection in ("nh_stereo", "sh_stereo") and feature_refinement in (
            "spherical_quadratic",
            "spherical_bspline",
        ):
            raise ValueError(
                "spherical refinements require a global periodic longitude grid; "
                "use bspline or quadratic for regional data"
            )
        if group_adjacent_extrema and feature_refinement != "grid":
            raise ValueError(
                "group_adjacent_extrema requires feature_refinement='grid'"
            )
        resolve_filter_bounds(lmin, lmax)

        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.mge_max_iterations = mge_max_iterations
        self.min_track_points = min_track_points
        self.projection = projection
        self.stereo_grid_spacing_km = stereo_grid_spacing_km
        self.extent = extent
        self.lmin = lmin
        self.lmax = lmax
        self.taper_points = taper_points
        self.spectral_taper = spectral_taper
        self.search_window_size = DEFAULT_SEARCH_WINDOW_SIZE
        self.min_object_grid_points = min_object_grid_points
        self.feature_refinement = feature_refinement
        self.group_adjacent_extrema = group_adjacent_extrema
        self.exclude_boundary_extrema = exclude_boundary_extrema
        self.bspline_smoothing = 0.0
        self.track_smoopy_optimization_scale = track_smoopy_optimization_scale
        self.bspline_max_iterations = 100
        self.bspline_gradient_tolerance = 1.0e-5
        self.backend = backend
        self.frame_workers = frame_workers
        self.sht_threads = sht_threads
        self.mge_workers = mge_workers
        self.segment_frames = segment_frames

        if dmax_zones is None:
            self.dmax_zones = constants.DEFAULT_DMAX_ZONES.copy()
        else:
            self.dmax_zones = np.asarray(dmax_zones, dtype=np.float64)
        HodgesLinker._validate_dmax_zones(self.dmax_zones)
        if self.dmax_zones.shape[0] > 0:
            self.dmax = float(np.max(self.dmax_zones[:, 4]))

        if adaptive_smoothness is None:
            if self.phimax > 0:
                self.adaptive_smoothness = constants.DEFAULT_ADAPTIVE_SMOOTHNESS.copy()
            else:
                self.adaptive_smoothness = np.zeros((2, 0), dtype=np.float64)
        else:
            self.adaptive_smoothness = np.asarray(adaptive_smoothness, dtype=np.float64)
        HodgesLinker._validate_adaptive_smoothness(self.adaptive_smoothness)
        if self.adaptive_smoothness.shape[1] == 4:
            self.phimax = max(
                self.phimax,
                float(np.max(self.adaptive_smoothness[1])),
            )

        self.missing_frame_parameters = HodgesLinker._validate_missing_frame_parameters(
            missing_frame_parameters,
            dmax=dmax,
            phimax=phimax,
        )
        if self.missing_frame_parameters.shape[0] > 1 and (
            self.dmax_zones.shape[0] > 0 or self.adaptive_smoothness.shape[1] > 0
        ):
            raise ValueError(
                "multiple missing-frame parameter sets require dmax_zones and "
                "adaptive_smoothness to be disabled; per-parameter zone and "
                "adaptive tables are not implemented"
            )
        if self.adaptive_smoothness.shape[1] == 4:
            self.missing_frame_parameters = self.missing_frame_parameters.copy()
            self.missing_frame_parameters[:, 1] = np.maximum(
                self.missing_frame_parameters[:, 1],
                np.max(self.adaptive_smoothness[1]),
            )
        if self.dmax_zones.shape[0] == 0:
            self.dmax = float(self.missing_frame_parameters[0, 0])
        self.phimax = max(self.phimax, float(self.missing_frame_parameters[0, 1]))

    @staticmethod
    def _time_step_ms(time_step: np.timedelta64 | None) -> int | None:
        """Validate an optional expected source cadence at millisecond precision."""
        if time_step is None:
            return None
        milliseconds = time_step.astype("timedelta64[ms]")
        if milliseconds.astype(time_step.dtype) != time_step:
            raise ValueError("time_step must have millisecond precision")
        cadence_ms = int(milliseconds.astype(np.int64))
        if cadence_ms <= 0:
            raise ValueError("time_step must be positive")
        return cadence_ms

    def _preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        spectral_taper: float | None = None,
        projection: Projection = "global",
        nside: int | None = None,
        stereo_grid_spacing_km: float | None = 100.0,
        extent: MapExtent | None = None,
        filter_type: Literal["sht", "dct", "auto"] = "auto",
        backend: Backend | None = None,
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            spectral_taper=(
                spectral_taper if spectral_taper is not None else self.spectral_taper
            ),
            projection=projection,
            nside=nside,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=extent,
            filter_type=filter_type,
            backend=backend or self.backend,
            sht_threads=self.sht_threads,
        )

    def _run_segment_task(
        self,
        segment_data: xr.DataArray,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None,
        unit: str | None,
        processing: tuple[ProcessingStep, ...],
        threshold: float | None,
        time_step_ms: int | None = None,
    ) -> Tracks:
        return _track_hodges_segment_task(
            segment_data,
            primary_variable=primary_variable,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
            threshold=threshold,
            search_window_size=self.search_window_size,
            min_grid_points=self.min_object_grid_points,
            feature_refinement=self.feature_refinement,
            group_adjacent_extrema=self.group_adjacent_extrema,
            exclude_boundary_extrema=self.exclude_boundary_extrema,
            bspline_smoothing=self.bspline_smoothing,
            track_smoopy_optimization_scale=self.track_smoopy_optimization_scale,
            bspline_max_iterations=self.bspline_max_iterations,
            bspline_gradient_tolerance=self.bspline_gradient_tolerance,
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            mge_max_iterations=self.mge_max_iterations,
            dmax_zones=self.dmax_zones,
            adaptive_smoothness=self.adaptive_smoothness,
            missing_frame_parameters=self.missing_frame_parameters,
            time_step_ms=time_step_ms,
        )

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        time_step: np.timedelta64 | None = None,
        detection_mode: DetectionMode = "auto",
        object_threshold: float | None = None,
        engine: str | None = None,
        **kwargs: object,
    ) -> Tracks:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                "HodgesTracker.track() got unexpected keyword argument(s): "
                f"{unexpected}"
            )

        if self.missing_frame_parameters.shape[0] > 1 and time_step is None:
            raise ValueError(
                "time_step is required with multiple missing-frame parameter sets"
            )

        time_step_ms = self._time_step_ms(time_step)
        resolved_mode = resolve_mode(variable, detection_mode)
        t_total_start = timeit.default_timer()
        LOGGER.info(
            "Hodges tracking started: backend=%s variable=%s mode=%s",
            self.backend,
            variable,
            resolved_mode,
        )
        LOGGER.debug(
            "Hodges execution configuration: backend=%s "
            "frame_workers=%d sht_threads=%d mge_workers=%d "
            "segment_frames=%r refinement=%s lmin=%r lmax=%r dmax=%g phimax=%g",
            self.backend,
            resolve_frame_workers(self.frame_workers, self.backend),
            resolve_sht_threads(self.sht_threads, self.backend),
            resolve_mge_workers(self.mge_workers, self.backend),
            self.segment_frames,
            self.feature_refinement,
            self.lmin,
            self.lmax,
            self.dmax,
            self.phimax,
        )

        if self.backend == "mpi":
            tracks = self._track_mpi(
                data,
                primary_variable=variable,
                mode=resolved_mode,
                start_time=start_time,
                end_time=end_time,
                threshold=object_threshold,
                time_step_ms=time_step_ms,
                engine=engine,
            )
            t_total_end = timeit.default_timer()
            LOGGER.info(
                "Hodges tracking completed in %.4fs", t_total_end - t_total_start
            )
            return tracks

        # 1. Load and optionally filter data
        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            data,
            variable,
            start_time=start_time,
            end_time=end_time,
            engine=engine,
            backend=self.backend,
        )
        data_xr, object_threshold, stored_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=object_threshold,
        )

        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self._preprocess_standard_track(
            data_xr,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_points=self.taper_points,
            spectral_taper=self.spectral_taper,
            projection=self.projection,
            stereo_grid_spacing_km=self.stereo_grid_spacing_km,
            extent=self.extent,
            backend=self.backend,
        )
        t1 = timeit.default_timer()
        if self.backend == "dask":
            LOGGER.info(
                "Hodges Dask preprocessing graph prepared in %.4fs",
                t1 - t0,
            )
        else:
            LOGGER.info("Hodges preprocessing completed in %.4fs", t1 - t0)

        if self.backend == "serial":
            tracks = self._track_serial(
                data_xr,
                primary_variable=variable,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
                threshold=object_threshold,
                time_step_ms=time_step_ms,
            )
        elif self.backend == "dask":
            tracks = self._track_dask(
                data_xr,
                primary_variable=variable,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
                threshold=object_threshold,
                time_step_ms=time_step_ms,
            )
        else:
            raise ValueError(f"unsupported backend {self.backend!r}")

        t_total_end = timeit.default_timer()
        LOGGER.info("Hodges tracking completed in %.4fs", t_total_end - t_total_start)
        return tracks

    def _track_serial(
        self,
        data_xr: xr.DataArray,
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None,
        unit: str | None,
        processing: tuple[ProcessingStep, ...],
        threshold: float | None,
        time_step_ms: int | None = None,
    ) -> Tracks:
        time_dim = DataLoader(data_xr).get_coords()[0]
        n_steps = int(data_xr.sizes[time_dim])
        segments = plan_tracking_segments(n_steps, self.segment_frames, overlap=2)
        LOGGER.debug(
            "Hodges serial segment plan: frames=%d segments=%s",
            n_steps,
            [(segment.start, segment.stop) for segment in segments],
        )

        detections = self._detect_frames(
            data_xr,
            primary_variable=primary_variable,
            mode=mode,
            threshold=threshold,
        )
        LOGGER.info("Hodges serial detection/refinement completed")

        projection = data_xr.attrs.get("projection", self.projection)
        if projection in ("nh_stereo", "sh_stereo"):
            from ..models.geo import stereo_to_latlon

            hemi = 1 if projection == "nh_stereo" else -1
            converted_detections: list[HodgesCenterFrame] = []
            for step in detections:
                lats = step.latitudes
                lons = step.longitudes
                new_lats = np.zeros_like(lats)
                new_lons = np.zeros_like(lons)
                for i in range(len(lats)):
                    plat, plon = stereo_to_latlon(lons[i], lats[i], hemi)
                    new_lats[i] = plat
                    new_lons[i] = plon
                converted_detections.append(
                    HodgesCenterFrame(
                        step.time,
                        new_lats,
                        new_lons,
                        step.values,
                        step.diagnostics,
                        step.diagnostic_units,
                    )
                )
            detections = converted_detections

        if len(segments) == 1:
            tracks = _link_hodges_segment_task(
                detections,
                primary_variable=primary_variable,
                mode=mode,
                bounds=bounds,
                unit=unit,
                processing=processing,
                w1=self.w1,
                w2=self.w2,
                dmax=self.dmax,
                phimax=self.phimax,
                mge_max_iterations=self.mge_max_iterations,
                dmax_zones=self.dmax_zones,
                adaptive_smoothness=self.adaptive_smoothness,
                missing_frame_parameters=self.missing_frame_parameters,
                time_step_ms=time_step_ms,
            )
        else:
            segment_results: list[Tracks] = []
            for seg in segments:
                seg_detections = detections[seg.start : seg.stop]
                seg_tracks = _link_hodges_segment_task(
                    seg_detections,
                    primary_variable=primary_variable,
                    mode=mode,
                    bounds=bounds,
                    unit=unit,
                    processing=processing,
                    w1=self.w1,
                    w2=self.w2,
                    dmax=self.dmax,
                    phimax=self.phimax,
                    mge_max_iterations=self.mge_max_iterations,
                    dmax_zones=self.dmax_zones,
                    adaptive_smoothness=self.adaptive_smoothness,
                    missing_frame_parameters=self.missing_frame_parameters,
                    time_step_ms=time_step_ms,
                )
                segment_results.append(seg_tracks)
            tracks = merge_segments(segment_results, segments)

        if self.min_track_points > 1 and len(tracks) > 0:
            valid_indices = np.asarray(
                [
                    index
                    for index, tr in enumerate(tracks)
                    if len(tr) >= self.min_track_points
                ],
                dtype=np.int64,
            )
            tracks = tracks.subset(valid_indices)
        return tracks

    def _detect_frames(
        self,
        data: xr.DataArray,
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        threshold: float | None = None,
    ) -> list[HodgesCenterFrame]:
        detector = HodgesDetector.from_xarray(data, variable_name=primary_variable)
        detections = detector.detect(
            search_window_size=self.search_window_size,
            intensity_threshold=threshold,
            detection_mode=mode,
            min_grid_points=self.min_object_grid_points,
            feature_refinement=self.feature_refinement,
            group_adjacent_extrema=self.group_adjacent_extrema,
            exclude_boundary_extrema=self.exclude_boundary_extrema,
            bspline_smoothing=self.bspline_smoothing,
            track_smoopy_optimization_scale=self.track_smoopy_optimization_scale,
            bspline_max_iterations=self.bspline_max_iterations,
            bspline_gradient_tolerance=self.bspline_gradient_tolerance,
        )
        return [
            step if isinstance(step, HodgesCenterFrame) else HodgesCenterFrame(*step)
            for step in detections
        ]

    def _track_dask(
        self,
        data_xr: xr.DataArray,
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None,
        unit: str | None,
        processing: tuple[ProcessingStep, ...],
        threshold: float | None,
        time_step_ms: int | None = None,
    ) -> Tracks:
        import dask

        frame_workers = resolve_frame_workers(self.frame_workers, "dask")
        sht_threads = resolve_sht_threads(self.sht_threads, "dask")
        mge_workers = resolve_mge_workers(self.mge_workers, "dask")
        configure_sht_threads(sht_threads)

        frames = extract_dask_frame_delayed_blocks(data_xr)
        segments = plan_tracking_segments(
            frames.n_steps, self.segment_frames, overlap=2
        )
        LOGGER.debug(
            "Hodges Dask graph: frames=%d segments=%d",
            frames.n_steps,
            len(segments),
        )

        if threshold is None:
            if primary_variable == "vo":
                threshold = constants.DEFAULT_VO_OBJECT_THRESHOLD
            else:
                threshold = constants.DEFAULT_MSL_OBJECT_THRESHOLD

        projection_attr: Projection = data_xr.attrs.get("projection", self.projection)

        frame_tasks = [
            dask.delayed(_detect_hodges_frame_task)(
                dask.delayed(_mark_hodges_prepared_frame)(
                    frames.frame_blocks[i],
                    dask_key_name=f"hodges-prepared-frame-{i:06d}",
                ),
                time_val=frames.times[i],
                lat=frames.lat_arr,
                lon=frames.lon_arr,
                intensity_threshold=threshold,
                mode=mode,
                search_window_size=self.search_window_size,
                min_grid_points=self.min_object_grid_points,
                feature_refinement=self.feature_refinement,
                group_adjacent_extrema=self.group_adjacent_extrema,
                exclude_boundary_extrema=self.exclude_boundary_extrema,
                bspline_smoothing=self.bspline_smoothing,
                track_smoopy_optimization_scale=self.track_smoopy_optimization_scale,
                bspline_max_iterations=self.bspline_max_iterations,
                bspline_gradient_tolerance=self.bspline_gradient_tolerance,
                periodic_x=frames.periodic_x,
                projected_xy=frames.projected_xy,
                projection=projection_attr,
                dask_key_name=f"hodges-frame-{i:06d}",
            )
            for i in range(frames.n_steps)
        ]

        progress = (
            HodgesDaskProgress(
                total_frames=frames.n_steps,
                total_segments=len(segments),
                frame_workers=frame_workers,
                mge_workers=mge_workers,
            )
            if hodges_dask_progress_enabled()
            else None
        )
        computed_detections: list[HodgesCenterFrame]
        computed_segments: list[Tracks]
        compute_started = timeit.default_timer()
        try:
            progress_context = progress if progress is not None else nullcontext()
            with progress_context:
                with local_dask_executor(frame_workers):
                    computed_detections = list(
                        dask.compute(  # type: ignore[no-untyped-call]
                            *frame_tasks,
                            scheduler="threads",
                        )
                    )

                # MGE receives only the small immutable detection objects. Drop
                # the frame graph before creating segment tasks so filtered
                # source arrays are not retained by the linking stage.
                del frame_tasks
                del frames
                segment_tasks = [
                    dask.delayed(_link_hodges_segment_task)(
                        computed_detections[seg.start : seg.stop],
                        primary_variable=primary_variable,
                        mode=mode,
                        bounds=bounds,
                        unit=unit,
                        processing=processing,
                        w1=self.w1,
                        w2=self.w2,
                        dmax=self.dmax,
                        phimax=self.phimax,
                        mge_max_iterations=self.mge_max_iterations,
                        dmax_zones=self.dmax_zones,
                        adaptive_smoothness=self.adaptive_smoothness,
                        missing_frame_parameters=self.missing_frame_parameters,
                        time_step_ms=time_step_ms,
                        dask_key_name=f"hodges-mge-segment-{seg.index:06d}",
                    )
                    for seg in segments
                ]
                with local_dask_executor(mge_workers):
                    computed_segments = list(
                        dask.compute(  # type: ignore[no-untyped-call]
                            *segment_tasks,
                            scheduler="threads",
                        )
                    )
        except KeyboardInterrupt:
            if progress is not None:
                progress.interrupted()
            raise
        except BaseException:
            if progress is not None:
                progress.failed()
            raise
        else:
            if progress is not None:
                progress.mge_complete()
            LOGGER.info(
                "Hodges Dask frame detection/refinement and MGE completed in %.4fs "
                "(frame_workers=%d sht_threads=%d mge_workers=%d)",
                timeit.default_timer() - compute_started,
                frame_workers,
                sht_threads,
                mge_workers,
            )

        if progress is not None:
            progress.splicing_segments()
        if len(segments) == 1:
            tracks = computed_segments[0]
        else:
            tracks = merge_segments(computed_segments, segments)

        if progress is not None:
            progress.applying_postfilters()
        if self.min_track_points > 1 and len(tracks) > 0:
            valid_indices = np.asarray(
                [
                    index
                    for index, tr in enumerate(tracks)
                    if len(tr) >= self.min_track_points
                ],
                dtype=np.int64,
            )
            tracks = tracks.subset(valid_indices)
        if progress is not None:
            progress.done(len(tracks), int(tracks.times.size))
        return tracks

    def _track_mpi(
        self,
        data: TrackingInput,
        *,
        primary_variable: str,
        mode: ResolvedDetectionMode = "min",
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        bounds: SpatialBounds | None = None,
        unit: str | None = None,
        processing: tuple[ProcessingStep, ...] = (),
        threshold: float | None = None,
        time_step_ms: int | None = None,
        engine: str | None = None,
    ) -> Tracks:
        from mpi4py import MPI

        comm: MPI.Intracomm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        root = 0

        # 1. Inexpensive metadata open / selection
        if isinstance(data, xr.DataArray):
            selected_var = data
            if start_time is not None or end_time is not None:
                selected_var = cast(
                    xr.DataArray,
                    select_time_range(
                        selected_var, start_time=start_time, end_time=end_time
                    ),
                )
        else:
            loader = DataLoader(data, engine=engine)
            ds = loader.ensure_open()
            actual_name = loader.resolve_variable_name(ds, primary_variable)
            if actual_name is None:
                raise KeyError(
                    f"Variable {primary_variable!r} not found. Available: "
                    f"{list(ds.data_vars)}"
                )
            selected_var = cast(
                xr.DataArray,
                select_time_range(
                    ds[actual_name], start_time=start_time, end_time=end_time
                ),
            )

        time_dim = DataLoader(selected_var).get_coords()[0]
        n_steps = int(selected_var.sizes[time_dim])
        segments = plan_tracking_segments(n_steps, self.segment_frames, overlap=2)

        assigned_segments = [
            (s_idx, seg)
            for s_idx, seg in enumerate(segments)
            if s_idx % world_size == rank
        ]

        local_results: list[tuple[int, Tracks]] = []
        local_error: str | None = None
        global_bounds = (
            bounds if bounds is not None else spatial_bounds_from_xarray(selected_var)
        )
        stored_unit = unit
        stored_processing = processing

        for s_idx, seg in assigned_segments:
            try:
                seg_data = selected_var.isel({time_dim: slice(seg.start, seg.stop)})
                seg_data, seg_thresh, seg_unit = normalize_variable_units(
                    seg_data,
                    variable=primary_variable,
                    intensity_threshold=threshold,
                )
                stored_unit = seg_unit
                seg_preprocessed, seg_proc = self._preprocess_standard_track(
                    seg_data,
                    lmin=self.lmin,
                    lmax=self.lmax,
                    taper_points=self.taper_points,
                    spectral_taper=self.spectral_taper,
                    projection=self.projection,
                    stereo_grid_spacing_km=self.stereo_grid_spacing_km,
                    extent=self.extent,
                    backend="mpi",
                )
                stored_processing = seg_proc
                raw_tr = self._run_segment_task(
                    seg_preprocessed,
                    primary_variable=primary_variable,
                    mode=mode,
                    bounds=global_bounds,
                    unit=seg_unit,
                    processing=seg_proc,
                    threshold=seg_thresh,
                    time_step_ms=time_step_ms,
                )
                local_results.append((s_idx, raw_tr))
            except Exception as exc:  # noqa: BLE001
                source_repr = (
                    str(getattr(selected_var, "encoding", {}).get("source", "<array>"))
                    or "<array>"
                )
                local_error = (
                    f"rank {rank} segment {s_idx} "
                    f"frames [{seg.start}:{seg.stop}] "
                    f"source {source_repr!r}: {exc}"
                )
                break

        all_errors = cast(list[str | None] | None, comm.gather(local_error, root=root))
        all_results_gathered: list[list[tuple[int, Tracks]]] | None = comm.gather(
            local_results, root=root
        )

        if rank != root:
            metadata = TracksMetadata(
                primary_variable=primary_variable,
                mode=mode,
                units={primary_variable: stored_unit or "1"},
                bounds=global_bounds,
                processing=stored_processing,
            )
            return Tracks.empty(metadata)

        assert all_errors is not None
        assert all_results_gathered is not None

        failed = [msg for msg in all_errors if msg is not None]
        if failed:
            raise RuntimeError(
                f"MPI tracking failed on {len(failed)} rank(s):\n"
                + "\n".join(f"  {m}" for m in failed)
            )

        flattened: list[tuple[int, Tracks]] = [
            pair for rank_res in all_results_gathered for pair in rank_res
        ]
        flattened.sort(key=lambda x: x[0])
        ordered_segments = [pair[1] for pair in flattened]

        if len(segments) == 1:
            tracks = ordered_segments[0]
        else:
            tracks = merge_segments(ordered_segments, segments)

        if self.min_track_points > 1 and len(tracks) > 0:
            valid_indices = np.asarray(
                [
                    index
                    for index, tr in enumerate(tracks)
                    if len(tr) >= self.min_track_points
                ],
                dtype=np.int64,
            )
            tracks = tracks.subset(valid_indices)
        return tracks
