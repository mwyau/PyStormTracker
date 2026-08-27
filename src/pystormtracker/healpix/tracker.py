from __future__ import annotations

import logging
import timeit
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..backends import (
    Backend,
    extract_dask_frame_delayed_blocks,
    local_dask_executor,
    resolve_dask_workers,
    validate_execution_parameters,
)
from ..hodges import constants
from ..hodges.linker import HodgesLinker
from ..hodges.segments import (
    DEFAULT_SEGMENT_FRAMES,
    merge_segments,
    plan_tracking_segments,
)
from ..io.data_loader import DataLoader, normalize_tracking_data
from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.time import TimeInput, select_time_range
from ..models.tracker import CenterFrame, Tracker, TrackingInput
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
from .constants import (
    DEFAULT_MSL_OBJECT_THRESHOLD,
    DEFAULT_VO_OBJECT_THRESHOLD,
    SPECTRAL_TAPER_DEFAULT,
)
from .detector import HealpixDetector, HealpixFeatureRefinement, detect_healpix_frame

LOGGER = logging.getLogger(__name__)


def _detect_healpix_frame_task(
    frame_arr: NDArray[np.float64],
    time_val: TimeInput,
    neighbor_table: NDArray[np.int64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    object_threshold: float,
    mode: ResolvedDetectionMode,
    min_object_grid_points: int,
    feature_refinement: HealpixFeatureRefinement,
) -> CenterFrame:
    """Worker task: Detect and refine features for a single HEALPix frame."""
    frame_arr_1d = np.asarray(frame_arr, dtype=np.float64).squeeze()
    return detect_healpix_frame(
        frame_arr_1d,
        time_val,
        neighbor_table,
        lat,
        lon,
        object_threshold=object_threshold,
        mode=mode,
        min_object_grid_points=min_object_grid_points,
        feature_refinement=feature_refinement,
    )


def _link_healpix_segment_task(
    detections: list[CenterFrame],
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
    )
    return linker.link(
        detections,
        primary_variable=primary_variable,
        mode=mode,
        bounds=bounds,
        unit=unit,
        processing=processing,
    )


def _track_healpix_segment_task(
    data: xr.DataArray,
    *,
    primary_variable: str,
    mode: ResolvedDetectionMode,
    bounds: SpatialBounds | None,
    unit: str | None,
    processing: tuple[ProcessingStep, ...],
    object_threshold: float | None,
    min_object_grid_points: int,
    feature_refinement: HealpixFeatureRefinement,
    w1: float,
    w2: float,
    dmax: float,
    phimax: float,
    mge_max_iterations: int,
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
) -> Tracks:
    """Worker task: Detect and link one independent HEALPix segment (unfiltered)."""
    detector = HealpixDetector.from_xarray(data, variable_name=primary_variable)
    detections = detector.detect(
        object_threshold=object_threshold,
        detection_mode=mode,
        min_object_grid_points=min_object_grid_points,
        feature_refinement=feature_refinement,
    )
    return _link_healpix_segment_task(
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
    )


class HealpixTracker(Tracker):
    """Apply the Hodges-style tracker to a native HEALPix grid.

    HEALPix is the underlying grid of Górski et al. (2005),
    https://doi.org/10.1086/427976.  The threshold/object/local-feature
    detector is a PyStormTracker adaptation of Hodges-style processing to
    HEALPix topology, followed by the shared Hodges/TRACK-compatible MGE
    linker where configured.  The spherical quadratic refinement option is a
    PyStormTracker extension, not a TRACK algorithm.
    """

    def __init__(
        self,
        *,
        nside: int | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        spectral_taper: float = SPECTRAL_TAPER_DEFAULT,
        min_object_grid_points: int = constants.MIN_OBJECT_GRID_POINTS_DEFAULT,
        feature_refinement: HealpixFeatureRefinement = "quadratic",
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
        backend: Backend = "dask",
        workers: int | None = None,
    ) -> None:
        validate_execution_parameters(backend, workers, segment_frames=segment_frames)

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
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")
        if not 0.0 < spectral_taper <= 1.0:
            raise ValueError("spectral_taper must be in the interval (0, 1]")
        if nside is not None and (nside <= 0 or (nside & (nside - 1)) != 0):
            raise ValueError("nside must be a positive power of two")
        if feature_refinement not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; "
                "expected 'grid' or 'quadratic'"
            )
        resolve_filter_bounds(lmin, lmax)

        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.mge_max_iterations = mge_max_iterations
        self.min_track_points = min_track_points
        self.nside = nside
        self.lmin = lmin
        self.lmax = lmax
        self.taper_points = taper_points
        self.spectral_taper = spectral_taper
        self.min_object_grid_points = min_object_grid_points
        self.feature_refinement = feature_refinement
        self.backend = backend
        self.workers = workers
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

    def _preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        spectral_taper: float = SPECTRAL_TAPER_DEFAULT,
        nside: int | None = None,
        backend: Backend | None = None,
    ) -> tuple[xr.DataArray, tuple[ProcessingStep, ...]]:
        return preprocess_tracking_data(
            data,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            spectral_taper=spectral_taper,
            projection="healpix",
            nside=nside,
            backend=backend or self.backend,
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
    ) -> Tracks:
        return _track_healpix_segment_task(
            segment_data,
            primary_variable=primary_variable,
            mode=mode,
            bounds=bounds,
            unit=unit,
            processing=processing,
            object_threshold=threshold,
            min_object_grid_points=self.min_object_grid_points,
            feature_refinement=self.feature_refinement,
            w1=self.w1,
            w2=self.w2,
            dmax=self.dmax,
            phimax=self.phimax,
            mge_max_iterations=self.mge_max_iterations,
            dmax_zones=self.dmax_zones,
            adaptive_smoothness=self.adaptive_smoothness,
        )

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        detection_mode: DetectionMode = "auto",
        object_threshold: float | None = None,
        engine: str | None = None,
        **kwargs: object,
    ) -> Tracks:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                "HealpixTracker.track() got unexpected keyword argument(s): "
                f"{unexpected}"
            )

        resolved_mode = resolve_mode(variable, detection_mode)
        t_total_start = timeit.default_timer()
        LOGGER.info(
            "HEALPix tracking started: backend=%s variable=%s mode=%s",
            self.backend,
            variable,
            resolved_mode,
        )
        LOGGER.debug(
            "HEALPix execution configuration: workers=%r nside=%r "
            "refinement=%s lmin=%r lmax=%r",
            self.workers,
            self.nside,
            self.feature_refinement,
            self.lmin,
            self.lmax,
        )

        if self.backend == "mpi":
            tracks = self._track_mpi(
                data,
                primary_variable=variable,
                mode=resolved_mode,
                start_time=start_time,
                end_time=end_time,
                threshold=object_threshold,
                engine=engine,
            )
            t_total_end = timeit.default_timer()
            LOGGER.info(
                "HEALPix tracking completed in %.4fs", t_total_end - t_total_start
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
            nside=self.nside,
            backend=self.backend,
        )
        t1 = timeit.default_timer()
        if self.backend == "dask":
            LOGGER.info("HEALPix Dask preprocessing graph prepared in %.4fs", t1 - t0)
        else:
            LOGGER.info("HEALPix preprocessing completed in %.4fs", t1 - t0)

        if self.backend == "serial":
            tracks = self._track_serial(
                data_xr,
                primary_variable=variable,
                mode=resolved_mode,
                bounds=bounds,
                unit=stored_unit,
                processing=processing,
                threshold=object_threshold,
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
            )
        else:
            raise ValueError(f"unsupported backend {self.backend!r}")

        t_total_end = timeit.default_timer()
        LOGGER.info("HEALPix tracking completed in %.4fs", t_total_end - t_total_start)
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
    ) -> Tracks:
        time_dim = DataLoader(data_xr).get_coords()[0]
        n_steps = int(data_xr.sizes[time_dim])
        segments = plan_tracking_segments(n_steps, self.segment_frames, overlap=2)

        detections = self._detect_frames(
            data_xr,
            primary_variable=primary_variable,
            mode=mode,
            threshold=threshold,
        )

        if len(segments) == 1:
            tracks = _link_healpix_segment_task(
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
            )
        else:
            segment_results: list[Tracks] = []
            for seg in segments:
                seg_detections = detections[seg.start : seg.stop]
                seg_tracks = _link_healpix_segment_task(
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
    ) -> list[CenterFrame]:
        detector = HealpixDetector.from_xarray(data, variable_name=primary_variable)
        return detector.detect(
            object_threshold=threshold,
            detection_mode=mode,
            min_object_grid_points=self.min_object_grid_points,
            feature_refinement=self.feature_refinement,
        )

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
    ) -> Tracks:
        import dask

        frames = extract_dask_frame_delayed_blocks(data_xr)
        segments = plan_tracking_segments(
            frames.n_steps, self.segment_frames, overlap=2
        )
        LOGGER.debug(
            "HEALPix Dask graph: frames=%d segments=%d",
            frames.n_steps,
            len(segments),
        )

        workers = resolve_dask_workers(self.workers)

        if threshold is None:
            if primary_variable == "vo":
                threshold = DEFAULT_VO_OBJECT_THRESHOLD
            else:
                threshold = DEFAULT_MSL_OBJECT_THRESHOLD

        detector_proto = HealpixDetector.from_xarray(
            data_xr, variable_name=primary_variable
        )
        assert detector_proto._neighbor_table is not None
        assert detector_proto._lat is not None
        assert detector_proto._lon is not None

        neighbor_table = detector_proto._neighbor_table
        lat = detector_proto._lat
        lon = detector_proto._lon

        frame_tasks = [
            dask.delayed(_detect_healpix_frame_task)(
                frames.frame_blocks[i],
                time_val=frames.times[i],
                neighbor_table=neighbor_table,
                lat=lat,
                lon=lon,
                object_threshold=threshold,
                mode=mode,
                min_object_grid_points=self.min_object_grid_points,
                feature_refinement=self.feature_refinement,
            )
            for i in range(frames.n_steps)
        ]

        segment_tasks = [
            dask.delayed(_link_healpix_segment_task)(
                [frame_tasks[i] for i in range(seg.start, seg.stop)],
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
            )
            for seg in segments
        ]

        LOGGER.info(
            "HEALPix Dask detection/MGE graph prepared for %d frames and %d segments",
            frames.n_steps,
            len(segments),
        )
        with local_dask_executor(workers):
            computed_segments: list[Tracks] = list(
                dask.compute(  # type: ignore[no-untyped-call]
                    *segment_tasks,
                    scheduler="threads",
                )
            )

        if len(segments) == 1:
            tracks = computed_segments[0]
        else:
            tracks = merge_segments(computed_segments, segments)

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
        global_bounds = (
            bounds if bounds is not None else spatial_bounds_from_xarray(selected_var)
        )
        stored_unit = unit
        stored_processing = processing

        for s_idx, seg in assigned_segments:
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
                nside=self.nside,
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
            )
            local_results.append((s_idx, raw_tr))

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

        assert all_results_gathered is not None

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
