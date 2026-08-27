from __future__ import annotations

import logging
import timeit
from typing import TYPE_CHECKING, Literal, cast

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
from ..io.data_loader import DataLoader, normalize_tracking_data
from ..models.geo import Projection, SpatialBounds, spatial_bounds_from_xarray
from ..models.time import (
    TimeInput,
    TimeRange,
    coerce_time_input,
    encode_time_values,
    select_time_range,
)
from ..models.tracker import (
    CenterFrame,
    Tracker,
    TrackingInput,
)
from ..models.tracks import (
    DetectionMode,
    ProcessingStep,
    ResolvedDetectionMode,
    Tracks,
    TracksMetadata,
    _TracksBuilder,
)
from ..models.units import (
    canonical_unit_for,
    normalize_variable_units,
    resolve_mode,
)
from ..preprocessing.tracking import (
    preprocess_tracking_data,
    resolve_filter_bounds,
)
from .constants import (
    DEFAULT_MSL_FEATURE_THRESHOLD,
    DEFAULT_SEARCH_WINDOW_SIZE,
    DEFAULT_VO_FEATURE_THRESHOLD,
)
from .detector import SimpleDetector, SimpleFeatureRefinement, detect_simple_frame
from .linker import SimpleLinker

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..models.geo import MapExtent


def _partition_frame_ranges(total_steps: int, world_size: int) -> list[tuple[int, int]]:
    """Partition total_steps into contiguous frame ranges across MPI ranks."""
    range_size = total_steps // world_size
    remainder = total_steps % world_size
    ranges: list[tuple[int, int]] = []
    for i in range(world_size):
        start = i * range_size + min(i, remainder)
        stop = (i + 1) * range_size + min(i + 1, remainder)
        if start < stop:
            ranges.append((start, stop))
    return ranges


def _link_centers(
    raw_steps: list[CenterFrame],
    *,
    primary_variable: str,
    mode: ResolvedDetectionMode,
    bounds: SpatialBounds | None = None,
    unit: str | None = None,
    processing: tuple[ProcessingStep, ...] = (),
) -> Tracks:
    """Sequentially link raw detection steps into a global Tracks object."""
    units = {primary_variable: unit or canonical_unit_for(primary_variable) or "1"}
    numeric_steps: list[CenterFrame] = [
        CenterFrame(
            int(encode_time_values([step[0]])[0]),
            step[1],
            step[2],
            step[3],
        )
        for step in raw_steps
    ]
    builder = _TracksBuilder(
        TracksMetadata(primary_variable, mode, units, bounds, processing)
    )
    linker = SimpleLinker()
    for step_data in numeric_steps:
        linker.append(builder, step_data)
    return builder.finish()


def _detect_with_detector(
    detector: SimpleDetector,
    search_window_size: int,
    threshold: float | None,
    detection_mode: ResolvedDetectionMode,
    feature_refinement: SimpleFeatureRefinement = "quadratic",
) -> list[CenterFrame]:
    """Worker task: Detect centers and return raw results for central linking."""
    return detector.detect(
        search_window_size=search_window_size,
        intensity_threshold=threshold,
        detection_mode=detection_mode,
        feature_refinement=feature_refinement,
    )


def _convert_stereo_steps(
    raw_steps: list[CenterFrame],
    projection: Projection,
) -> list[CenterFrame]:
    """Convert projected detection coordinates back to latitude and longitude."""
    if projection not in ("nh_stereo", "sh_stereo"):
        return raw_steps

    from ..models.geo import stereo_to_latlon

    hemisphere = 1 if projection == "nh_stereo" else -1
    converted: list[CenterFrame] = []
    for time_coord, y, x, values in raw_steps:
        lats = np.empty_like(y)
        lons = np.empty_like(x)
        for i in range(len(y)):
            lats[i], lons[i] = stereo_to_latlon(x[i], y[i], hemisphere)
        converted.append(CenterFrame(time_coord, lats, lons, values))
    return converted


def _detect_simple_frame_task(
    frame_arr: NDArray[np.float64],
    time_val: TimeInput,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    *,
    search_window_size: int,
    threshold: float | None,
    mode: ResolvedDetectionMode = "min",
    feature_refinement: SimpleFeatureRefinement = "quadratic",
    periodic_x: bool = True,
    primary_variable: str = "msl",
) -> CenterFrame:
    """Worker task: Detect and refine centers for a single frame."""
    frame_arr_2d = np.asarray(frame_arr, dtype=np.float64).squeeze()

    if threshold is None:
        if primary_variable == "vo":
            threshold = DEFAULT_VO_FEATURE_THRESHOLD
        else:
            threshold = DEFAULT_MSL_FEATURE_THRESHOLD

    return detect_simple_frame(
        frame_arr_2d,
        time_val,
        lat,
        lon,
        intensity_threshold=threshold,
        mode=mode,
        search_window_size=search_window_size,
        feature_refinement=feature_refinement,
        periodic_x=periodic_x,
    )


class SimpleTracker(Tracker):
    """A tracker implementing the PyStormTracker simple algorithm.

    The high-level tracker concept follows Yau and Chang (2020): closest
    features are linked within 500 km over consecutive 6-hourly periods.  The
    current detector/linker rules and the parallel execution path are
    PyStormTracker behavior; they are not presented as an exact reproduction
    of the paper.
    """

    def __init__(
        self,
        *,
        projection: Projection = "global",
        stereo_grid_spacing_km: float = 100.0,
        extent: MapExtent | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        search_window_size: int = DEFAULT_SEARCH_WINDOW_SIZE,
        feature_refinement: SimpleFeatureRefinement = "grid",
        backend: Backend = "dask",
        workers: int | None = None,
    ) -> None:
        validate_execution_parameters(backend, workers)

        if search_window_size <= 0 or search_window_size % 2 == 0:
            raise ValueError("search_window_size must be a positive odd integer")
        if stereo_grid_spacing_km <= 0.0:
            raise ValueError(
                "stereo_grid_spacing_km must be positive stereographic grid spacing "
                "in kilometres"
            )
        if projection not in ("global", "nh_stereo", "sh_stereo"):
            raise ValueError(f"unsupported projection {projection!r}")
        if feature_refinement not in ("grid", "quadratic"):
            raise ValueError(
                f"unsupported feature_refinement {feature_refinement!r}; "
                "expected 'grid' or 'quadratic'"
            )
        resolve_filter_bounds(lmin, lmax)
        if taper_points < 0:
            raise ValueError("taper_points must be nonnegative")

        self.projection = projection
        self.stereo_grid_spacing_km = stereo_grid_spacing_km
        self.extent = extent
        self.lmin = lmin
        self.lmax = lmax
        self.taper_points = taper_points
        self.search_window_size = search_window_size
        self.feature_refinement = feature_refinement
        self.backend = backend
        self.workers = workers

    def _preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
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
            projection=projection,
            nside=nside,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=extent,
            filter_type=filter_type,
            backend=backend or self.backend,
        )

    def _track_serial(
        self,
        data: TrackingInput,
        variable: str,
        time_range: TimeRange | None,
        detection_mode: ResolvedDetectionMode,
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            data,
            variable,
            start_time=time_range.start if time_range else None,
            end_time=time_range.end if time_range else None,
            engine=engine,
            backend="serial",
        )
        data_xr, threshold, variable_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=threshold,
        )
        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self._preprocess_standard_track(
            data_xr,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_points=self.taper_points,
            projection=self.projection,
            stereo_grid_spacing_km=self.stereo_grid_spacing_km,
            extent=self.extent,
            backend="serial",
        )
        t_pre = timeit.default_timer()
        LOGGER.info("Simple serial preprocessing completed in %.4fs", t_pre - t0)

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr, variable_name=variable)
        raw_steps = _detect_with_detector(
            detector,
            search_window_size=self.search_window_size,
            threshold=threshold,
            detection_mode=detection_mode,
            feature_refinement=self.feature_refinement,
        )

        raw_steps = _convert_stereo_steps(raw_steps, self.projection)
        t1 = timeit.default_timer()
        LOGGER.info("Simple serial detection completed in %.4fs", t1 - t0_detect)

        t2 = timeit.default_timer()
        tracks = _link_centers(
            raw_steps,
            primary_variable=variable,
            mode=detection_mode,
            bounds=bounds,
            unit=variable_unit,
            processing=processing,
        )
        t3 = timeit.default_timer()
        LOGGER.info("Simple serial linking completed in %.4fs", t3 - t2)
        return tracks

    def _track_dask(
        self,
        data: TrackingInput,
        variable: str,
        time_range: TimeRange | None,
        detection_mode: ResolvedDetectionMode,
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        """Dask Orchestrator: Maps single-frame detection tasks using threads."""
        import dask

        workers = resolve_dask_workers(self.workers)

        t0 = timeit.default_timer()
        data_xr = normalize_tracking_data(
            data,
            variable,
            start_time=time_range.start if time_range else None,
            end_time=time_range.end if time_range else None,
            engine=engine,
            backend="dask",
        )
        data_xr, threshold, variable_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=threshold,
        )
        bounds = spatial_bounds_from_xarray(data_xr)

        data_xr, processing = self._preprocess_standard_track(
            data_xr,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_points=self.taper_points,
            projection=self.projection,
            stereo_grid_spacing_km=self.stereo_grid_spacing_km,
            extent=self.extent,
            backend="dask",
        )

        frames = extract_dask_frame_delayed_blocks(data_xr)

        t1 = timeit.default_timer()
        LOGGER.info("Simple Dask preprocessing graph prepared in %.4fs", t1 - t0)
        LOGGER.info(
            "Simple Dask detection graph prepared for %d frames", frames.n_steps
        )

        tasks = [
            dask.delayed(_detect_simple_frame_task)(
                frames.frame_blocks[i],
                time_val=frames.times[i],
                lat=frames.lat_arr,
                lon=frames.lon_arr,
                search_window_size=self.search_window_size,
                threshold=threshold,
                mode=detection_mode,
                feature_refinement=self.feature_refinement,
                periodic_x=frames.periodic_x,
                primary_variable=variable,
            )
            for i in range(frames.n_steps)
        ]

        with local_dask_executor(workers):
            all_raw_steps: list[CenterFrame] = list(
                dask.compute(*tasks, scheduler="threads")  # type: ignore[no-untyped-call]
            )
        all_raw_steps = _convert_stereo_steps(all_raw_steps, self.projection)

        t2 = timeit.default_timer()
        LOGGER.info("Simple Dask detection completed in %.4fs", t2 - t1)

        t3 = timeit.default_timer()
        tracks = _link_centers(
            all_raw_steps,
            primary_variable=variable,
            mode=detection_mode,
            bounds=bounds,
            unit=variable_unit,
            processing=processing,
        )
        t4 = timeit.default_timer()
        LOGGER.info("Simple Dask linking completed in %.4fs", t4 - t3)
        return tracks

    def _track_mpi(
        self,
        data: TrackingInput,
        variable: str,
        time_range: TimeRange | None,
        detection_mode: ResolvedDetectionMode,
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks:
        """MPI Orchestrator: Splits frames across ranks, gathers raw detections."""
        from mpi4py import MPI

        comm: MPI.Intracomm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        root = 0

        t0 = timeit.default_timer()

        # 1. Inexpensive metadata open / selection
        if isinstance(data, xr.DataArray):
            selected_var = data
            if time_range is not None:
                selected_var = cast(
                    xr.DataArray,
                    select_time_range(
                        selected_var,
                        start_time=time_range.start,
                        end_time=time_range.end,
                    ),
                )
        else:
            loader = DataLoader(data, engine=engine)
            ds = loader.ensure_open()
            actual_name = loader.resolve_variable_name(ds, variable)
            if actual_name is None:
                raise KeyError(
                    f"Variable {variable!r} not found. Available: {list(ds.data_vars)}"
                )
            selected_var = cast(
                xr.DataArray,
                select_time_range(
                    ds[actual_name],
                    start_time=time_range.start if time_range else None,
                    end_time=time_range.end if time_range else None,
                ),
            )

        time_dim = DataLoader(selected_var).get_coords()[0]
        total_steps = int(selected_var.sizes[time_dim])
        frame_ranges = _partition_frame_ranges(total_steps, world_size)

        assigned_ranges = [
            (idx, (start, stop))
            for idx, (start, stop) in enumerate(frame_ranges)
            if idx == rank
        ]

        global_bounds = spatial_bounds_from_xarray(selected_var)
        stored_unit: str | None = None
        stored_processing: tuple[ProcessingStep, ...] = ()

        local_results: list[tuple[int, list[CenterFrame]]] = []
        for r_idx, (start, stop) in assigned_ranges:
            partition_data = selected_var.isel({time_dim: slice(start, stop)})
            partition_data, part_thresh, part_unit = normalize_variable_units(
                partition_data,
                variable=variable,
                intensity_threshold=threshold,
            )
            stored_unit = part_unit
            partition_preprocessed, part_proc = self._preprocess_standard_track(
                partition_data,
                lmin=self.lmin,
                lmax=self.lmax,
                taper_points=self.taper_points,
                projection=self.projection,
                stereo_grid_spacing_km=self.stereo_grid_spacing_km,
                extent=self.extent,
                backend="mpi",
            )
            stored_processing = part_proc
            partition_detector = SimpleDetector.from_xarray(
                partition_preprocessed,
                variable_name=variable,
            )
            raw_steps = _detect_with_detector(
                partition_detector,
                search_window_size=self.search_window_size,
                threshold=part_thresh,
                detection_mode=detection_mode,
                feature_refinement=self.feature_refinement,
            )
            local_results.append((r_idx, raw_steps))

        all_gathered = comm.gather(local_results, root=root)
        all_units = comm.gather(stored_unit, root=root)
        all_procs = comm.gather(stored_processing, root=root)
        t3 = timeit.default_timer()

        if rank == root:
            LOGGER.info("Simple MPI detection and gather completed in %.4fs", t3 - t0)
            assert all_gathered is not None
            flattened: list[tuple[int, list[CenterFrame]]] = [
                pair for rank_res in all_gathered for pair in rank_res
            ]
            flattened.sort(key=lambda x: x[0])
            all_raw_steps: list[CenterFrame] = [
                step for _, partition_steps in flattened for step in partition_steps
            ]
            all_raw_steps = _convert_stereo_steps(all_raw_steps, self.projection)
            resolved_unit = next((u for u in (all_units or []) if u is not None), "1")
            resolved_proc = next((p for p in (all_procs or []) if p), ())
            t4 = timeit.default_timer()
            tracks = _link_centers(
                all_raw_steps,
                primary_variable=variable,
                mode=detection_mode,
                bounds=global_bounds,
                unit=resolved_unit,
                processing=resolved_proc,
            )
            t5 = timeit.default_timer()
            LOGGER.info("Simple MPI linking completed in %.4fs", t5 - t4)
            return tracks

        metadata = TracksMetadata(
            primary_variable=variable,
            mode=detection_mode,
            units={variable: stored_unit or "1"},
            bounds=global_bounds,
            processing=stored_processing,
        )
        return Tracks.empty(metadata)

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        detection_mode: DetectionMode = "auto",
        feature_threshold: float | None = None,
        engine: str | None = None,
        **kwargs: object,
    ) -> Tracks:
        if kwargs:
            unexpected = ", ".join(repr(k) for k in kwargs)
            raise TypeError(
                "SimpleTracker.track() got unexpected keyword argument(s): "
                f"{unexpected}"
            )

        t0 = timeit.default_timer()
        resolved_mode = resolve_mode(variable, detection_mode)
        LOGGER.info(
            "Simple tracking started: backend=%s variable=%s mode=%s",
            self.backend,
            variable,
            resolved_mode,
        )
        LOGGER.debug(
            "Simple execution configuration: workers=%r projection=%s "
            "refinement=%s lmin=%r lmax=%r",
            self.workers,
            self.projection,
            self.feature_refinement,
            self.lmin,
            self.lmax,
        )

        time_range = None
        if start_time is not None or end_time is not None:
            st = coerce_time_input(start_time)
            et = coerce_time_input(end_time)
            time_range = TimeRange(start=st, end=et)

        data_target = (
            str(data) if not isinstance(data, (xr.DataArray, xr.Dataset)) else data
        )
        if self.backend == "mpi":
            tracks = self._track_mpi(
                data=data_target,
                variable=variable,
                time_range=time_range,
                detection_mode=resolved_mode,
                threshold=feature_threshold,
                engine=engine,
            )
        elif self.backend == "dask":
            tracks = self._track_dask(
                data=data_target,
                variable=variable,
                time_range=time_range,
                detection_mode=resolved_mode,
                threshold=feature_threshold,
                engine=engine,
            )
        else:
            tracks = self._track_serial(
                data,
                variable,
                time_range,
                resolved_mode,
                threshold=feature_threshold,
                engine=engine,
            )

        t_end = timeit.default_timer()
        rank = 0
        if self.backend == "mpi":
            try:
                from mpi4py import MPI

                rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                pass

        if rank == 0:
            LOGGER.info("Simple tracking completed in %.4fs", t_end - t0)

        return tracks
