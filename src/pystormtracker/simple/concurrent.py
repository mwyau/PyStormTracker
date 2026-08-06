from __future__ import annotations

import os
import timeit
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..models.geo import MapExtent

from ..models.geo import SpatialBounds, spatial_bounds_from_xarray
from ..models.time import TimeRange
from ..models.tracker import FeaturePointMethod, RawDetectionStep
from ..models.tracks import ProcessingStep, Tracks, TracksMetadata
from ..models.units import ResolvedDetectionMode, normalize_variable_units
from .detector import SimpleDetector
from .tracker import _convert_stereo_steps, _detect_and_link, _link_centers


def run_simple_dask(
    data: str,
    variable: str,
    time_range: TimeRange | None,
    detection_mode: ResolvedDetectionMode,
    workers: int | None,
    chunk_size: int | None = None,
    intensity_threshold: float | None = None,
    engine: str | None = None,
    filter_lmin: int | None = None,
    filter_lmax: int | None = None,
    taper_points: int = 0,
    nside: int | None = None,
    projection: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
    stereo_grid_spacing_km: float = 100.0,
    extent: MapExtent | None = None,
    search_window_size: int = 5,
    feature_point_method: FeaturePointMethod = "grid",
) -> Tracks:
    """Dask Orchestrator: Maps detection tasks using threads."""
    import dask

    if workers is None or workers <= 0:
        workers = min(os.cpu_count() or 1, 4)

    detector_peek = SimpleDetector(
        pathname=data,
        variable_name=variable,
        time_range=time_range,
        engine=engine,
    )
    data_xr = detector_peek.get_xarray()
    data_xr, intensity_threshold, variable_unit = normalize_variable_units(
        data_xr,
        variable=variable,
        intensity_threshold=intensity_threshold,
    )
    bounds = spatial_bounds_from_xarray(data_xr)

    from .tracker import SimpleTracker

    data_xr, processing = SimpleTracker()._preprocess_standard_track(
        data_xr,
        filter_lmin=filter_lmin,
        filter_lmax=filter_lmax,
        taper_points=taper_points,
        projection=projection,
        nside=nside,
        stereo_grid_spacing_km=stereo_grid_spacing_km,
        extent=extent,
    )

    detector_obj = SimpleDetector.from_xarray(data_xr, variable_name=variable)

    times = detector_obj.get_time()
    total_steps = len(times) if times is not None else 1

    max_chunk_size = 60 if chunk_size is None or chunk_size <= 0 else max(1, chunk_size)

    n_splits = max(workers, (total_steps + max_chunk_size - 1) // max_chunk_size)
    detectors = detector_obj.split(n_splits)

    t0 = timeit.default_timer()
    t1 = timeit.default_timer()
    print(f"    [Dask] Setup time: {t1 - t0:.4f}s")
    print(
        f"    [Dask] Splitting {total_steps} steps into {n_splits} "
        f"tasks (across {workers} threads)"
    )

    tasks = [
        dask.delayed(_detect_and_link)(
            d,
            search_window_size=search_window_size,
            intensity_threshold=intensity_threshold,
            detection_mode=detection_mode,
            feature_point_method=feature_point_method,
        )
        for d in detectors
    ]

    all_raw_chunks = dask.compute(*tasks, scheduler="threads", num_workers=workers)  # type: ignore[no-untyped-call]

    all_raw_steps: list[RawDetectionStep] = [
        step for chunk in all_raw_chunks for step in chunk
    ]
    all_raw_steps = _convert_stereo_steps(all_raw_steps, projection)

    t2 = timeit.default_timer()
    print(f"    [Dask] Task execution & gather time: {t2 - t1:.4f}s")

    t3 = timeit.default_timer()
    tracks = _link_centers(
        all_raw_steps,
        primary_var=variable,
        mode=detection_mode,
        bounds=bounds,
        unit=variable_unit,
        processing=processing,
    )
    t4 = timeit.default_timer()
    print(f"    [Dask] Linking time: {t4 - t3:.4f}s")
    return tracks


def run_simple_mpi(
    data: str,
    variable: str,
    time_range: TimeRange | None,
    detection_mode: ResolvedDetectionMode,
    intensity_threshold: float | None = None,
    engine: str | None = None,
    filter_lmin: int | None = None,
    filter_lmax: int | None = None,
    taper_points: int = 0,
    nside: int | None = None,
    projection: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
    stereo_grid_spacing_km: float = 100.0,
    extent: MapExtent | None = None,
    search_window_size: int = 5,
    feature_point_method: FeaturePointMethod = "grid",
) -> Tracks:
    """MPI Orchestrator: Splits frames across ranks, gathers raw detections."""
    from mpi4py import MPI

    comm: MPI.Intracomm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    root = 0

    t0 = timeit.default_timer()
    variable_unit: str | None = None
    bounds: SpatialBounds | None = None
    processing: tuple[ProcessingStep, ...] = ()
    if rank == root:
        detector_peek = SimpleDetector(
            pathname=data,
            variable_name=variable,
            time_range=time_range,
            engine=engine,
        )
        data_xr = detector_peek.get_xarray()
        data_xr, intensity_threshold, variable_unit = normalize_variable_units(
            data_xr,
            variable=variable,
            intensity_threshold=intensity_threshold,
        )
        bounds = spatial_bounds_from_xarray(data_xr)

        from .tracker import SimpleTracker

        data_xr, processing = SimpleTracker()._preprocess_standard_track(
            data_xr,
            filter_lmin=filter_lmin,
            filter_lmax=filter_lmax,
            taper_points=taper_points,
            projection=projection,
            nside=nside,
            stereo_grid_spacing_km=stereo_grid_spacing_km,
            extent=extent,
        )

        detector_obj = SimpleDetector.from_xarray(data_xr, variable_name=variable)
        detectors: list[SimpleDetector] | None = detector_obj.split(world_size)
    else:
        detectors = None

    intensity_threshold, variable_unit, bounds, processing = comm.bcast(
        (intensity_threshold, variable_unit, bounds, processing)
        if rank == root
        else None,
        root=root,
    )
    if variable_unit is None:
        raise RuntimeError("MPI variable-unit normalization failed")
    metadata = TracksMetadata(
        primary_var=variable,
        mode=detection_mode,
        units={variable: variable_unit},
        bounds=bounds,
        processing=processing,
    )

    detector: SimpleDetector = comm.scatter(detectors, root=root)
    t_scatter = timeit.default_timer()
    if rank == root:
        print(f"    [MPI] Prep & Scatter time: {t_scatter - t0:.4f}s")

    t1 = timeit.default_timer()
    raw_chunk = _detect_and_link(
        detector,
        search_window_size=search_window_size,
        intensity_threshold=intensity_threshold,
        detection_mode=detection_mode,
        feature_point_method=feature_point_method,
    )

    all_raw_chunks = comm.gather(raw_chunk, root=root)
    t3 = timeit.default_timer()

    if rank == root:
        print(f"    [MPI] Detection & Gather time: {t3 - t1:.4f}s")
        assert all_raw_chunks is not None
        all_raw_steps: list[RawDetectionStep] = [
            step for chunk in all_raw_chunks for step in chunk
        ]
        all_raw_steps = _convert_stereo_steps(all_raw_steps, projection)
        t4 = timeit.default_timer()
        tracks = _link_centers(
            all_raw_steps,
            primary_var=variable,
            mode=detection_mode,
            bounds=bounds,
            unit=variable_unit,
            processing=processing,
        )
        t5 = timeit.default_timer()
        print(f"    [MPI] Linking time: {t5 - t4:.4f}s")
        return tracks

    return Tracks.empty(metadata)
