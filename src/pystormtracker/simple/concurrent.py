from __future__ import annotations

import os
from typing import Literal

import dask
import numpy as np

from ..hodges import constants
from ..models import TimeRange, Tracks
from .detector import SimpleDetector
from .tracker import _detect_and_link, _link_centers


def run_simple_mpi(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    threshold: float | None = None,
    engine: str | None = None,
    filter: bool = False,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
    taper_points: int = constants.TAPER_DEFAULT,
    **kwargs: float | int | str | None,
) -> Tracks:
    """MPI Orchestrator: Splits frames across ranks, gathers raw detections."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Prevent OpenMP oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"

    # Step 1: Initialize metadata peek
    detector_peek = SimpleDetector(
        pathname=infile, varname=varname, time_range=time_range, engine=engine
    )
    if filter:
        from .tracker import SimpleTracker

        tracker = SimpleTracker()
        data_xr = tracker.preprocess_standard_track(
            detector_peek.get_xarray(),
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
        )
    else:
        data_xr = detector_peek.get_xarray()

    time_dim = detector_peek._loader.get_coords()[0]
    full_times = data_xr[time_dim].values
    n_frames = len(full_times)

    # Parallel split
    chunk_size = n_frames // size
    remainder = n_frames % size
    s_idx = rank * chunk_size + min(rank, remainder)
    e_idx = (rank + 1) * chunk_size + min(rank + 1, remainder)

    if s_idx < e_idx:
        # Step 2: Extract local frames
        local_data = data_xr.isel({time_dim: slice(s_idx, e_idx)})
        local_detector = SimpleDetector.from_xarray(local_data)

        # Step 3: Local Detection
        window_size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
        local_raw = _detect_and_link(
            local_detector, size=window_size, threshold=threshold, mode=mode
        )
    else:
        local_raw = []

    # Step 4: Gather
    all_raw = comm.gather(local_raw, root=0)

    # Step 5: Final Link
    if rank == 0:
        flat_raw = [step for sublist in all_raw for step in sublist]  # type: ignore[union-attr]
        return _link_centers(flat_raw, time_range=detector_peek.time_range)

    return Tracks()


def run_simple_dask(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None = None,
    max_chunk_size: int | None = None,
    threshold: float | None = None,
    engine: str | None = None,
    filter: bool = False,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
    taper_points: int = constants.TAPER_DEFAULT,
    **kwargs: float | int | str | None,
) -> Tracks:
    """Dask Orchestrator: Maps detection tasks using threads."""
    # Prevent OpenMP oversubscription when using threads
    os.environ["OMP_NUM_THREADS"] = "1"

    detector_peek = SimpleDetector(
        pathname=infile, varname=varname, time_range=time_range, engine=engine
    )
    if filter:
        from .tracker import SimpleTracker

        tracker = SimpleTracker()
        data_xr = tracker.preprocess_standard_track(
            detector_peek.get_xarray(),
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
        )
    else:
        data_xr = detector_peek.get_xarray()

    time_dim = detector_peek._loader.get_coords()[0]
    n_frames = data_xr.sizes[time_dim]

    if max_chunk_size is None or max_chunk_size <= 0:
        max_chunk_size = 60

    # Ensure we at least split into tasks
    tasks = []
    start_idx = 0
    window_size = int(kwargs.get("size", 5))  # type: ignore[arg-type]

    while start_idx < n_frames:
        end_idx = min(start_idx + max_chunk_size, n_frames)
        chunk_data = data_xr.isel({time_dim: slice(start_idx, end_idx)})

        # Create delayed task
        task = dask.delayed(_detect_and_link)(
            SimpleDetector.from_xarray(chunk_data),
            window_size,
            threshold,
            mode,
        )
        tasks.append(task)
        start_idx = end_idx

    # Execute using threaded scheduler
    results = dask.compute(*tasks, scheduler="threads", num_workers=n_workers)

    # Flatten and Link
    flat_raw = [step for sublist in results for step in sublist]
    return _link_centers(flat_raw, time_range=detector_peek.time_range)
