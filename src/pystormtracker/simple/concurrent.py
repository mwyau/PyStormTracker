from __future__ import annotations

import os
import timeit
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from ..models import TimeRange, Tracks
from .detector import RawDetectionStep, SimpleDetector
from .tracker import _detect_and_link, _link_centers


def run_simple_dask(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None,
    threshold: float = 0.0,
    engine: str | None = None,
) -> Tracks:
    from dask.distributed import Client, LocalCluster

    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 4

    detector_obj = SimpleDetector(
        pathname=infile, varname=varname, time_range=time_range, engine=engine
    )
    detectors = detector_obj.split(n_workers)

    t0 = timeit.default_timer()
    # Use processes=True to ensure Xarray/HDF5 locking doesn't serialize I/O
    with (
        LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
    ):
        t1 = timeit.default_timer()
        print(f"    [Dask] Cluster setup time: {t1 - t0:.4f}s")

        futures = []
        for d in detectors:
            futures.append(
                client.submit(
                    _detect_and_link,
                    d,
                    5,
                    threshold,
                    360,
                    mode,
                )
            )

        # Gather results from all workers
        all_raw_chunks: list[list[RawDetectionStep]] = client.gather(futures)

        # Flatten chunks into a single sequence of steps
        # They are gathered in the order of the futures list (sorted by split)
        all_raw_steps: list[RawDetectionStep] = [
            step for chunk in all_raw_chunks for step in chunk
        ]

        t2 = timeit.default_timer()
        print(f"    [Dask] Task execution & gather time: {t2 - t1:.4f}s")

    # Centralized linking guarantees bit-wise identity with Serial
    t3 = timeit.default_timer()
    tracks = _link_centers(all_raw_steps, time_range=time_range)
    t4 = timeit.default_timer()
    print(f"    [Dask] Linking time: {t4 - t3:.4f}s")
    return tracks


def run_simple_mpi(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    threshold: float = 0.0,
    engine: str | None = None,
) -> Tracks:
    from mpi4py import MPI

    comm: MPI.Intracomm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    t0 = timeit.default_timer()
    if rank == root:
        detector_obj = SimpleDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )
        detectors: list[SimpleDetector] | None = detector_obj.split(size)
    else:
        detectors = None

    detector: SimpleDetector = comm.scatter(detectors, root=root)
    t_scatter = timeit.default_timer()
    if rank == root:
        print(f"    [MPI] Prep & Scatter time: {t_scatter - t0:.4f}s")

    t1 = timeit.default_timer()
    raw_chunk = _detect_and_link(
        detector, size=5, threshold=threshold, time_chunk_size=360, mode=mode
    )
    # t2 = timeit.default_timer()
    # print(f"    [MPI Rank {rank}] Task execution time: {t2 - t1:.4f}s")

    # Gather all raw chunks at root
    all_raw_chunks = comm.gather(raw_chunk, root=root)
    t3 = timeit.default_timer()

    if rank == root:
        print(f"    [MPI] Detection & Gather time: {t3 - t1:.4f}s")
        assert all_raw_chunks is not None
        all_raw_steps: list[RawDetectionStep] = [
            step for chunk in all_raw_chunks for step in chunk
        ]
        t4 = timeit.default_timer()
        tracks = _link_centers(all_raw_steps, time_range=time_range)
        t5 = timeit.default_timer()
        print(f"    [MPI] Linking time: {t5 - t4:.4f}s")
        return tracks

    # Non-root ranks return empty Tracks (orchestrated by cli.py)
    return Tracks()
