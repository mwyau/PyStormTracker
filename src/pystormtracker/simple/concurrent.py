from __future__ import annotations

import os
import timeit
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from ..models import TimeRange, Tracks
from .detector import SimpleDetector
from .linker import SimpleLinker
from .tracker import _detect_and_link


def _dask_extend_track(tracks1: Tracks, tracks2: Tracks) -> Tracks:
    """Worker task: Tree reduction node that merges two Tracks objects."""
    linker = SimpleLinker()
    linker.extend_track(tracks1, tracks2)
    return tracks1


def run_simple_dask(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None,
) -> Tracks:
    from dask.distributed import Client, LocalCluster

    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 4

    detector_obj = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
    detectors = detector_obj.split(n_workers)

    t0 = timeit.default_timer()
    # Use processes=True to ensure Xarray/HDF5 locking doesn't serialize I/O
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            t1 = timeit.default_timer()
            print(f"    [Dask] Cluster setup time: {t1 - t0:.4f}s")

            futures = []
            for d in detectors:
                futures.append(
                    client.submit(
                        _detect_and_link,
                        d,
                        5,
                        0.0,
                        360,
                        mode,
                    )
                )

            # Dask Tree Reduction for Linking
            while len(futures) > 1:
                next_futures = []
                for i in range(0, len(futures), 2):
                    if i + 1 < len(futures):
                        f = client.submit(_dask_extend_track, futures[i], futures[i + 1])
                        next_futures.append(f)
                    else:
                        next_futures.append(futures[i])
                futures = next_futures

            final_tracks: Tracks = futures[0].result()
            t2 = timeit.default_timer()
            print(f"    [Dask] Task execution & gather time: {t2 - t1:.4f}s")

    return final_tracks


def run_simple_mpi(
    infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
) -> Tracks:
    from mpi4py import MPI

    comm: MPI.Intracomm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    if rank == root:
        detector_obj = SimpleDetector(
            pathname=infile, varname=varname, time_range=time_range
        )
        detectors: list[SimpleDetector] | None = detector_obj.split(size)
    else:
        detectors = None

    detector: SimpleDetector = comm.scatter(detectors, root=root)
    tracks = _detect_and_link(detector, size=5, threshold=0.0, time_chunk_size=360, mode=mode)

    comm.Barrier()

    # Combiner phase
    linker = SimpleLinker()
    nstripe = 2
    while nstripe <= size:
        if rank % nstripe == nstripe // 2:
            comm.send(tracks, dest=rank - nstripe // 2, tag=nstripe)
        elif rank % nstripe == 0 and rank + nstripe // 2 < size:
            tracks_recv: Tracks = comm.recv(source=rank + nstripe // 2, tag=nstripe)
            linker.extend_track(tracks, tracks_recv)
        nstripe *= 2

    return tracks
