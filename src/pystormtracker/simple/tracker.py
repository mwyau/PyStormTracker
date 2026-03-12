from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from ..models import DetectedCenters, Grid, TimeRange, Tracks
from .detector import SimpleDetector
from .linker import SimpleLinker


def _link_centers(centers: DetectedCenters) -> Tracks:
    tracks = Tracks()
    linker = SimpleLinker()
    for step_centers in centers:
        linker.append_center(tracks, step_centers)
    return tracks


def _detect_and_link(
    grid: Grid, size: int, threshold: float, time_chunk_size: int, mode: Literal["min", "max"]
) -> Tracks:
    """Worker task: Detects centers and immediately links them into a local Tracks object."""
    centers = grid.detect(
        size=size, threshold=threshold, time_chunk_size=time_chunk_size, minmaxmode=mode
    )
    return _link_centers(centers)


def _dask_extend_track(tracks1: Tracks, tracks2: Tracks) -> Tracks:
    """Worker task: Tree reduction node that merges two Tracks objects."""
    linker = SimpleLinker()
    linker.extend_track(tracks1, tracks2)
    return tracks1


class SimpleTracker:
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

    def _detect_serial(
        self, infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
    ) -> Tracks:
        grid = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
        tracks = _detect_and_link(grid, size=5, threshold=0.0, time_chunk_size=360, mode=mode)
        return tracks

    def _detect_mpi(
        self, infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
    ) -> tuple[Tracks, MPI.Intracomm]:
        from mpi4py import MPI

        comm: MPI.Intracomm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        if rank == root:
            grid_obj = SimpleDetector(
                pathname=infile, varname=varname, time_range=time_range
            )
            grids: list[Grid] | None = grid_obj.split(size)
        else:
            grids = None

        grid: Grid = comm.scatter(grids, root=root)
        tracks = _detect_and_link(grid, size=5, threshold=0.0, time_chunk_size=360, mode=mode)
        
        return tracks, comm

    def _combine_mpi_tracks(self, tracks: Tracks, comm: MPI.Intracomm) -> Tracks:
        linker = SimpleLinker()
        rank = comm.Get_rank()
        size = comm.Get_size()

        nstripe = 2
        while nstripe <= size:
            if rank % nstripe == nstripe // 2:
                comm.send(tracks, dest=rank - nstripe // 2, tag=nstripe)
            elif rank % nstripe == 0 and rank + nstripe // 2 < size:
                tracks_recv: Tracks = comm.recv(source=rank + nstripe // 2, tag=nstripe)
                linker.extend_track(tracks, tracks_recv)
            nstripe *= 2
        return tracks

    def _detect_dask(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None,
        mode: Literal["min", "max"],
        n_workers: int | None,
    ) -> Tracks:
        from dask.distributed import Client, LocalCluster

        if n_workers is None or n_workers <= 0:
            n_workers = os.cpu_count() or 4

        grid_obj = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
        grids = grid_obj.split(n_workers)

        # Use processes=True to ensure Xarray/HDF5 locking doesn't serialize I/O
        with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
            with Client(cluster) as client:
                futures = []
                for g in grids:
                    futures.append(
                        client.submit(
                            _detect_and_link,
                            g,
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

        return final_tracks

    def track(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "dask",
        n_workers: int | None = None,
    ) -> Tracks:
        use_mpi = backend == "mpi"
        use_dask = backend == "dask"

        if use_mpi:
            tracks, comm = self._detect_mpi(infile, varname, time_range, mode)
            comm.Barrier()
            tracks = self._combine_mpi_tracks(tracks, comm)
        elif use_dask:
            tracks = self._detect_dask(infile, varname, time_range, mode, n_workers)
        else:
            tracks = self._detect_serial(infile, varname, time_range, mode)
            
        return tracks
