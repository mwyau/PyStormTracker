from __future__ import annotations

import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from .models import DetectedCenters, Grid, TimeRange, Tracks
from .simple import SimpleDetector, SimpleLinker

Backend = Literal["serial", "mpi", "dask"]


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


def _detect_serial(
    infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
) -> tuple[Tracks, Grid]:
    grid = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
    tracks = _detect_and_link(grid, size=5, threshold=0.0, time_chunk_size=360, mode=mode)
    return tracks, grid


def _detect_mpi(
    infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
) -> tuple[Tracks, Grid, MPI.Intracomm]:
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
    
    return tracks, grid, comm


def _combine_mpi_tracks(tracks: Tracks, comm: MPI.Intracomm) -> Tracks:
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
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None,
) -> tuple[Tracks, Grid]:
    import os
    import timeit
    from dask.distributed import Client, LocalCluster

    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 4

    grid_obj = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
    grids = grid_obj.split(n_workers)

    t0 = timeit.default_timer()
    # Use processes=True to ensure Xarray/HDF5 locking doesn't serialize I/O
    with LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            t1 = timeit.default_timer()
            print(f"    [Dask] Cluster setup time: {t1 - t0:.4f}s")
            
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

            final_tracks = futures[0].result()
            t2 = timeit.default_timer()
            print(f"    [Dask] Task execution & gather time: {t2 - t1:.4f}s")

    return final_tracks, grid_obj


def run_tracker(
    infile: str,
    varname: str,
    outfile: str | None,
    time_range: TimeRange | None = None,
    mode: Literal["min", "max"] = "min",
    backend: Backend = "dask",
    n_workers: int | None = None,
) -> None:
    """Orchestrates the storm tracking process."""
    timer: dict[str, float] = {}
    use_mpi = backend == "mpi"
    use_dask = backend == "dask"

    rank = 0
    if use_mpi:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        timer["worker_phase"] = timeit.default_timer()

    # Detection & Local Linking Phase
    comm: MPI.Intracomm | None = None
    if use_mpi:
        tracks, _grid, comm = _detect_mpi(infile, varname, time_range, mode)
    elif use_dask:
        tracks, _grid = _detect_dask(infile, varname, time_range, mode, n_workers)
    else:
        tracks, _grid = _detect_serial(infile, varname, time_range, mode)

    if use_mpi and comm is not None:
        comm.Barrier()

    if rank == 0:
        timer["worker_phase"] = timeit.default_timer() - timer["worker_phase"]

    # MPI Consolidation Phase (Dask is already reduced in _detect_dask)
    if use_mpi and comm is not None:
        timer["combiner"] = timeit.default_timer()
        tracks = _combine_mpi_tracks(tracks, comm)
        timer["combiner"] = timeit.default_timer() - timer["combiner"]

    # Export Phase
    if rank == 0:
        print(f"Worker phase (Detect + Local Link + Dask Reduce): {timer['worker_phase']:.4f}s")
        if outfile is None:
            print("Skipping linker metrics and export as outfile is None.")
            return

        if "combiner" in timer:
            print(f"MPI Combiner time: {timer['combiner']:.4f}s")

        num_tracks = len(
            [t for t in tracks if len(t) >= 8 and t[0].abs_dist(t[-1]) >= 1000.0]
        )
        print(f"Number of long tracks (>= 8 steps, >= 1000km): {num_tracks}")

        from .io.imilast import write_imilast
        write_imilast(tracks, outfile)
        out_path = Path(outfile)

        final_outfile = (
            out_path if out_path.suffix == ".txt" else out_path.with_suffix(".txt")
        )
        print(f"Results exported to {final_outfile}")


def parse_args() -> Namespace:
    """Parses command line arguments."""
    parser = ArgumentParser(description="PyStormTracker: A tool for tracking storms.")
    parser.add_argument("-i", "--input", required=True, help="Input NetCDF file.")
    parser.add_argument("-v", "--var", required=True, help="Variable to track.")
    parser.add_argument(
        "-o", "--output", required=True, help="Output track file (.txt)."
    )
    parser.add_argument("-n", "--num", type=int, help="Number of time steps.")
    parser.add_argument(
        "-m", "--mode", choices=["min", "max"], default="min", help="Detection mode."
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["serial", "mpi", "dask"],
        default="dask",
        help="Parallel backend. Default is 'dask'.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of workers for Dask. Defaults to number of CPU cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_range: TimeRange | None = None

    if args.num is not None:
        # Determine actual times for the first n steps
        grid_preview = SimpleDetector(pathname=args.input, varname=args.var)
        times = grid_preview.get_time()
        assert times is not None
        num = min(args.num, len(times))
        time_range = TimeRange(
            start=times[0],
            end=times[num - 1],
        )

    run_tracker(
        infile=args.input,
        varname=args.var,
        outfile=args.output,
        time_range=time_range,
        mode=args.mode,
        backend=args.backend,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
