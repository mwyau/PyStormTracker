from __future__ import annotations

import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from .models import DetectionResult, Grid, TimeRange, Tracks
from .simple import SimpleDetector, SimpleLinker

Backend = Literal["serial", "mpi", "dask"]


def _detect_serial(
    infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
) -> tuple[DetectionResult, Grid]:
    grid = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
    return (
        grid.detect(size=5, threshold=0.0, time_chunk_size=360, minmaxmode=mode),
        grid,
    )


def _detect_mpi(
    infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
) -> tuple[DetectionResult, Grid, MPI.Intracomm]:
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
    return (
        grid.detect(size=5, threshold=0.0, time_chunk_size=360, minmaxmode=mode),
        grid,
        comm,
    )


def _detect_dask(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None,
) -> tuple[DetectionResult, Grid]:
    import os

    import dask
    from dask.base import compute
    from dask.delayed import delayed

    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 4

    # Configure dask to use the threaded scheduler with n_workers
    # Alternatively, we can let dask decide, but here we want to honor n_workers
    # dask.config.set(num_workers=n_workers) is global.

    grid_obj = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
    grids = grid_obj.split(n_workers)

    delayed_results = [
        delayed(g.detect)(
            size=5, threshold=0.0, time_chunk_size=360, minmaxmode=mode
        )
        for g in grids
    ]

    with dask.config.set(num_workers=n_workers):
        results = list(compute(*delayed_results))

    # Flatten results from chunks
    flattened_results: DetectionResult = []
    for chunk in results:
        flattened_results.extend(chunk)

    return flattened_results, grid_obj


def _link_centers(centers: DetectionResult) -> Tracks:
    tracks = Tracks()
    linker = SimpleLinker()
    for step_centers in centers:
        linker.append_center(tracks, step_centers)
    return tracks


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


def run_tracker(
    infile: str,
    varname: str,
    outfile: str,
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
        timer["detector"] = timeit.default_timer()

    # Detection Phase
    comm: MPI.Intracomm | None = None
    if use_mpi:
        centers, _grid, comm = _detect_mpi(infile, varname, time_range, mode)
    elif use_dask:
        centers, _grid = _detect_dask(infile, varname, time_range, mode, n_workers)
    else:
        centers, _grid = _detect_serial(infile, varname, time_range, mode)

    if use_mpi and comm is not None:
        comm.Barrier()

    if rank == 0:
        timer["detector"] = timeit.default_timer() - timer["detector"]
        timer["linker"] = timeit.default_timer()

    # Linking Phase
    tracks = _link_centers(centers)

    # Consolidation Phase
    if use_mpi and comm is not None:
        timer["combiner"] = timeit.default_timer()
        tracks = _combine_mpi_tracks(tracks, comm)
        timer["combiner"] = timeit.default_timer() - timer["combiner"]

    # Export Phase
    if rank == 0:
        timer["linker"] = timeit.default_timer() - timer["linker"]

        print(f"Detector time: {timer['detector']:.4f}s")
        print(f"Linker time: {timer['linker']:.4f}s")
        if "combiner" in timer:
            print(f"Combiner time: {timer['combiner']:.4f}s")

        num_tracks = len(
            [t for t in tracks if len(t) >= 8 and t[0].abs_dist(t[-1]) >= 1000.0]
        )
        print(f"Number of long tracks (>= 8 steps, >= 1000km): {num_tracks}")

        tracks.to_imilast(outfile)
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
