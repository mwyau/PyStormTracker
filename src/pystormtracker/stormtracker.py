import os
import timeit
from argparse import ArgumentParser, Namespace
from typing import Any, Literal

import netCDF4
import numpy as np

from .models import Center, Grid, Tracks
from .simple import SimpleDetector, SimpleLinker

Backend = Literal["serial", "mpi", "dask"]


def export_to_csv(
    tracks: Tracks, outfile: str, grid: Grid, decimal_places: int = 4
) -> None:
    """Exports detected tracks to a user-friendly text file."""
    time_obj = grid.get_time_obj()
    units = getattr(time_obj, "units", "")
    calendar = getattr(time_obj, "calendar", "standard")

    if not any(outfile.endswith(ext) for ext in [".txt", ".csv"]):
        outfile += ".txt"

    with open(outfile, "w", newline="") as f:
        # IMILAST Intercomparison Protocol format
        f.write(
            "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,LongE,LatN,Intensity1\n"
        )

        for i, track in enumerate(tracks, start=1):
            f.write(f"90 {i} {len(track)}\n")
            for step, center in enumerate(track, start=1):
                try:
                    # num2date returns an array-like if input is a list
                    dt_any: Any = netCDF4.num2date(
                        [center.time], units=units, calendar=calendar
                    )
                    dt = dt_any[0]
                    yyyy = f"{dt.year:04d}"
                    mm = f"{dt.month:02d}"
                    dd = f"{dt.day:02d}"
                    hh = f"{dt.hour:02d}"
                    yyyymmddhh = f"{yyyy}{mm}{dd}{hh}"
                except Exception:
                    yyyy = "0000"
                    mm = "00"
                    dd = "00"
                    hh = "00"
                    yyyymmddhh = "0000000000"

                if center.var is None or np.ma.is_masked(center.var):
                    var_val = "-999.99"
                else:
                    var_val = f"{float(center.var):.{decimal_places}f}"

                lon = center.lon
                if lon > 180:
                    lon -= 360

                f.write(
                    f"00 {i} {step} {yyyymmddhh} {dt.year} {dt.month:02d} "
                    f"{dt.day:02d} {dt.hour:02d} {lon:.2f} {center.lat:.2f} "
                    f"{var_val}\n"
                )

def _detect_serial(
    infile: str, varname: str, trange: tuple[int, int] | None, mode: str
) -> tuple[list[list[Center]], Grid]:
    grid = SimpleDetector(pathname=infile, varname=varname, trange=trange)
    return grid.detect(minmaxmode=mode), grid  # type: ignore


def _detect_mpi(
    infile: str, varname: str, trange: tuple[int, int] | None, mode: str
) -> tuple[list[list[Center]], Grid, Any]:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    if rank == root:
        grid_obj = SimpleDetector(pathname=infile, varname=varname, trange=trange)
        grids: list[Grid] | None = grid_obj.split(size)
    else:
        grids = None

    grid = comm.scatter(grids, root=root)
    return grid.detect(minmaxmode=mode), grid, comm


def _detect_dask(
    infile: str,
    varname: str,
    trange: tuple[int, int] | None,
    mode: str,
    n_workers: int | None,
) -> tuple[list[list[Center]], Grid]:
    import dask
    from distributed import Client, LocalCluster

    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 4

    grid_obj = SimpleDetector(pathname=infile, varname=varname, trange=trange)
    grids = grid_obj.split(n_workers)

    with LocalCluster(
        n_workers=n_workers, threads_per_worker=1
    ) as cluster, Client(cluster):  # type: ignore
        delayed_results = [
            dask.delayed(g.detect)(minmaxmode=mode) for g in grids  # type: ignore
        ]
        results = dask.compute(*delayed_results)  # type: ignore

    # Flatten results from chunks
    flattened_results: list[list[Center]] = []
    for chunk in results:
        flattened_results.extend(chunk)

    return flattened_results, grid_obj


def _link_centers(centers: list[list[Center]]) -> Tracks:
    tracks = Tracks()
    linker = SimpleLinker()
    for step_centers in centers:
        linker.append_center(tracks, step_centers)
    return tracks


def _combine_mpi_tracks(tracks: Tracks, comm: Any) -> Tracks:  # noqa: ANN401
    rank = comm.Get_rank()
    size = comm.Get_size()
    linker = SimpleLinker()

    nstripe = 2
    while nstripe <= size:
        if rank % nstripe == nstripe // 2:
            comm.send(tracks, dest=rank - nstripe // 2, tag=nstripe)
        elif rank % nstripe == 0 and rank + nstripe // 2 < size:
            tracks_recv = comm.recv(source=rank + nstripe // 2, tag=nstripe)
            linker.extend_track(tracks, tracks_recv)
        nstripe *= 2
    return tracks


def run_tracker(
    infile: str,
    varname: str,
    outfile: str,
    trange: tuple[int, int] | None = None,
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
    comm = None
    if use_mpi:
        centers, grid, comm = _detect_mpi(infile, varname, trange, mode)
    elif use_dask:
        centers, grid = _detect_dask(infile, varname, trange, mode, n_workers)
    else:
        centers, grid = _detect_serial(infile, varname, trange, mode)

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

        export_to_csv(tracks, outfile, grid)
        final_outfile = outfile if outfile.endswith(".csv") else f"{outfile}.csv"
        print(f"Results exported to {final_outfile}")


def parse_args() -> Namespace:
    """Parses command line arguments."""
    parser = ArgumentParser(description="PyStormTracker: A tool for tracking storms.")
    parser.add_argument("-i", "--input", required=True, help="Input NetCDF file.")
    parser.add_argument("-v", "--var", required=True, help="Variable to track.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")
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
    trange = (0, args.num) if args.num is not None else None

    if True:
        run_tracker(
            infile=args.input,
            varname=args.var,
            outfile=args.output,
            trange=trange,
            mode=args.mode,
            backend=args.backend,
            n_workers=args.workers,
        )


if __name__ == "__main__":
    main()
