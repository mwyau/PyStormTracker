import csv
import sys
import timeit
from argparse import ArgumentParser, Namespace
from typing import Literal

import netCDF4
import numpy as np

from .models import Grid, Tracks
from .simple import SimpleDetector, SimpleLinker


def export_to_csv(
    tracks: Tracks, outfile: str, grid: Grid, decimal_places: int = 4
) -> None:
    """Exports detected tracks to a user-friendly CSV file."""
    time_obj = grid.get_time_obj()
    units = getattr(time_obj, "units", "")
    calendar = getattr(time_obj, "calendar", "standard")

    if not outfile.endswith(".csv"):
        outfile += ".csv"

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "time", "lat", "lon", "var"])
        for i, track in enumerate(tracks):
            for center in track:
                # Convert numeric time to user-friendly string
                try:
                    dt = netCDF4.num2date(center.time, units=units, calendar=calendar)
                    # Use type ignore because num2date can return array or single object
                    time_val = dt.strftime("%Y-%m-%d %H:%M:%S")  # type: ignore
                except Exception:
                    time_val = str(center.time)

                # Convert masked values to '--' and format float
                if center.var is None or np.ma.is_masked(center.var):
                    var_val = "--"
                else:
                    var_val = f"{float(center.var):.{decimal_places}f}"
                writer.writerow([i, time_val, center.lat, center.lon, var_val])


def run_tracker(
    infile: str,
    varname: str,
    outfile: str,
    trange: tuple[int, int] | None = None,
    mode: Literal["min", "max"] = "min",
) -> None:
    """Orchestrates the storm tracking process."""
    try:
        from mpi4py import MPI

        use_mpi = True
    except ImportError:
        use_mpi = False

    timer: dict[str, float] = {}

    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0
    else:
        rank = 0
        root = 0

    if not use_mpi or rank == root:
        timer["detector"] = timeit.default_timer()

    # Initialize grid and detect centers
    if use_mpi:
        if rank == root:
            grid_obj = SimpleDetector(pathname=infile, varname=varname, trange=trange)
            grids = grid_obj.split(size)
        else:
            grids = None
        grid = comm.scatter(grids, root=root)
    else:
        grid = SimpleDetector(pathname=infile, varname=varname, trange=trange)

    centers = grid.detect(minmaxmode=mode)

    if use_mpi:
        comm.Barrier()

    if not use_mpi or rank == root:
        timer["detector"] = timeit.default_timer() - timer["detector"]
        timer["linker"] = timeit.default_timer()

    # Link centers into tracks
    tracks = Tracks()
    linker = SimpleLinker()

    for step_centers in centers:
        linker.append_center(tracks, step_centers)

    # Combine tracks if running with MPI
    if use_mpi:
        timer["combiner"] = timeit.default_timer()
        nstripe = 2
        while nstripe <= size:
            if rank % nstripe == nstripe // 2:
                comm.send(tracks, dest=rank - nstripe // 2, tag=nstripe)
            elif rank % nstripe == 0 and rank + nstripe // 2 < size:
                tracks_recv = comm.recv(source=rank + nstripe // 2, tag=nstripe)
                linker.extend_track(tracks, tracks_recv)
            nstripe *= 2
        timer["combiner"] = timeit.default_timer() - timer["combiner"]

    # Finalize and export results
    if not use_mpi or rank == root:
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
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input NetCDF file."
    )
    parser.add_argument(
        "-v", "--var", required=True, help="Variable to track (e.g., 'slp')."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output CSV file."
    )
    parser.add_argument(
        "-n", "--num", type=int, help="Number of time steps to process."
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["min", "max"],
        default="min",
        help="Detection mode: 'min' or 'max'. Default is 'min'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trange = (0, args.num) if args.num is not None else None

    try:
        run_tracker(
            infile=args.input,
            varname=args.var,
            outfile=args.output,
            trange=trange,
            mode=args.mode,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
