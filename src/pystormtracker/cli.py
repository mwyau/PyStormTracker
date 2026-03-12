from __future__ import annotations

import timeit
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

from .models import TimeRange
from .simple.detector import SimpleDetector
from .simple.tracker import SimpleTracker

Backend = Literal["serial", "mpi", "dask"]


def run_tracker(
    infile: str,
    varname: str,
    outfile: str | None,
    time_range: TimeRange | None = None,
    mode: Literal["min", "max"] = "min",
    backend: Backend = "serial",
    n_workers: int | None = None,
) -> None:
    """Orchestrates the storm tracking process from the CLI."""
    timer: dict[str, float] = {}
    use_mpi = backend == "mpi"

    rank = 0
    if use_mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        timer["total"] = timeit.default_timer()

    tracker = SimpleTracker()
    tracks = tracker.track(
        infile=infile,
        varname=varname,
        time_range=time_range,
        mode=mode,
        backend=backend,
        n_workers=n_workers,
    )

    # Export Phase
    if rank == 0:
        timer["total"] = timeit.default_timer() - timer["total"]
        print(f"Total tracking time: {timer['total']:.4f}s")
        
        if outfile is None:
            print("Skipping export as outfile is None.")
            return

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
        default="serial",
        help="Parallel backend. Default is 'serial'.",
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
