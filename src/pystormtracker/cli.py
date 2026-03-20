from __future__ import annotations

import argparse
import timeit
from argparse import Namespace
from pathlib import Path
from typing import Literal

import numpy as np

from .hodges.tracker import HodgesTracker
from .models.tracker import Tracker
from .simple.detector import SimpleDetector
from .simple.tracker import SimpleTracker

Backend = Literal["serial", "mpi", "dask"]
Algorithm = Literal["simple", "hodges"]


def run_tracker(
    infile: str,
    varname: str,
    outfile: str,
    start_time: str | np.datetime64 | None = None,
    end_time: str | np.datetime64 | None = None,
    mode: Literal["min", "max"] = "min",
    backend: Backend = "serial",
    n_workers: int | None = None,
    max_chunk_size: int | None = None,
    threshold: float | None = None,
    engine: str | None = None,
    algorithm: Algorithm = "simple",
    output_format: str = "imilast",
    # Hodges-specific
    min_points: int = 1,
    w1: float = 0.2,
    w2: float = 0.8,
    dmax: float = 6.5,
    phimax: float = 0.5,
    n_iterations: int = 3,
    min_lifetime: int = 3,
    max_missing: int = 0,
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

    tracker: Tracker
    if algorithm == "simple":
        tracker = SimpleTracker()
    else:
        tracker = HodgesTracker(
            w1=w1,
            w2=w2,
            dmax=dmax,
            phimax=phimax,
            n_iterations=n_iterations,
            min_lifetime=min_lifetime,
            max_missing=max_missing,
        )

    tracks = tracker.track(
        infile=infile,
        varname=varname,
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        backend=backend,
        n_workers=n_workers,
        max_chunk_size=max_chunk_size,
        threshold=threshold,
        engine=engine,
        min_points=min_points,
    )

    # Export Phase
    if rank == 0:
        num_tracks = len(tracks)
        print(f"Total number of tracks: {num_tracks}")

        timer["export"] = timeit.default_timer()
        tracks.write(outfile, format=output_format)
        timer["export"] = timeit.default_timer() - timer["export"]

        out_path = Path(outfile)
        final_outfile = (
            out_path if out_path.suffix == ".txt" else out_path.with_suffix(".txt")
        )
        print(f"Export time: {timer['export']:.4f}s")
        print(f"Results exported to {final_outfile}")

        timer["total"] = timeit.default_timer() - timer["total"]
        print(f"Total time: {timer['total']:.4f}s")


def parse_args() -> Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyStormTracker: A High-Performance Cyclone Tracker in Python",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40),
    )

    # 1. Required Arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("-i", "--input", required=True, help="Input NetCDF file.")
    required.add_argument(
        "-v", "--var", required=True, help="Variable to track (e.g., 'vo', 'msl')."
    )
    required.add_argument(
        "-o", "--output", required=True, help="Output track file (.txt)."
    )

    # 2. General Tracking Options
    general = parser.add_argument_group("General Tracking Options")
    general.add_argument(
        "-a",
        "--algorithm",
        choices=["simple", "hodges"],
        default="simple",
        help="Tracking algorithm. Default is 'simple'.",
    )
    general.add_argument(
        "-f",
        "--format",
        choices=["imilast", "hodges"],
        default="imilast",
        help="Output format. Default is 'imilast'.",
    )
    general.add_argument(
        "-m",
        "--mode",
        choices=["min", "max"],
        default="min",
        help="Detection mode: 'min' for cyclones in SLP, 'max' for vorticity.",
    )
    general.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Intensity threshold for features.",
    )
    general.add_argument(
        "-n", "--num", type=int, help="Number of time steps to process."
    )

    # 3. Performance & Parallelism
    perf = parser.add_argument_group("Performance & Parallelism")
    perf.add_argument(
        "-b",
        "--backend",
        choices=["serial", "mpi", "dask"],
        default="serial",
        help="Parallel backend. Default is 'serial'.",
    )
    perf.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of workers for Dask. Defaults to CPU cores.",
    )
    perf.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=60,
        help="Steps per chunk for Dask/RSPLICE. Default 60.",
    )
    perf.add_argument(
        "-e",
        "--engine",
        choices=["h5netcdf", "netcdf4", "cfgrib"],
        default=None,
        help="Xarray engine for reading input.",
    )

    # 4. Hodges (TRACK) Specific Options
    hodges = parser.add_argument_group("Hodges (TRACK) Algorithm Options")
    hodges.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Min grid points per object (noise filter).",
    )
    hodges.add_argument(
        "--w1", type=float, default=0.2, help="Cost weight for direction. Default 0.2."
    )
    hodges.add_argument(
        "--w2", type=float, default=0.8, help="Cost weight for speed. Default 0.8."
    )
    hodges.add_argument(
        "--dmax",
        type=float,
        default=6.5,
        help="Max search radius in degrees. Default 6.5.",
    )
    hodges.add_argument(
        "--phimax",
        type=float,
        default=0.5,
        help="Smoothness penalty (static). Default 0.5.",
    )
    hodges.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Max MGE optimization passes. Default 3.",
    )
    hodges.add_argument(
        "--min-lifetime",
        type=int,
        default=3,
        help="Min steps for a valid track. Default 3.",
    )
    hodges.add_argument(
        "--max-missing",
        type=int,
        default=0,
        help="Max consecutive missing frames. Default 0.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = None
    end_time = None

    if args.num is not None:
        # Determine actual times for the first n steps
        from .hodges.detector import HodgesDetector

        detector_preview: SimpleDetector | HodgesDetector
        if args.algorithm == "simple":
            detector_preview = SimpleDetector(
                pathname=args.input, varname=args.var, engine=args.engine
            )
        else:
            detector_preview = HodgesDetector(
                pathname=args.input, varname=args.var, engine=args.engine
            )

        times = detector_preview.get_time()
        assert times is not None
        num = min(args.num, len(times))
        start_time = times[0]
        end_time = times[num - 1]

    run_tracker(
        infile=args.input,
        varname=args.var,
        outfile=args.output,
        start_time=start_time,
        end_time=end_time,
        mode=args.mode,
        backend=args.backend,
        n_workers=args.workers,
        max_chunk_size=args.chunk_size,
        threshold=args.threshold,
        engine=args.engine,
        algorithm=args.algorithm,
        output_format=args.format,
        # Hodges-specific
        min_points=args.min_points,
        w1=args.w1,
        w2=args.w2,
        dmax=args.dmax,
        phimax=args.phimax,
        n_iterations=args.iterations,
        min_lifetime=args.min_lifetime,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
