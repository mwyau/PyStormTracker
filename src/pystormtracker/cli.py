from __future__ import annotations

import argparse
import os
import timeit
from argparse import Namespace
from typing import Literal

import numpy as np

from .hodges import constants
from .hodges.tracker import HodgesTracker
from .models.tracker import Tracker
from .simple.detector import SimpleDetector
from .simple.tracker import SimpleTracker

Backend = Literal["serial", "mpi", "dask"]
Algorithm = Literal["simple", "hodges"]


def is_mpi_env() -> bool:
    """Detects if the current process is running in an MPI environment."""
    # Common MPI environment variables
    mpi_vars = ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"]
    return any(v in os.environ for v in mpi_vars)


def run_tracker(
    infile: str,
    varname: str,
    outfile: str,
    start_time: str | np.datetime64 | None = None,
    end_time: str | np.datetime64 | None = None,
    mode: Literal["min", "max"] = "min",
    backend: Backend | None = None,
    n_workers: int | None = None,
    max_chunk_size: int | None = None,
    threshold: float | None = None,
    engine: str | None = None,
    algorithm: Algorithm = "simple",
    output_format: str = "imilast",
    # Hodges-specific
    min_points: int = 1,
    w1: float | None = None,
    w2: float | None = None,
    dmax: float | None = None,
    phimax: float | None = None,
    n_iterations: int | None = None,
    min_lifetime: int | None = None,
    max_missing: int | None = None,
    filter: bool = True,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
) -> None:
    """Orchestrates the storm tracking process from the CLI."""
    timer: dict[str, float] = {}

    # 1. Backend Auto-detection
    detected_backend: Backend = "serial"
    if backend:
        detected_backend = backend
    elif is_mpi_env():
        detected_backend = "mpi"
    elif n_workers is not None:
        detected_backend = "dask"

    use_mpi = detected_backend == "mpi"

    rank = 0
    if use_mpi:
        if not is_mpi_env():
            print(
                "Warning: MPI backend selected but no MPI environment detected "
                "(e.g., OMPI_COMM_WORLD_SIZE not set)."
            )
            print("Ensure you are running with 'mpirun' or 'mpiexec'.")

        try:
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()
            if n_workers is None:
                n_workers = MPI.COMM_WORLD.Get_size()
        except ImportError:
            if backend == "mpi":
                raise ImportError(
                    "mpi4py is required for MPI backend. "
                    "Install it with 'pip install PyStormTracker[mpi]'."
                ) from None
            # If auto-detected but not installed, fallback to serial or dask
            if is_mpi_env():
                print(
                    "Warning: MPI environment detected but mpi4py is not installed. "
                    "Falling back."
                )
            detected_backend = "dask" if n_workers else "serial"
            use_mpi = False

    if rank == 0:
        timer["total"] = timeit.default_timer()
        print(f"Using backend: {detected_backend}")
        if n_workers:
            print(f"Workers: {n_workers}")

    tracker: Tracker
    if algorithm == "simple":
        tracker = SimpleTracker()
    else:
        # Pass only provided values to use tracker defaults
        hodges_kwargs: dict[str, float | int] = {}
        if w1 is not None:
            hodges_kwargs["w1"] = w1
        if w2 is not None:
            hodges_kwargs["w2"] = w2
        if dmax is not None:
            hodges_kwargs["dmax"] = dmax
        if phimax is not None:
            hodges_kwargs["phimax"] = phimax
        if n_iterations is not None:
            hodges_kwargs["n_iterations"] = n_iterations
        if min_lifetime is not None:
            hodges_kwargs["min_lifetime"] = min_lifetime
        if max_missing is not None:
            hodges_kwargs["max_missing"] = max_missing

        tracker = HodgesTracker(**hodges_kwargs)  # type: ignore[arg-type]

    tracks = tracker.track(
        infile=infile,
        varname=varname,
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        backend=detected_backend,
        n_workers=n_workers,
        max_chunk_size=max_chunk_size,
        threshold=threshold,
        engine=engine,
        min_points=min_points,
        filter=filter,
        lmin=lmin,
        lmax=lmax,
    )

    # Export Phase
    if rank == 0:
        num_tracks = len(tracks)
        print(f"Total number of tracks: {num_tracks}")

        timer["export"] = timeit.default_timer()
        tracks.write(outfile, format=output_format)
        timer["export"] = timeit.default_timer() - timer["export"]

        print(f"Export time: {timer['export']:.4f}s")
        print(f"Results exported to {outfile}")

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

    # Filtering Options (Mutually Exclusive)
    filter_group = general.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filter-range",
        type=str,
        default=f"{constants.LMIN_DEFAULT}-{constants.LMAX_DEFAULT}",
        help=f"Spectral filter range (min-max). Default '{constants.LMIN_DEFAULT}-{constants.LMAX_DEFAULT}'.",
    )
    filter_group.add_argument(
        "--no-filter",
        action="store_false",
        dest="filter",
        help="Disable default T5-42 spectral filtering.",
    )
    parser.set_defaults(filter=True)

    general.add_argument(
        "-n", "--num", type=int, help="Number of time steps to process."
    )

    # 3. Performance & Parallelism
    perf = parser.add_argument_group("Performance & Parallelism")
    perf.add_argument(
        "-b",
        "--backend",
        choices=["serial", "mpi", "dask"],
        default=None,
        help="Parallel backend. Auto-detected by default.",
    )
    perf.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of workers. Auto-detected for MPI. Sets Dask if not MPI.",
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
        "--w1", type=float, default=None, help="Cost weight for direction. Default 0.2."
    )
    hodges.add_argument(
        "--w2", type=float, default=None, help="Cost weight for speed. Default 0.8."
    )
    hodges.add_argument(
        "--dmax",
        type=float,
        default=None,
        help="Max search radius in degrees. Default 6.5.",
    )
    hodges.add_argument(
        "--phimax",
        type=float,
        default=None,
        help="Smoothness penalty (static). Default 0.5.",
    )
    hodges.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Max MGE optimization passes. Default 3.",
    )
    hodges.add_argument(
        "--min-lifetime",
        type=int,
        default=None,
        help="Min steps for a valid track. Default 3.",
    )
    hodges.add_argument(
        "--max-missing",
        type=int,
        default=None,
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

    lmin, lmax = constants.LMIN_DEFAULT, constants.LMAX_DEFAULT
    if args.filter and args.filter_range:
        try:
            parts = args.filter_range.split("-")
            if len(parts) == 2:
                lmin, lmax = int(parts[0]), int(parts[1])
            elif len(parts) == 1:
                lmax = int(parts[0])
        except ValueError:
            print(
                f"Warning: Could not parse filter-range '{args.filter_range}'. "
                f"Using {constants.LMIN_DEFAULT}-{constants.LMAX_DEFAULT}."
            )

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
        filter=args.filter,
        lmin=lmin,
        lmax=lmax,
    )



if __name__ == "__main__":
    main()
