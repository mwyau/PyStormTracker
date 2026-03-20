from __future__ import annotations

import timeit
from argparse import ArgumentParser, Namespace
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
    zone_file: str | None = None,
    adapt_file: str | None = None,
    min_points: int = 1,
    w1: float = 0.2,
    w2: float = 0.8,
    dmax: float = 5.0,
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
        tracker = HodgesTracker.from_config(
            zone_file=zone_file,
            adapt_file=adapt_file,
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
    parser = ArgumentParser(
        description="PyStormTracker: A High-Performance Cyclone Tracker in Python"
    )
    parser.add_argument("-i", "--input", required=True, help="Input NetCDF file.")
    parser.add_argument("-v", "--var", required=True, help="Variable to track.")
    parser.add_argument(
        "-o", "--output", required=True, help="Output track file (.txt)."
    )
    parser.add_argument("-n", "--num", type=int, help="Number of time steps.")
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["simple", "hodges"],
        default="simple",
        help="Tracking algorithm. Default is 'simple'.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["imilast", "hodges"],
        default="imilast",
        help="Output format. Default is 'imilast'.",
    )
    parser.add_argument(
        "-m", "--mode", choices=["min", "max"], default="min", help="Detection mode."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=None, help="Detection threshold."
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
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=60,
        help="Maximum time steps per chunk for Dask. Defaults to 60.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["h5netcdf", "netcdf4", "cfgrib"],
        default=None,
        help="Xarray engine to use for reading the input file.",
    )
    # Hodges-specific
    parser.add_argument(
        "--zone", help="Path to TRACK-style zone.dat file for regional dmax."
    )
    parser.add_argument(
        "--adapt", help="Path to TRACK-style adapt.dat file for adaptive smoothness."
    )
    parser.add_argument(
        "--min-points", type=int, default=1, help="Minimum points per object."
    )
    parser.add_argument(
        "--w1", type=float, default=0.2, help="Cost weight for direction."
    )
    parser.add_argument("--w2", type=float, default=0.8, help="Cost weight for speed.")
    parser.add_argument(
        "--dmax", type=float, default=5.0, help="Default search radius."
    )
    parser.add_argument(
        "--phimax", type=float, default=0.5, help="Smoothness penalty (static)."
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of MGE iterations."
    )
    parser.add_argument(
        "--lifetime", type=int, default=3, help="Minimum track lifetime."
    )
    parser.add_argument(
        "--max-missing", type=int, default=0, help="Max consecutive missing frames."
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
        zone_file=args.zone,
        adapt_file=args.adapt,
        min_points=args.min_points,
        w1=args.w1,
        w2=args.w2,
        dmax=args.dmax,
        phimax=args.phimax,
        n_iterations=args.iterations,
        min_lifetime=args.lifetime,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
