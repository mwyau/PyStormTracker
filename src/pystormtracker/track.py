from __future__ import annotations

import argparse
import json
import os
import timeit
from argparse import Namespace
from typing import Literal, cast

import numpy as np

from .healpix.tracker import HealpixTracker
from .hodges import constants
from .hodges.tracker import HodgesTracker
from .io.format import SUPPORTED_FORMATS, SupportedFormat
from .models.tracker import Tracker
from .preprocessing.tracking import resolve_filter_bounds
from .simple.detector import SimpleDetector
from .simple.tracker import SimpleTracker
from .utils.cli import (
    finite_float,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)

Backend = Literal["serial", "mpi", "dask"]
Algorithm = Literal["simple", "hodges"]


def _parse_extent(value: str) -> tuple[float, float, float, float]:
    """Parse xmin,xmax,ymin,ymax and validate both axes."""
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected xmin,xmax,ymin,ymax") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected xmin,xmax,ymin,ymax")
    if not np.isfinite(parts).all():
        raise argparse.ArgumentTypeError("extent values must be finite")
    xmin, xmax, ymin, ymax = parts
    if xmin >= xmax or ymin >= ymax:
        raise argparse.ArgumentTypeError("extent minima must be less than maxima")
    return xmin, xmax, ymin, ymax


def _validate_zones(zones: np.ndarray) -> np.ndarray:
    """Validate TRACK regional constraints as rows of five values."""
    zones = np.atleast_2d(zones).astype(np.float64, copy=False)
    if not np.isfinite(zones).all():
        raise ValueError("zone values must be finite")
    if zones.shape[1] != 5:
        raise ValueError(
            "zones must contain rows of [lon_min, lon_max, lat_min, lat_max, dmax]"
        )
    if np.any(zones[:, 0] >= zones[:, 1]) or np.any(zones[:, 2] >= zones[:, 3]):
        raise ValueError("zone minima must be less than zone maxima")
    if np.any(zones[:, 4] <= 0.0):
        raise ValueError("zone dmax values must be greater than zero")
    return zones


def _validate_adapt_params(params: np.ndarray) -> np.ndarray:
    """Validate adaptive smoothness thresholds and values."""
    if params.shape != (2, 4):
        raise ValueError("adaptive parameters must have shape (2, 4)")
    if not np.isfinite(params).all():
        raise ValueError("adaptive parameters must be finite")
    if np.any(np.diff(params[0]) < 0.0):
        raise ValueError("adaptive distance thresholds must be nondecreasing")
    if np.any(params[1] < 0.0):
        raise ValueError("adaptive smoothness values must be nonnegative")
    return params.astype(np.float64, copy=False)


def is_mpi_env() -> bool:
    """Detects if the current process is running in an MPI environment."""
    mpi_vars = ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"]
    return any(v in os.environ for v in mpi_vars)


def setup_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Sets up the argument parser for the track command."""
    parser = subparsers.add_parser(
        "track",
        description="Run the storm tracking algorithm.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=40),
    )

    # 1. Required Arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("-i", "--input", required=True, help="Input NetCDF file.")
    required.add_argument(
        "-v", "--var", required=True, help="Variable to track (e.g., 'vo', 'msl')."
    )
    required.add_argument(
        "-o",
        "--out",
        "--output",
        dest="output",
        required=True,
        help="Output track file.",
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
        choices=["auto", *SUPPORTED_FORMATS],
        default="auto",
        help="Output format; inferred from the extension, defaulting to TrackJSON.",
    )
    general.add_argument(
        "-m",
        "--mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Detection mode; inferred from known variable aliases.",
    )
    general.add_argument(
        "--map-proj",
        choices=["global", "nh_stereo", "sh_stereo", "healpix"],
        default="global",
        help="Map projection for detection. Default 'global'.",
    )
    general.add_argument(
        "--resolution",
        type=positive_float,
        default=100.0,
        help="Grid resolution in km for stereographic projections. Default 100.0.",
    )
    general.add_argument(
        "--extent",
        type=_parse_extent,
        default=(-13000.0, 13000.0, -13000.0, 13000.0),
        help="Bounding box in km (xmin,xmax,ymin,ymax) for stereographic projections.",
    )
    general.add_argument(
        "-t",
        "--threshold",
        type=finite_float,
        default=None,
        help="Intensity threshold for features.",
    )

    general.add_argument(
        "--lmin",
        type=nonnegative_int,
        default=None,
        help="Optional lower spectral filter bound; supply with --lmax.",
    )
    general.add_argument(
        "--lmax",
        type=nonnegative_int,
        default=None,
        help="Optional upper spectral filter bound; supply with --lmin.",
    )
    general.add_argument(
        "--taper-points",
        type=nonnegative_int,
        default=0,
        help="Independent spatial taper width; zero disables tapering.",
    )
    general.add_argument(
        "--nside",
        type=positive_int,
        default=None,
        help=(
            "Target HEALPix resolution; omitted values are derived from the "
            "source grid."
        ),
    )

    general.add_argument(
        "-n", "--num", type=positive_int, help="Number of time steps to process."
    )
    general.add_argument(
        "--subgrid-refine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control quadratic subgrid refinement. Disabled by default for "
            "simple tracking and enabled by default for Hodges and HEALPix."
        ),
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
        type=positive_int,
        default=None,
        help="Number of workers. Auto-detected for MPI. Sets Dask if not MPI.",
    )
    perf.add_argument(
        "-c",
        "--chunk-size",
        type=positive_int,
        default=None,
        help="Detection steps per chunk. Backend default when omitted.",
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
        type=positive_int,
        default=1,
        help="Min grid points per object (noise filter).",
    )
    hodges.add_argument(
        "--w1",
        type=nonnegative_float,
        default=None,
        help="Cost weight for direction. Default 0.2.",
    )
    hodges.add_argument(
        "--w2",
        type=nonnegative_float,
        default=None,
        help="Cost weight for speed. Default 0.8.",
    )
    hodges.add_argument(
        "--dmax",
        type=positive_float,
        default=None,
        help="Max search radius in degrees. Default 6.5.",
    )
    hodges.add_argument(
        "--phimax",
        type=nonnegative_float,
        default=None,
        help="Smoothness penalty (static). Default 0.5.",
    )
    hodges.add_argument(
        "--iterations",
        type=positive_int,
        default=None,
        help="Max MGE optimization passes. Default 3.",
    )
    hodges.add_argument(
        "--min-lifetime",
        type=positive_int,
        default=None,
        help="Min steps for a valid track. Default 3.",
    )
    hodges.add_argument(
        "--max-missing",
        type=nonnegative_int,
        default=None,
        help="Max consecutive missing frames. Default 0.",
    )

    zone_group = hodges.add_mutually_exclusive_group()
    zone_group.add_argument(
        "--zone-file",
        type=str,
        default=None,
        help="Path to legacy zone.dat file for regional DMAX.",
    )
    zone_group.add_argument(
        "--zones",
        type=str,
        default=None,
        help="JSON string defining regional DMAX zones.",
    )

    adapt_group = hodges.add_mutually_exclusive_group()
    adapt_group.add_argument(
        "--adapt-file",
        type=str,
        default=None,
        help="Path to legacy adapt.dat file for adaptive smoothness.",
    )
    adapt_group.add_argument(
        "--adapt-params",
        type=str,
        default=None,
        help="JSON string defining adaptive smoothness parameters (2x4 array).",
    )
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    """
    Main entry point for the track command.
    """
    start_time = None
    end_time = None

    if args.num is not None:
        from .hodges.detector import HodgesDetector

        detector_preview: SimpleDetector | HodgesDetector
        if args.algorithm == "simple":
            detector_preview = SimpleDetector(
                pathname=args.input, variable_name=args.var, engine=args.engine
            )
        else:
            detector_preview = HodgesDetector(
                pathname=args.input, variable_name=args.var, engine=args.engine
            )

        times = detector_preview.get_time()
        assert times is not None
        num = min(args.num, len(times))
        start_time = times[0]
        end_time = times[num - 1]

    lmin, lmax = args.lmin, args.lmax
    resolve_filter_bounds(lmin, lmax)

    zones_arr = None
    if args.zone_file:
        with open(args.zone_file) as f:
            first_line = f.readline().split()
            has_header = len(first_line) == 1
        zones_arr = _validate_zones(
            np.loadtxt(args.zone_file, skiprows=1 if has_header else 0)
        )
    elif args.zones:
        try:
            zones_arr = _validate_zones(
                np.array(json.loads(args.zones), dtype=np.float64)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid zones JSON: {exc.msg}") from exc

    adapt_params_arr = None
    if args.adapt_file:
        arr = np.loadtxt(args.adapt_file)
        adapt_params_arr = _validate_adapt_params(arr.T if arr.shape == (4, 2) else arr)
    elif args.adapt_params:
        try:
            adapt_params_arr = _validate_adapt_params(
                np.array(json.loads(args.adapt_params), dtype=np.float64)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adaptive-parameters JSON: {exc.msg}") from exc

    # Auto-detect backend
    detected_backend: Backend = "serial"
    if args.backend:
        detected_backend = args.backend
    elif is_mpi_env():
        detected_backend = "mpi"
    elif args.workers is not None:
        detected_backend = "dask"

    use_mpi = detected_backend == "mpi"
    rank = 0
    n_workers = args.workers

    if use_mpi:
        import shutil

        if not shutil.which("mpiexec"):
            if args.backend == "mpi":
                raise RuntimeError(
                    "MPI backend requested but 'mpiexec' not found in PATH. "
                    "Please install an MPI implementation (e.g., OpenMPI or MS-MPI)."
                )
            else:
                print("Warning: MPI environment detected but 'mpiexec' missing.")
                detected_backend = "dask" if n_workers else "serial"
                use_mpi = False

    if use_mpi:
        if not is_mpi_env():
            print(
                "Warning: MPI backend selected but no MPI environment detected "
                "(e.g., OMPI_COMM_WORLD_SIZE not set)."
            )
            print("Ensure you are running with 'mpirun' or 'mpiexec'.")

        try:
            from mpi4py import MPI

            rank = comm.Get_rank() if (comm := MPI.COMM_WORLD) else 0
            if n_workers is None:
                n_workers = MPI.COMM_WORLD.Get_size()
        except ImportError:
            if args.backend == "mpi":
                raise ImportError(
                    "mpi4py is required for MPI backend. "
                    "Install it with 'pip install PyStormTracker[mpi]'."
                ) from None
            if is_mpi_env():
                print(
                    "Warning: MPI environment detected but mpi4py is not installed. "
                    "Falling back."
                )
            detected_backend = "dask" if n_workers else "serial"
            use_mpi = False

    timer: dict[str, float] = {}
    if rank == 0:
        timer["total"] = timeit.default_timer()
        print(f"Using backend: {detected_backend}")
        if n_workers:
            print(f"Workers: {n_workers}")

    # Validate options against selected tracker and instantiate tracker
    tracker: Tracker
    if args.map_proj == "healpix":
        if detected_backend != "serial":
            raise ValueError("HealpixTracker supports only the serial backend.")
        if args.chunk_size is not None:
            raise ValueError("HealpixTracker does not support chunking.")
        if args.resolution != 100.0 or args.extent != (
            -13000.0,
            13000.0,
            -13000.0,
            13000.0,
        ):
            raise ValueError(
                "Stereographic resolution and extent options are not supported with "
                "HEALPix projection."
            )

        effective_subgrid = (
            args.subgrid_refine if args.subgrid_refine is not None else True
        )
        tracker = HealpixTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            n_iterations=args.iterations
            if args.iterations is not None
            else constants.ITERATIONS_DEFAULT,
            min_lifetime=args.min_lifetime
            if args.min_lifetime is not None
            else constants.LIFETIME_DEFAULT,
            max_missing=args.max_missing
            if args.max_missing is not None
            else constants.MISSING_DEFAULT,
            zones=zones_arr,
            adapt_params=adapt_params_arr,
            nside=args.nside,
            lmin=lmin,
            lmax=lmax,
            taper_points=args.taper_points,
            min_points=args.min_points,
            subgrid_refine=effective_subgrid,
        )
    elif args.algorithm == "hodges":
        if detected_backend != "serial":
            raise ValueError("HodgesTracker supports only the serial backend.")
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")

        effective_subgrid = (
            args.subgrid_refine if args.subgrid_refine is not None else True
        )
        tracker = HodgesTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            n_iterations=args.iterations
            if args.iterations is not None
            else constants.ITERATIONS_DEFAULT,
            min_lifetime=args.min_lifetime
            if args.min_lifetime is not None
            else constants.LIFETIME_DEFAULT,
            max_missing=args.max_missing
            if args.max_missing is not None
            else constants.MISSING_DEFAULT,
            zones=zones_arr,
            adapt_params=adapt_params_arr,
            map_proj=args.map_proj,
            resolution=args.resolution,
            extent=args.extent,
            lmin=lmin,
            lmax=lmax,
            taper_points=args.taper_points,
            min_points=args.min_points,
            subgrid_refine=effective_subgrid,
            max_chunk_size=args.chunk_size,
        )
    else:  # simple tracker
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")
        has_hodges_option = (
            args.min_points != 1
            or args.w1 is not None
            or args.w2 is not None
            or args.dmax is not None
            or args.phimax is not None
            or args.iterations is not None
            or args.min_lifetime is not None
            or args.max_missing is not None
            or args.zone_file is not None
            or args.zones is not None
            or args.adapt_file is not None
            or args.adapt_params is not None
        )
        if has_hodges_option:
            raise ValueError(
                "Hodges options (w1, w2, dmax, phimax, iterations, min_lifetime, "
                "max_missing, min_points, zones, adapt) are not supported "
                "with SimpleTracker."
            )

        effective_subgrid = (
            args.subgrid_refine if args.subgrid_refine is not None else False
        )
        tracker = SimpleTracker(
            map_proj=args.map_proj,
            resolution=args.resolution,
            extent=args.extent,
            lmin=lmin,
            lmax=lmax,
            taper_points=args.taper_points,
            size=5,
            subgrid_refine=effective_subgrid,
            backend=detected_backend,
            n_workers=n_workers,
            max_chunk_size=args.chunk_size,
        )

    tracks = tracker.track(
        infile=args.input,
        variable_name=args.var,
        start_time=start_time,
        end_time=end_time,
        mode=args.mode,
        threshold=args.threshold,
        engine=args.engine,
    )

    if rank == 0:
        num_tracks = len(tracks)
        print(f"Total number of tracks: {num_tracks}")

        timer["export"] = timeit.default_timer()
        selected_format = (
            None
            if args.format in (None, "auto")
            else cast("SupportedFormat", args.format)
        )
        tracks.write(args.output, format=selected_format)
        timer["export"] = timeit.default_timer() - timer["export"]

        print(f"Export time: {timer['export']:.4f}s")
        print(f"Results exported to {args.output}")

        timer["total"] = timeit.default_timer() - timer["total"]
        print(f"Total time: {timer['total']:.4f}s")
