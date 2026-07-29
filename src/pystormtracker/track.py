from __future__ import annotations

import argparse
import json
import os
import timeit
from argparse import Namespace
from typing import Literal

import numpy as np

from .hodges import constants
from .hodges.tracker import HodgesTracker
from .models import constants as model_constants
from .models.tracks import Tracks
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


def _parse_filter_range(value: str) -> tuple[int, int]:
    """Parse an inclusive spectral wave-number range."""
    try:
        parts = [int(part) for part in value.split("-")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MIN-MAX or MAX") from exc

    if len(parts) == 1:
        lmin, lmax = 0, parts[0]
    elif len(parts) == 2:
        lmin, lmax = parts
    else:
        raise argparse.ArgumentTypeError("expected MIN-MAX or MAX")
    if lmin < 0 or lmax < lmin:
        raise argparse.ArgumentTypeError("wave numbers must satisfy 0 <= MIN <= MAX")
    return lmin, lmax


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
    map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
    resolution: float = 100.0,
    extent: tuple[float, float, float, float] | None = None,
    backend: Backend | None = None,
    n_workers: int | None = None,
    max_chunk_size: int | None = None,
    threshold: float | None = None,
    engine: str | None = None,
    algorithm: Algorithm = "simple",
    output_format: str = "imilast",
    # Hodges-specific
    min_points: int = constants.MIN_POINTS_DEFAULT,
    w1: float | None = None,
    w2: float | None = None,
    dmax: float | None = None,
    phimax: float | None = None,
    n_iterations: int | None = None,
    min_lifetime: int | None = None,
    max_missing: int | None = None,
    zones: np.ndarray | None = None,
    adapt_params: np.ndarray | None = None,
    filter: bool | None = None,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
    taper_points: int = constants.TAPER_DEFAULT,
    overlap: int = model_constants.OVERLAP_DEFAULT,
    subgrid_refine: bool | None = None,
) -> Tracks:
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
        import shutil

        if not shutil.which("mpiexec"):
            if backend == "mpi":
                raise RuntimeError(
                    "MPI backend requested but 'mpiexec' not found in PATH. "
                    "Please install an MPI implementation (e.g., OpenMPI or MS-MPI)."
                )
            else:
                # Auto-detected via env vars but binary missing
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

    from .models.tracker import Tracker

    tracker: Tracker
    if map_proj == "healpix":
        from .healpix.tracker import HealpixTracker

        tracker = HealpixTracker(
            w1=w1 if w1 is not None else constants.W1_DEFAULT,
            w2=w2 if w2 is not None else constants.W2_DEFAULT,
            dmax=dmax if dmax is not None else constants.DMAX_DEFAULT,
            phimax=phimax if phimax is not None else constants.PHIMAX_DEFAULT,
            n_iterations=n_iterations
            if n_iterations is not None
            else constants.ITERATIONS_DEFAULT,
            min_lifetime=min_lifetime
            if min_lifetime is not None
            else constants.LIFETIME_DEFAULT,
            max_missing=max_missing
            if max_missing is not None
            else constants.MISSING_DEFAULT,
            zones=zones,
            adapt_params=adapt_params,
        )
    elif algorithm == "simple":
        tracker = SimpleTracker()
    else:
        # Initialize with standard defaults and override if provided
        tracker = HodgesTracker(
            w1=w1 if w1 is not None else constants.W1_DEFAULT,
            w2=w2 if w2 is not None else constants.W2_DEFAULT,
            dmax=dmax if dmax is not None else constants.DMAX_DEFAULT,
            phimax=phimax if phimax is not None else constants.PHIMAX_DEFAULT,
            n_iterations=n_iterations
            if n_iterations is not None
            else constants.ITERATIONS_DEFAULT,
            min_lifetime=min_lifetime
            if min_lifetime is not None
            else constants.LIFETIME_DEFAULT,
            max_missing=max_missing
            if max_missing is not None
            else constants.MISSING_DEFAULT,
            zones=zones,
            adapt_params=adapt_params,
        )

    effective_subgrid_refine = (
        subgrid_refine
        if subgrid_refine is not None
        else algorithm != "simple" or map_proj == "healpix"
    )
    effective_filter = (
        filter if filter is not None else algorithm != "simple" or map_proj == "healpix"
    )

    tracks = tracker.track(
        infile=infile,
        varname=varname,
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        map_proj=map_proj,
        resolution=resolution,
        extent=extent,
        backend=detected_backend,
        n_workers=n_workers,
        max_chunk_size=max_chunk_size,
        threshold=threshold,
        engine=engine,
        min_points=min_points,
        filter=effective_filter,
        lmin=lmin,
        lmax=lmax,
        taper_points=taper_points,
        overlap=overlap,
        subgrid_refine=effective_subgrid_refine,
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

    return tracks


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
    required.add_argument("-o", "--output", required=True, help="Output track file.")

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
        choices=["imilast", "hodges", "json"],
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

    # Filtering Options (Mutually Exclusive)
    filter_group = general.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filter-range",
        type=_parse_filter_range,
        default=None,
        help=(
            f"Spectral filter range (min-max). "
            f"Default '{constants.LMIN_DEFAULT}-{constants.LMAX_DEFAULT}'."
        ),
    )
    filter_group.add_argument(
        "--filter",
        action=argparse.BooleanOptionalAction,
        dest="filter",
        default=None,
        help=(
            "Control spectral filtering. Disabled by default for simple tracking "
            "and enabled by default for Hodges and HEALPix."
        ),
    )
    # Default is determined in main() based on algorithm

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
        "--overlap",
        type=nonnegative_int,
        default=model_constants.OVERLAP_DEFAULT,
        help=(
            "Compatibility option retained for older commands; Gather-then-Link "
            "does not require overlapping chunks."
        ),
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
        "--taper",
        type=nonnegative_int,
        default=constants.TAPER_DEFAULT,
        help="Number of points for boundary tapering. Default 0.",
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
    parser.set_defaults(func=run_track_command)


def main(args: Namespace) -> None:
    """
    Main entry point for the track command.
    """
    run_track_command(args)


def run_track_command(args: Namespace) -> Tracks:
    """
    Executes the tracking algorithm based on CLI arguments and returns
    the generated Tracks object.
    """
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

    if args.filter is None:
        args.filter = (
            args.filter_range is not None
            or args.algorithm != "simple"
            or args.map_proj == "healpix"
        )

    lmin, lmax = args.filter_range or (
        constants.LMIN_DEFAULT,
        constants.LMAX_DEFAULT,
    )

    zones_arr = None
    if args.zone_file:
        # Check if the file has a header line (single element)
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
        # Standard adapt.dat in TRACK is 4 points with (thresh, value) per line (4x2)
        # We need it as 2x4 (row 0: thresholds, row 1: values)
        arr = np.loadtxt(args.adapt_file)
        adapt_params_arr = _validate_adapt_params(arr.T if arr.shape == (4, 2) else arr)
    elif args.adapt_params:
        try:
            adapt_params_arr = _validate_adapt_params(
                np.array(json.loads(args.adapt_params), dtype=np.float64)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adaptive-parameters JSON: {exc.msg}") from exc

    return run_tracker(
        infile=args.input,
        varname=args.var,
        outfile=args.output,
        start_time=start_time,
        end_time=end_time,
        mode=args.mode,
        map_proj=args.map_proj,
        resolution=args.resolution,
        extent=args.extent,
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
        zones=zones_arr,
        adapt_params=adapt_params_arr,
        filter=args.filter,
        lmin=lmin,
        lmax=lmax,
        taper_points=args.taper,
        overlap=args.overlap,
        subgrid_refine=args.subgrid_refine,
    )
