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
from .models.tracker import Backend, Tracker
from .preprocessing.tracking import resolve_filter_bounds
from .simple.tracker import SimpleTracker
from .utils.cli import (
    finite_float,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)

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


def _validate_dmax_zones(zones: np.ndarray) -> np.ndarray:
    """Validate TRACK regional constraints as rows of five values."""
    zones = np.atleast_2d(zones).astype(np.float64, copy=False)
    if not np.isfinite(zones).all():
        raise ValueError("dmax_zones values must be finite")
    if zones.shape[1] != 5:
        raise ValueError(
            "dmax_zones must contain rows of [lon_min, lon_max, lat_min, lat_max, dmax]"
        )
    if np.any(zones[:, 0] >= zones[:, 1]) or np.any(zones[:, 2] >= zones[:, 3]):
        raise ValueError("dmax_zones minima must be less than maxima")
    if np.any(zones[:, 4] <= 0.0):
        raise ValueError("dmax_zones dmax values must be greater than zero")
    return zones


def _validate_adaptive_smoothness(params: np.ndarray) -> np.ndarray:
    """Validate adaptive smoothness thresholds and values."""
    if params.shape != (2, 4):
        raise ValueError("adaptive_smoothness parameters must have shape (2, 4)")
    if not np.isfinite(params).all():
        raise ValueError("adaptive_smoothness parameters must be finite")
    if np.any(np.diff(params[0]) < 0.0):
        raise ValueError(
            "adaptive_smoothness distance thresholds must be nondecreasing"
        )
    if np.any(params[1] < 0.0):
        raise ValueError("adaptive_smoothness values must be nonnegative")
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
        "-v",
        "--variable",
        required=True,
        help="Variable to track (e.g., 'vo', 'msl').",
    )
    required.add_argument(
        "-o",
        "--output",
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
        "--detection-mode",
        choices=["auto", "min", "max"],
        default="auto",
        help="Detection mode; inferred from known variable aliases.",
    )
    general.add_argument(
        "-p",
        "--projection",
        choices=["global", "nh_stereo", "sh_stereo", "healpix"],
        default="global",
        help="Map projection for detection. Default 'global'.",
    )
    general.add_argument(
        "-r",
        "--stereo-grid-spacing-km",
        type=positive_float,
        default=100.0,
        help="Grid spacing in km for stereographic projections. Default 100.0.",
    )
    general.add_argument(
        "-t",
        "--intensity-threshold",
        type=finite_float,
        default=None,
        help="Intensity threshold for features.",
    )
    general.add_argument(
        "-n", "--num", type=positive_int, help="Number of time steps to process."
    )
    general.add_argument(
        "-b",
        "--backend",
        choices=["serial", "mpi", "dask"],
        default=None,
        help="Parallel backend. Auto-detected by default.",
    )
    general.add_argument(
        "-w",
        "--workers",
        type=positive_int,
        default=None,
        help="Number of workers. Auto-detected for MPI. Sets Dask if not MPI.",
    )
    general.add_argument(
        "-c",
        "--chunk-size",
        type=positive_int,
        default=None,
        help="Detection steps per chunk. Backend default when omitted.",
    )
    general.add_argument(
        "-e",
        "--engine",
        choices=["h5netcdf", "netcdf4", "cfgrib"],
        default=None,
        help="Xarray engine for reading input.",
    )

    # 3. Scientific and Algorithm-Specific Options (Long-only)
    science = parser.add_argument_group("Scientific & Algorithm Options")
    science.add_argument(
        "--feature-point-method",
        choices=["grid", "quadratic"],
        default=None,
        help="Feature point extraction method ('grid' or 'quadratic').",
    )
    science.add_argument(
        "--search-window-size",
        type=positive_int,
        default=5,
        help="Search window size for local extrema (must be positive odd integer).",
    )
    science.add_argument(
        "--filter-lmin",
        type=nonnegative_int,
        default=None,
        help="Optional lower spectral filter bound; supply with --filter-lmax.",
    )
    science.add_argument(
        "--filter-lmax",
        type=nonnegative_int,
        default=None,
        help="Optional upper spectral filter bound; supply with --filter-lmin.",
    )
    science.add_argument(
        "--taper-points",
        type=nonnegative_int,
        default=0,
        help="Independent spatial taper width; zero disables tapering.",
    )
    science.add_argument(
        "--extent",
        type=_parse_extent,
        default=(-13000.0, 13000.0, -13000.0, 13000.0),
        help="Bounding box in km (xmin,xmax,ymin,ymax) for stereographic projections.",
    )
    science.add_argument(
        "--nside",
        type=positive_int,
        default=None,
        help="Target HEALPix resolution; derived from source grid when omitted.",
    )
    science.add_argument(
        "--min-grid-points",
        type=positive_int,
        default=None,
        help="Minimum grid points in an object before feature-point extraction.",
    )
    science.add_argument(
        "--w1",
        type=nonnegative_float,
        default=None,
        help="Cost weight for direction. Default 0.2.",
    )
    science.add_argument(
        "--w2",
        type=nonnegative_float,
        default=None,
        help="Cost weight for speed. Default 0.8.",
    )
    science.add_argument(
        "--dmax",
        type=positive_float,
        default=None,
        help="Max search radius in degrees. Default 6.5.",
    )
    science.add_argument(
        "--phimax",
        type=nonnegative_float,
        default=None,
        help="Smoothness penalty (static). Default 0.5.",
    )
    science.add_argument(
        "--min-lifetime-steps",
        type=positive_int,
        default=None,
        help="Min time steps for a valid track. Default 3.",
    )
    science.add_argument(
        "--max-missing-steps",
        type=nonnegative_int,
        default=None,
        help="Max consecutive missing frames. Default 0.",
    )

    zone_group = science.add_mutually_exclusive_group()
    zone_group.add_argument(
        "--dmax-zone-file",
        type=str,
        default=None,
        help="Path to legacy zone.dat file for regional DMAX.",
    )
    zone_group.add_argument(
        "--dmax-zones",
        type=str,
        default=None,
        help="JSON string defining regional DMAX zones.",
    )

    adapt_group = science.add_mutually_exclusive_group()
    adapt_group.add_argument(
        "--adaptive-smoothness-file",
        type=str,
        default=None,
        help="Path to legacy adapt.dat file for adaptive smoothness.",
    )
    adapt_group.add_argument(
        "--adaptive-smoothness",
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
        from .io.data_loader import DataLoader

        loader = DataLoader(args.input, engine=args.engine)
        ds = loader.ensure_open()
        time_dim, _lat, _lon = loader.get_coords()
        if time_dim in ds.coords:
            times = np.asarray(ds[time_dim].values)
            if len(times) > 0:
                num = min(args.num, len(times))
                start_time = times[0]
                end_time = times[num - 1]

    resolve_filter_bounds(args.filter_lmin, args.filter_lmax)

    dmax_zones_arr = None
    if args.dmax_zone_file:
        with open(args.dmax_zone_file) as f:
            first_line = f.readline().split()
            has_header = len(first_line) == 1
        dmax_zones_arr = _validate_dmax_zones(
            np.loadtxt(args.dmax_zone_file, skiprows=1 if has_header else 0)
        )
    elif args.dmax_zones:
        try:
            dmax_zones_arr = _validate_dmax_zones(
                np.array(json.loads(args.dmax_zones), dtype=np.float64)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid dmax_zones JSON: {exc.msg}") from exc

    adaptive_smoothness_arr = None
    if args.adaptive_smoothness_file:
        arr = np.loadtxt(args.adaptive_smoothness_file)
        adaptive_smoothness_arr = _validate_adaptive_smoothness(
            arr.T if arr.shape == (4, 2) else arr
        )
    elif args.adaptive_smoothness:
        try:
            adaptive_smoothness_arr = _validate_adaptive_smoothness(
                np.array(json.loads(args.adaptive_smoothness), dtype=np.float64)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adaptive_smoothness JSON: {exc.msg}") from exc

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
    if args.projection == "healpix":
        if detected_backend != "serial":
            raise ValueError("HealpixTracker supports only the serial backend.")
        if args.chunk_size is not None:
            raise ValueError("HealpixTracker does not support chunking.")
        if args.stereo_grid_spacing_km != 100.0 or args.extent != (
            -13000.0,
            13000.0,
            -13000.0,
            13000.0,
        ):
            raise ValueError(
                "Stereographic grid spacing in kilometres and extent options are "
                "not supported with HEALPix projection."
            )

        tracker = HealpixTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            min_lifetime_steps=args.min_lifetime_steps
            if args.min_lifetime_steps is not None
            else constants.LIFETIME_DEFAULT,
            max_missing_steps=args.max_missing_steps
            if args.max_missing_steps is not None
            else constants.MISSING_DEFAULT,
            min_grid_points=args.min_grid_points
            if args.min_grid_points is not None
            else constants.MIN_POINTS_DEFAULT,
            dmax_zones=dmax_zones_arr,
            adaptive_smoothness=adaptive_smoothness_arr,
            nside=args.nside,
            filter_lmin=args.filter_lmin,
            filter_lmax=args.filter_lmax,
            taper_points=args.taper_points,
            feature_point_method=args.feature_point_method
            if args.feature_point_method is not None
            else "quadratic",
        )
    elif args.algorithm == "hodges":
        if detected_backend != "serial":
            raise ValueError("HodgesTracker supports only the serial backend.")
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")

        tracker = HodgesTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            min_lifetime_steps=args.min_lifetime_steps
            if args.min_lifetime_steps is not None
            else constants.LIFETIME_DEFAULT,
            max_missing_steps=args.max_missing_steps
            if args.max_missing_steps is not None
            else constants.MISSING_DEFAULT,
            min_grid_points=args.min_grid_points
            if args.min_grid_points is not None
            else constants.MIN_POINTS_DEFAULT,
            dmax_zones=dmax_zones_arr,
            adaptive_smoothness=adaptive_smoothness_arr,
            projection=args.projection,
            stereo_grid_spacing_km=args.stereo_grid_spacing_km,
            extent=args.extent,
            filter_lmin=args.filter_lmin,
            filter_lmax=args.filter_lmax,
            taper_points=args.taper_points,
            search_window_size=args.search_window_size,
            feature_point_method=args.feature_point_method
            if args.feature_point_method is not None
            else "quadratic",
            chunk_size=args.chunk_size,
        )
    else:  # simple tracker
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")
        has_hodges_option = (
            args.min_grid_points is not None
            or args.w1 is not None
            or args.w2 is not None
            or args.dmax is not None
            or args.phimax is not None
            or args.min_lifetime_steps is not None
            or args.max_missing_steps is not None
            or args.dmax_zone_file is not None
            or args.dmax_zones is not None
            or args.adaptive_smoothness_file is not None
            or args.adaptive_smoothness is not None
        )
        if has_hodges_option:
            raise ValueError(
                "Hodges options (w1, w2, dmax, phimax, min_lifetime_steps, "
                "max_missing_steps, min_grid_points, dmax_zones, adaptive_smoothness) "
                "are not supported with SimpleTracker."
            )

        tracker = SimpleTracker(
            projection=args.projection,
            stereo_grid_spacing_km=args.stereo_grid_spacing_km,
            extent=args.extent,
            filter_lmin=args.filter_lmin,
            filter_lmax=args.filter_lmax,
            taper_points=args.taper_points,
            search_window_size=args.search_window_size,
            feature_point_method=args.feature_point_method
            if args.feature_point_method is not None
            else "grid",
            backend=detected_backend,
            workers=n_workers,
            chunk_size=args.chunk_size,
        )

    tracks = tracker.track(
        data=args.input,
        variable=args.variable,
        start_time=start_time,
        end_time=end_time,
        detection_mode=args.detection_mode,
        intensity_threshold=args.intensity_threshold,
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
