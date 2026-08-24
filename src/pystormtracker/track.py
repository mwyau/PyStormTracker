from __future__ import annotations

import argparse
import logging
import os
import re
import timeit
from argparse import Namespace
from typing import cast

import numpy as np

from .backends import Backend, resolve_dask_workers
from .healpix.constants import (
    DEFAULT_MSL_OBJECT_THRESHOLD as HEALPIX_DEFAULT_MSL_OBJECT_THRESHOLD,
)
from .healpix.constants import (
    DEFAULT_VO_OBJECT_THRESHOLD as HEALPIX_DEFAULT_VO_OBJECT_THRESHOLD,
)
from .healpix.constants import (
    SPECTRAL_TAPER_DEFAULT as HEALPIX_SPECTRAL_TAPER_DEFAULT,
)
from .healpix.tracker import HealpixTracker
from .hodges import constants
from .hodges.detector import DEFAULT_SEARCH_WINDOW_SIZE as HODGES_SEARCH_WINDOW_SIZE
from .hodges.progress import hodges_dask_progress
from .hodges.segments import DEFAULT_SEGMENT_FRAMES
from .hodges.tracker import HodgesTracker
from .io.format import SUPPORTED_FORMATS, SupportedFormat
from .models.tracker import Tracker
from .preprocessing.tracking import resolve_filter_bounds
from .simple.constants import (
    DEFAULT_MSL_FEATURE_THRESHOLD,
    DEFAULT_VO_FEATURE_THRESHOLD,
)
from .simple.constants import (
    DEFAULT_SEARCH_WINDOW_SIZE as SIMPLE_SEARCH_WINDOW_SIZE,
)
from .simple.tracker import SimpleTracker
from .utils.cli import (
    add_cli_observability_options,
    finite_float,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
)

LOGGER = logging.getLogger(__name__)

_TIME_STEP_PATTERN = re.compile(r"([1-9][0-9]*)([smhD])\Z")


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


def _parse_time_step(value: str) -> np.timedelta64:
    """Parse the CLI cadence syntax ``<positive integer><unit>``."""
    match = _TIME_STEP_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected a positive integer followed by s, m, h, or D "
            "(for example: 30m, 6h, or 1D)"
        )
    amount, unit = match.groups()
    try:
        amount_int = int(amount)
        if unit == "s":
            return np.timedelta64(amount_int, "s")
        if unit == "m":
            return np.timedelta64(amount_int, "m")
        if unit == "h":
            return np.timedelta64(amount_int, "h")
        return np.timedelta64(amount_int, "D")
    except (OverflowError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid time-step {value!r}") from exc


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
    add_cli_observability_options(parser)

    # 1. Required Arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("-i", "--input", required=True, help="Input dataset path.")
    required.add_argument(
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
        choices=["simple", "hodges", "healpix"],
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
        choices=["global", "nh_stereo", "sh_stereo"],
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
        "--object-threshold",
        type=finite_float,
        default=None,
        help=(
            "Object segmentation threshold for Hodges and HEALPix trackers; "
            f"Hodges defaults are {constants.DEFAULT_MSL_OBJECT_THRESHOLD:g} for "
            f"MSL and {constants.DEFAULT_VO_OBJECT_THRESHOLD:g} for vo; "
            f"HEALPix defaults are {HEALPIX_DEFAULT_MSL_OBJECT_THRESHOLD:g} for "
            f"MSL and {HEALPIX_DEFAULT_VO_OBJECT_THRESHOLD:g} for vo."
        ),
    )
    general.add_argument(
        "--feature-threshold",
        type=finite_float,
        default=None,
        help=(
            "Feature detection threshold for SimpleTracker; defaults are "
            f"{DEFAULT_MSL_FEATURE_THRESHOLD:g} for MSL and "
            f"{DEFAULT_VO_FEATURE_THRESHOLD:g} for vo."
        ),
    )
    general.add_argument(
        "-n",
        "--n-frames",
        type=positive_int,
        dest="n_frames",
        help="Number of time steps to process.",
    )
    general.add_argument(
        "-b",
        "--backend",
        choices=["serial", "mpi", "dask"],
        default=None,
        help="Parallel backend. Defaults to 'dask' ('mpi' if MPI detected).",
    )
    general.add_argument(
        "-w",
        "--workers",
        type=positive_int,
        default=None,
        help=(
            "Number of generic Dask workers for Simple/HEALPix. Defaults to "
            "available CPU concurrency; not accepted for Hodges."
        ),
    )
    general.add_argument(
        "--frame-workers",
        type=positive_int,
        default=None,
        help=(
            "Hodges Dask frame-processing tasks (source read through detection "
            "and refinement)."
        ),
    )
    general.add_argument(
        "--sht-threads",
        type=positive_int,
        default=None,
        help=(
            "Hodges DUCC0 spherical-harmonic-transform threads per active "
            "frame/SHT task."
        ),
    )
    general.add_argument(
        "--mge-workers",
        type=positive_int,
        default=None,
        help="Hodges Dask MGE segment-linking tasks running concurrently.",
    )
    general.add_argument(
        "-c",
        "--segment-frames",
        type=positive_int,
        default=None,
        dest="segment_frames",
        help=(
            "MGE temporal segment length for Hodges/HEALPix trackers. "
            f"Tracker default ({DEFAULT_SEGMENT_FRAMES}) when omitted. "
            "Not used by SimpleTracker."
        ),
    )
    general.add_argument(
        "--no-segmentation",
        action="store_true",
        default=False,
        dest="no_segmentation",
        help="Disable temporal segmentation (run monolithic tracking).",
    )
    general.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help=(
            "Disable interactive Hodges Dask progress. Progress is otherwise shown "
            "when standard error is a terminal."
        ),
    )
    general.add_argument(
        "-e",
        "--engine",
        choices=["h5netcdf", "netcdf4", "cfgrib"],
        default=None,
        help="Xarray engine for reading input.",
    )

    # 3. Scientific and Algorithm-Specific Options
    science = parser.add_argument_group("Scientific & Algorithm Options")
    science.add_argument(
        "--feature-refinement",
        choices=[
            "grid",
            "quadratic",
            "spherical_quadratic",
            "bspline",
            "spherical_bspline",
        ],
        default=None,
        help=(
            "Subgrid feature-point refinement method: 'grid' (no subgrid "
            "refinement), 'quadratic' (local quadratic), 'spherical_quadratic', "
            "'bspline' (TRACK/SMOOPY-compatible rectangular B-spline), or "
            "'spherical_bspline' (spherical B-spline)."
            " Defaults: Simple='grid', Hodges='bspline', HEALPix='quadratic'."
        ),
    )
    science.add_argument(
        "--search-window-size",
        type=positive_int,
        default=None,
        help=(
            "Search window size for local extrema (must be positive odd integer). "
            f"Used by SimpleTracker (default {SIMPLE_SEARCH_WINDOW_SIZE}); "
            f"Hodges uses its TRACK detector default ({HODGES_SEARCH_WINDOW_SIZE})."
        ),
    )
    science.add_argument(
        "--lmin",
        type=nonnegative_int,
        default=None,
        help="Optional lower spectral filter bound; supply with --lmax.",
    )
    science.add_argument(
        "--lmax",
        type=nonnegative_int,
        default=None,
        help="Optional upper spectral filter bound; supply with --lmin.",
    )
    science.add_argument(
        "--taper-points",
        type=nonnegative_int,
        default=0,
        help="Independent spatial taper width; zero disables tapering.",
    )
    science.add_argument(
        "--spectral-taper",
        type=float,
        default=None,
        help=(
            "Hodges/HEALPix spectral coefficient taper in (0, 1]; "
            f"defaults are Hodges={constants.SPECTRAL_TAPER_DEFAULT:g} and "
            f"HEALPix={HEALPIX_SPECTRAL_TAPER_DEFAULT:g}. Not used by SimpleTracker."
        ),
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
        "--min-object-grid-points",
        type=positive_int,
        default=None,
        help=(
            "Minimum grid points in an object before feature-point extraction. "
            f"Default {constants.MIN_OBJECT_GRID_POINTS_DEFAULT}."
        ),
    )
    science.add_argument(
        "--w1",
        type=nonnegative_float,
        default=None,
        help=f"Cost weight for direction. Default {constants.W1_DEFAULT:g}.",
    )
    science.add_argument(
        "--w2",
        type=nonnegative_float,
        default=None,
        help=f"Cost weight for speed. Default {constants.W2_DEFAULT:g}.",
    )
    science.add_argument(
        "--dmax",
        type=positive_float,
        default=None,
        help=f"Max search radius in degrees. Default {constants.DMAX_DEFAULT:g}.",
    )
    science.add_argument(
        "--phimax",
        type=nonnegative_float,
        default=None,
        help=f"Smoothness penalty (static). Default {constants.PHIMAX_DEFAULT:g}.",
    )
    science.add_argument(
        "--min-track-points",
        type=positive_int,
        default=None,
        help=(
            "Min time steps for a valid track. "
            f"Default {constants.MIN_TRACK_POINTS_DEFAULT}."
        ),
    )
    science.add_argument(
        "--mge-max-iterations",
        type=positive_int,
        default=None,
        help=(
            "Maximum MGE iteration rounds. "
            f"Default {constants.MGE_MAX_ITERATIONS_DEFAULT}."
        ),
    )
    science.add_argument(
        "--time-step",
        type=_parse_time_step,
        default=None,
        help=(
            "Expected input cadence as a positive integer plus s, m, h, or D "
            "(e.g. '6h', '30m', or '1D')."
        ),
    )

    science.add_argument(
        "--dmax-zones",
        type=str,
        default=None,
        help=(
            "Path to regional DMAX definitions file "
            "(rows of lon_min, lon_max, lat_min, lat_max, dmax)."
        ),
    )
    science.add_argument(
        "--adaptive-smoothness",
        type=str,
        default=None,
        help="Path to adaptive smoothness parameters file (2x4 or 4x2 matrix).",
    )
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    """
    Main entry point for the track command.
    """
    start_time = None
    end_time = None

    if args.n_frames is not None:
        from .io.data_loader import DataLoader

        loader = DataLoader(args.input, engine=args.engine)
        ds = loader.ensure_open()
        time_dim, _lat, _lon = loader.get_coords()
        if time_dim in ds.coords:
            times = np.asarray(ds[time_dim].values)
            if len(times) > 0:
                num = min(args.n_frames, len(times))
                start_time = times[0]
                end_time = times[num - 1]

    resolve_filter_bounds(args.lmin, args.lmax)

    dmax_zones_arr = None
    if args.dmax_zones:
        with open(args.dmax_zones) as f:
            first_line = f.readline().split()
            has_header = len(first_line) == 1
        dmax_zones_arr = _validate_dmax_zones(
            np.loadtxt(args.dmax_zones, skiprows=1 if has_header else 0)
        )

    adaptive_smoothness_arr = None
    if args.adaptive_smoothness:
        arr = np.loadtxt(args.adaptive_smoothness)
        adaptive_smoothness_arr = _validate_adaptive_smoothness(
            arr.T if arr.shape == (4, 2) else arr
        )

    # Auto-detect backend
    detected_backend: Backend = "dask"
    if args.backend:
        detected_backend = args.backend
    elif is_mpi_env():
        detected_backend = "mpi"

    if args.algorithm == "hodges":
        if args.workers is not None:
            raise ValueError(
                "--workers is not supported with Hodges; use "
                "--frame-workers, --sht-threads, or --mge-workers"
            )
    elif any(
        value is not None
        for value in (args.frame_workers, args.sht_threads, args.mge_workers)
    ):
        raise ValueError(
            "--frame-workers, --sht-threads, and --mge-workers are only "
            "supported with Hodges"
        )

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
                LOGGER.warning("MPI environment detected but 'mpiexec' is missing")
                detected_backend = "dask"
                use_mpi = False

    if use_mpi:
        if not is_mpi_env():
            LOGGER.warning(
                "Warning: MPI backend selected but no MPI environment detected "
                "(e.g., OMPI_COMM_WORLD_SIZE not set)."
            )
            LOGGER.warning("Ensure you are running with 'mpirun' or 'mpiexec'.")

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
                LOGGER.warning(
                    "Warning: MPI environment detected but mpi4py is not installed. "
                    "Falling back."
                )
            detected_backend = "dask"
            use_mpi = False

    timer: dict[str, float] = {}
    if rank == 0:
        timer["total"] = timeit.default_timer()
        LOGGER.info("Using backend: %s", detected_backend)
        if detected_backend == "dask":
            LOGGER.debug(
                "Resolved Dask worker count: %d", resolve_dask_workers(n_workers)
            )
        elif n_workers:
            LOGGER.debug("Resolved worker count: %d", n_workers)

    # Determine segment size configuration
    if args.no_segmentation and args.segment_frames is not None:
        raise ValueError("cannot specify both --segment-frames and --no-segmentation")

    segment_frames_explicit: int | None = None
    has_explicit_segment = args.segment_frames is not None or args.no_segmentation
    if args.segment_frames is not None:
        segment_frames_explicit = args.segment_frames

    tracker_workers = None if detected_backend == "mpi" else n_workers

    # Resolve threshold
    if args.algorithm == "simple":
        if args.object_threshold is not None:
            raise ValueError("--object-threshold is not supported with SimpleTracker.")
        effective_threshold = args.feature_threshold
    else:
        if args.feature_threshold is not None:
            raise ValueError(
                "--feature-threshold is only supported with SimpleTracker."
            )
        effective_threshold = args.object_threshold

    time_step_td: np.timedelta64 | None = args.time_step

    # Validate options against selected tracker and instantiate tracker
    tracker: Tracker
    if args.algorithm == "healpix":
        if args.projection != "global":
            raise ValueError("Projections are not supported with HealpixTracker.")
        if args.time_step is not None:
            raise ValueError("time_step is not supported with HealpixTracker.")
        if args.search_window_size is not None:
            raise ValueError("search_window_size is not supported with HealpixTracker.")
        if args.stereo_grid_spacing_km != 100.0 or args.extent != (
            -13000.0,
            13000.0,
            -13000.0,
            13000.0,
        ):
            raise ValueError(
                "Stereographic grid spacing in kilometres and extent options are "
                "not supported with HealpixTracker."
            )

        tracker = HealpixTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            mge_max_iterations=args.mge_max_iterations
            if args.mge_max_iterations is not None
            else constants.MGE_MAX_ITERATIONS_DEFAULT,
            min_track_points=args.min_track_points
            if args.min_track_points is not None
            else constants.MIN_TRACK_POINTS_DEFAULT,
            min_object_grid_points=args.min_object_grid_points
            if args.min_object_grid_points is not None
            else constants.MIN_OBJECT_GRID_POINTS_DEFAULT,
            dmax_zones=dmax_zones_arr,
            adaptive_smoothness=adaptive_smoothness_arr,
            nside=args.nside,
            lmin=args.lmin,
            lmax=args.lmax,
            taper_points=args.taper_points,
            spectral_taper=args.spectral_taper
            if args.spectral_taper is not None
            else 0.1,
            feature_refinement=args.feature_refinement
            if args.feature_refinement is not None
            else "quadratic",
            backend=detected_backend,
            workers=tracker_workers,
            segment_frames=segment_frames_explicit
            if has_explicit_segment
            else DEFAULT_SEGMENT_FRAMES,
        )
        tracks = tracker.track(
            data=args.input,
            variable=args.variable,
            start_time=start_time,
            end_time=end_time,
            detection_mode=args.detection_mode,
            object_threshold=effective_threshold,
            engine=args.engine,
        )
    elif args.algorithm == "hodges":
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")
        if args.search_window_size is not None:
            raise ValueError("search_window_size is not supported with HodgesTracker.")

        tracker = HodgesTracker(
            w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
            w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
            dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
            phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
            mge_max_iterations=args.mge_max_iterations
            if args.mge_max_iterations is not None
            else constants.MGE_MAX_ITERATIONS_DEFAULT,
            min_track_points=args.min_track_points
            if args.min_track_points is not None
            else constants.MIN_TRACK_POINTS_DEFAULT,
            min_object_grid_points=args.min_object_grid_points
            if args.min_object_grid_points is not None
            else constants.MIN_OBJECT_GRID_POINTS_DEFAULT,
            dmax_zones=dmax_zones_arr,
            adaptive_smoothness=adaptive_smoothness_arr,
            projection=args.projection,
            stereo_grid_spacing_km=args.stereo_grid_spacing_km,
            extent=args.extent,
            lmin=args.lmin,
            lmax=args.lmax,
            taper_points=args.taper_points,
            spectral_taper=args.spectral_taper
            if args.spectral_taper is not None
            else 1.0,
            feature_refinement=args.feature_refinement
            if args.feature_refinement is not None
            else "bspline",
            backend=detected_backend,
            frame_workers=args.frame_workers,
            sht_threads=args.sht_threads,
            mge_workers=args.mge_workers,
            segment_frames=segment_frames_explicit
            if has_explicit_segment
            else DEFAULT_SEGMENT_FRAMES,
        )
        progress_override: bool | None = False if args.no_progress else None
        with hodges_dask_progress(progress_override):
            tracks = tracker.track(
                data=args.input,
                variable=args.variable,
                start_time=start_time,
                end_time=end_time,
                time_step=time_step_td,
                detection_mode=args.detection_mode,
                object_threshold=effective_threshold,
                engine=args.engine,
            )
    else:  # simple tracker
        if args.nside is not None:
            raise ValueError("nside is only supported with HEALPix projection.")
        has_hodges_option = (
            args.mge_max_iterations is not None
            or args.min_object_grid_points is not None
            or args.w1 is not None
            or args.w2 is not None
            or args.dmax is not None
            or args.phimax is not None
            or args.min_track_points is not None
            or args.dmax_zones is not None
            or args.adaptive_smoothness is not None
            or args.spectral_taper is not None
            or args.time_step is not None
            or args.segment_frames is not None
            or args.no_segmentation
        )
        if has_hodges_option:
            raise ValueError(
                "Hodges options (w1, w2, dmax, phimax, mge_max_iterations, "
                "min_track_points, min_object_grid_points, dmax_zones, "
                "adaptive_smoothness, spectral_taper, time_step, segment_frames) "
                "are not supported with SimpleTracker."
            )

        tracker = SimpleTracker(
            projection=args.projection,
            stereo_grid_spacing_km=args.stereo_grid_spacing_km,
            extent=args.extent,
            lmin=args.lmin,
            lmax=args.lmax,
            taper_points=args.taper_points,
            search_window_size=args.search_window_size
            if args.search_window_size is not None
            else 5,
            feature_refinement=args.feature_refinement
            if args.feature_refinement is not None
            else "grid",
            backend=detected_backend,
            workers=tracker_workers,
        )
        tracks = tracker.track(
            data=args.input,
            variable=args.variable,
            start_time=start_time,
            end_time=end_time,
            detection_mode=args.detection_mode,
            feature_threshold=effective_threshold,
            engine=args.engine,
        )

    if rank == 0:
        num_tracks = len(tracks)
        LOGGER.info(
            "Tracking completed: %d tracks / %d points",
            num_tracks,
            int(tracks.times.size),
        )

        timer["export"] = timeit.default_timer()
        selected_format = (
            None
            if args.format in (None, "auto")
            else cast("SupportedFormat", args.format)
        )
        tracks.write(args.output, format=selected_format)
        timer["export"] = timeit.default_timer() - timer["export"]

        LOGGER.info("Output written to %s in %.4fs", args.output, timer["export"])

        timer["total"] = timeit.default_timer() - timer["total"]
        LOGGER.info("Total tracking time: %.4fs", timer["total"])
