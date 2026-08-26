"""Run one explicit, reproducible PyStormTracker Hodges benchmark.

The runner accepts a user-supplied input and configuration and writes the
benchmark output and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import sysconfig
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from pystormtracker import __version__ as package_version
from pystormtracker.hodges import constants
from pystormtracker.hodges.progress import hodges_dask_progress
from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.trackjson import write_trackjson
from pystormtracker.models.tracks import ProcessingStep, Tracks, TracksMetadata
from pystormtracker.preprocessing.spectral import SHTFilter

_TIME_STEP_PATTERN = re.compile(r"([1-9][0-9]*)([smhD])$")
_NATIVE_THREAD_VARIABLES = (
    "DUCC0_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a finite number") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("expected a finite number")
    return parsed


def _datetime(value: str) -> np.datetime64:
    try:
        parsed = np.datetime64(value, "ms")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO timestamp") from exc
    if np.isnat(parsed):
        raise argparse.ArgumentTypeError("timestamp must not be NaT")
    return parsed


def _time_step(value: str) -> np.timedelta64:
    match = _TIME_STEP_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "expected a positive integer followed by s, m, h, or D"
        )
    amount, unit = match.groups()
    units = {"s": "s", "m": "m", "h": "h", "D": "D"}
    return np.timedelta64(int(amount), units[unit])


def _target_grid(value: str) -> tuple[str, int | None, int | None]:
    if value == "same":
        return "same", None, None
    parts = value.split(":")
    if len(parts) != 3 or parts[0] not in ("CC", "GL"):
        raise argparse.ArgumentTypeError(
            "target grid must be 'same' or GEOMETRY:NLAT:NLON (CC or GL)"
        )
    try:
        nlat, nlon = (int(part) for part in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "target grid latitude/longitude counts must be integers"
        ) from exc
    if nlat <= 0 or nlon <= 0:
        raise argparse.ArgumentTypeError("target grid counts must be positive")
    return parts[0], nlat, nlon


def _find_dimension(data: xr.DataArray, names: tuple[str, ...]) -> str:
    for name in names:
        if name in data.dims:
            return name
    raise ValueError(f"could not find one of {names!r} in dimensions {data.dims!r}")


def _matrix(path: Path, *, adaptive: bool) -> np.ndarray:
    with path.open(encoding="utf-8") as source:
        first_values = source.readline().split()
    skiprows = 1 if len(first_values) == 1 else 0
    values = np.loadtxt(path, dtype=np.float64, skiprows=skiprows)
    if adaptive and values.shape == (4, 2):
        values = values.T
    return np.atleast_2d(values)


def _dependency_versions() -> dict[str, str | None]:
    names = (
        "PyStormTracker",
        "numpy",
        "scipy",
        "numba",
        "ducc0",
        "xarray",
        "dask",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _runtime_metadata() -> dict[str, object]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    process_cpu_count = getattr(os, "process_cpu_count", lambda: None)()
    return {
        "python": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "os_cpu_count": os.cpu_count(),
        "process_cpu_count": process_cpu_count,
        "cpu_affinity": affinity,
        "native_thread_environment": {
            name: os.environ.get(name) for name in _NATIVE_THREAD_VARIABLES
        },
        "dependencies": _dependency_versions(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_data(
    args: argparse.Namespace,
) -> tuple[xr.Dataset, xr.DataArray, dict[str, object]]:
    dataset = xr.open_dataset(args.input, engine=args.engine, decode_times=True)
    if args.variable not in dataset.data_vars:
        available = list(dataset.data_vars)
        dataset.close()
        raise KeyError(
            f"variable {args.variable!r} is not present; available variables: "
            f"{available!r}"
        )
    data = dataset[args.variable]
    time_dim = _find_dimension(data, ("time", "valid_time"))
    lat_dim = _find_dimension(data, ("latitude", "lat"))
    lon_dim = _find_dimension(data, ("longitude", "lon"))

    if args.start_time is not None or args.end_time is not None:
        if args.start_time is None or args.end_time is None:
            dataset.close()
            raise ValueError("--start-time and --end-time must be supplied together")
        selected = data.sel({time_dim: slice(args.start_time, args.end_time)})
    elif args.frames is not None:
        selected = data.isel({time_dim: slice(0, args.frames)})
    else:
        selected = data

    if args.backend == "dask":
        selected = selected.chunk({time_dim: 1, lat_dim: -1, lon_dim: -1})

    times = np.asarray(selected[time_dim].values, dtype="datetime64[ms]")
    if times.size == 0:
        dataset.close()
        raise ValueError("selected input has no frames")
    if times.size > 1:
        differences = np.diff(times)
        if not np.all(differences == differences[0]):
            dataset.close()
            raise ValueError("selected input time coordinate is not regular")
        cadence_hours = float(differences[0] / np.timedelta64(1, "h"))
    else:
        cadence_hours = None

    source_metadata: dict[str, object] = {
        "path": str(args.input.resolve()),
        "sha256": args.input_sha256,
        "variable": args.variable,
        "units": data.attrs.get("units"),
        "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "selected_dimensions": {
            name: int(size) for name, size in selected.sizes.items()
        },
        "time_dimension": time_dim,
        "latitude_dimension": lat_dim,
        "longitude_dimension": lon_dim,
        "first_timestamp": str(times[0]),
        "last_timestamp": str(times[-1]),
        "frame_count": int(times.size),
        "cadence_hours": cadence_hours,
        "latitude_first": float(selected[lat_dim].values[0]),
        "latitude_last": float(selected[lat_dim].values[-1]),
        "longitude_first": float(selected[lon_dim].values[0]),
        "longitude_last": float(selected[lon_dim].values[-1]),
        "chunks": (
            {
                name: [int(size) for size in chunk_sizes]
                for name, chunk_sizes in zip(
                    selected.dims, selected.chunks, strict=True
                )
            }
            if selected.chunks is not None
            else None
        ),
    }
    return dataset, selected, source_metadata


def _filter_input(
    data: xr.DataArray,
    args: argparse.Namespace,
) -> tuple[xr.DataArray, dict[str, object]]:
    if args.lmin is None or args.lmax is None:
        if args.lmin is not None or args.lmax is not None:
            raise ValueError("--lmin and --lmax must be supplied together")
        geometry, nlat, nlon = args.target_grid
        if geometry != "same":
            raise ValueError("a target grid requires --lmin and --lmax")
        return data, {
            "enabled": False,
            "method": None,
            "lmin": None,
            "lmax": None,
            "taper_val": None,
            "source_geometry": args.source_geometry,
            "target_geometry": geometry,
            "target_latitudes": nlat,
            "target_longitudes": nlon,
        }

    geometry, nlat, nlon = args.target_grid
    spectral_taper = (
        args.spectral_taper
        if args.spectral_taper is not None
        else constants.SPECTRAL_TAPER_DEFAULT
    )
    kwargs: dict[str, object] = {}
    if geometry != "same":
        kwargs = {
            "out_geometry": geometry,
            "out_ntheta": nlat,
            "out_nphi": nlon,
        }
    filtered = SHTFilter(
        lmin=args.lmin,
        lmax=args.lmax,
        taper_val=spectral_taper,
        geometry=args.source_geometry,
        **kwargs,
    ).filter(data, backend=args.backend)
    if not isinstance(filtered, xr.DataArray):
        raise TypeError("SHTFilter returned a non-xarray result")
    return filtered, {
        "enabled": True,
        "method": "SHTFilter/ducc0",
        "lmin": args.lmin,
        "lmax": args.lmax,
        "taper_val": spectral_taper,
        "source_geometry": args.source_geometry,
        "target_geometry": geometry,
        "target_latitudes": nlat,
        "target_longitudes": nlon,
    }


def _tracker(args: argparse.Namespace) -> HodgesTracker:
    dmax_zones = (
        _matrix(args.dmax_zones, adaptive=False)
        if args.dmax_zones is not None
        else None
    )
    adaptive_smoothness = (
        _matrix(args.adaptive_smoothness, adaptive=True)
        if args.adaptive_smoothness is not None
        else None
    )
    return HodgesTracker(
        # Spectral preparation is explicit in _filter_input so target grids can
        # be part of a generic benchmark without coupling the tracker API to a
        # particular campaign.
        lmin=None,
        lmax=None,
        taper_points=0,
        spectral_taper=(
            args.spectral_taper
            if args.spectral_taper is not None
            else constants.SPECTRAL_TAPER_DEFAULT
        ),
        min_object_grid_points=(
            args.min_object_grid_points
            if args.min_object_grid_points is not None
            else constants.MIN_OBJECT_GRID_POINTS_DEFAULT
        ),
        feature_refinement=args.feature_refinement,
        track_smoopy_optimization_scale=(
            args.track_smoopy_optimization_scale
            if args.track_smoopy_optimization_scale is not None
            else constants.TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT
        ),
        w1=args.w1 if args.w1 is not None else constants.W1_DEFAULT,
        w2=args.w2 if args.w2 is not None else constants.W2_DEFAULT,
        dmax=args.dmax if args.dmax is not None else constants.DMAX_DEFAULT,
        phimax=args.phimax if args.phimax is not None else constants.PHIMAX_DEFAULT,
        mge_max_iterations=(
            args.mge_max_iterations
            if args.mge_max_iterations is not None
            else constants.MGE_MAX_ITERATIONS_DEFAULT
        ),
        min_track_points=(
            args.min_track_points
            if args.min_track_points is not None
            else constants.MIN_TRACK_POINTS_DEFAULT
        ),
        dmax_zones=dmax_zones,
        adaptive_smoothness=adaptive_smoothness,
        missing_frame_parameters=None,
        segment_frames=args.segment_frames,
        projection="global",
        group_adjacent_extrema=False,
        exclude_boundary_extrema=False,
        backend=args.backend,
        frame_workers=args.frame_workers,
        sht_threads=args.sht_threads,
        mge_workers=args.mge_workers,
    )


def _processing_metadata(
    tracks: Tracks,
    filter_metadata: dict[str, object],
    target_grid: tuple[str, int | None, int | None],
) -> Tracks:
    processing = tracks.metadata.processing
    if filter_metadata["enabled"]:
        processing += (ProcessingStep("spectral_filter", True, filter_metadata),)
    if target_grid[0] != "same":
        processing += (
            ProcessingStep(
                "sht_synthesis",
                True,
                {
                    "out_geometry": target_grid[0],
                    "out_ntheta": target_grid[1],
                    "out_nphi": target_grid[2],
                },
            ),
        )
    return tracks.with_metadata(
        TracksMetadata(
            primary_variable=tracks.metadata.primary_variable,
            mode=tracks.metadata.mode,
            units=tracks.metadata.units,
            bounds=tracks.metadata.bounds,
            processing=processing,
        )
    )


def _scientific_metadata(
    args: argparse.Namespace,
    tracker: HodgesTracker,
) -> dict[str, object]:
    return {
        "variable": args.variable,
        "detection_mode": args.detection_mode,
        "object_threshold": args.object_threshold,
        "feature_refinement": tracker.feature_refinement,
        "lmin": args.lmin,
        "lmax": args.lmax,
        "spectral_taper": (
            args.spectral_taper
            if args.spectral_taper is not None
            else constants.SPECTRAL_TAPER_DEFAULT
        ),
        "min_object_grid_points": tracker.min_object_grid_points,
        "track_smoopy_optimization_scale": tracker.track_smoopy_optimization_scale,
        "bspline_smoothing": tracker.bspline_smoothing,
        "bspline_max_iterations": tracker.bspline_max_iterations,
        "bspline_gradient_tolerance": tracker.bspline_gradient_tolerance,
        "w1": tracker.w1,
        "w2": tracker.w2,
        "dmax": tracker.dmax,
        "dmax_zones": tracker.dmax_zones.tolist(),
        "phimax": tracker.phimax,
        "adaptive_smoothness": tracker.adaptive_smoothness.tolist(),
        "mge_max_iterations": tracker.mge_max_iterations,
        "min_track_points": tracker.min_track_points,
        "segment_frames": tracker.segment_frames,
        "time_step": str(args.time_step) if args.time_step is not None else None,
        "missing_frame_parameters": None,
        "group_adjacent_extrema": tracker.group_adjacent_extrema,
        "exclude_boundary_extrema": tracker.exclude_boundary_extrema,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--variable", default="msl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--start-time", type=_datetime)
    parser.add_argument("--end-time", type=_datetime)
    parser.add_argument("--frames", type=_positive_int)
    parser.add_argument("--detection-mode", choices=("min", "max"), default="min")
    parser.add_argument("--backend", choices=("serial", "dask"), default="dask")
    parser.add_argument("--frame-workers", type=_positive_int)
    parser.add_argument("--sht-threads", type=_positive_int)
    parser.add_argument("--mge-workers", type=_positive_int)
    parser.add_argument(
        "--source-geometry",
        choices=("auto", "CC", "GL", "DH"),
        default="auto",
    )
    parser.add_argument(
        "--target-grid",
        type=_target_grid,
        default=("same", None, None),
    )
    parser.add_argument("--lmin", type=_nonnegative_int)
    parser.add_argument("--lmax", type=_nonnegative_int)
    parser.add_argument("--spectral-taper", type=_finite_float, default=None)
    parser.add_argument(
        "--feature-refinement",
        choices=(
            "grid",
            "quadratic",
            "spherical_quadratic",
            "bspline",
            "spherical_bspline",
        ),
        default="bspline",
    )
    parser.add_argument(
        "--track-smoopy-optimization-scale",
        type=_finite_float,
        default=None,
    )
    parser.add_argument("--object-threshold", type=_finite_float, default=None)
    parser.add_argument("--min-object-grid-points", type=_positive_int, default=None)
    parser.add_argument("--min-track-points", type=_positive_int, default=None)
    parser.add_argument("--w1", type=_finite_float, default=None)
    parser.add_argument("--w2", type=_finite_float, default=None)
    parser.add_argument("--dmax", type=_finite_float, default=None)
    parser.add_argument("--phimax", type=_finite_float, default=None)
    parser.add_argument("--mge-max-iterations", type=_positive_int, default=None)
    parser.add_argument("--segment-frames", type=_positive_int, default=62)
    parser.add_argument("--time-step", type=_time_step, default=None)
    parser.add_argument("--dmax-zones", type=Path, default=None)
    parser.add_argument("--adaptive-smoothness", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.frames is not None and (
        args.start_time is not None or args.end_time is not None
    ):
        raise ValueError("use either --frames or --start-time/--end-time")
    if args.lmin is not None and args.lmax is not None and args.lmin > args.lmax:
        raise ValueError("--lmin must not exceed --lmax")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset, selected, source_metadata = _selected_data(args)
    try:
        filter_started = time.perf_counter()
        filtered, filter_metadata = _filter_input(selected, args)
        filter_graph_seconds = time.perf_counter() - filter_started

        tracker = _tracker(args)
        tracking_started = time.perf_counter()
        progress_context = (
            hodges_dask_progress(False) if args.no_progress else nullcontext()
        )
        with progress_context:
            tracks = tracker.track(
                filtered,
                variable=args.variable,
                detection_mode=args.detection_mode,
                object_threshold=args.object_threshold,
                time_step=args.time_step,
            )
        tracking_seconds = time.perf_counter() - tracking_started

        tracks = _processing_metadata(tracks, filter_metadata, args.target_grid)
        export_started = time.perf_counter()
        write_trackjson(tracks, args.output)
        export_seconds = time.perf_counter() - export_started
    finally:
        dataset.close()

    elapsed_seconds = time.perf_counter() - started
    metadata: dict[str, Any] = {
        "runner": "benchmarks/run_benchmark_detailed.py",
        "input": source_metadata,
        "configuration": _scientific_metadata(args, tracker),
        "target_grid": {
            "geometry": args.target_grid[0],
            "latitude_count": args.target_grid[1],
            "longitude_count": args.target_grid[2],
        },
        "execution": {
            "backend": args.backend,
            "frame_workers_requested": args.frame_workers,
            "sht_threads_requested": args.sht_threads,
            "mge_workers_requested": args.mge_workers,
            "runtime": _runtime_metadata(),
        },
        "package": {
            "version": package_version,
            "module": "pystormtracker",
        },
        "timing": {
            "filter_graph_seconds": filter_graph_seconds,
            "tracking_seconds": tracking_seconds,
            "export_seconds": export_seconds,
            "runner_wall_seconds": elapsed_seconds,
        },
        "output": {
            "path": str(args.output.resolve()),
            "format": "TrackJSON/1.0",
            "sha256": _sha256(args.output),
            "bytes": args.output.stat().st_size,
            "tracks": len(tracks),
            "points": int(tracks.times.size),
        },
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["output"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
