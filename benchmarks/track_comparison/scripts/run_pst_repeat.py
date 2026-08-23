#!/usr/bin/env python3
"""Run one complete PyStormTracker repetition for the TRACK MSLP matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import sysconfig
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import xarray as xr

from pystormtracker.backends import (
    available_cpu_count,
    resolve_frame_workers,
    resolve_mge_workers,
    resolve_sht_threads,
)
from pystormtracker.hodges import constants
from pystormtracker.hodges.segments import TrackingSegment
from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.trackjson import write_trackjson
from pystormtracker.models.tracks import ProcessingStep, Tracks
from pystormtracker.preprocessing.spectral import SHTFilter

RESULT_ROOT_DEFAULT: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/pst_track_comparison-20260819-corrected"
)

F320_FULL: Final[Path] = Path(
    "/home/albert/PyStormTracker-Reference-Data/era5-2024/ERA5_mslp_6hr_2024_DET.nc"
)
F320_JAN: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/"
    "track_comparison-20260818/inputs/ERA5_mslp_6hr_2024-01_DET.nc"
)
REGULAR_2P5: Final[Path] = Path(
    "/home/albert/.cache/pystormtracker/era5_msl_2025-2026_djf_2.5x2.5.nc"
)
REGULAR_0P25: Final[Path] = Path(
    "/home/albert/.cache/pystormtracker/era5_msl_2025-2026_djf_0.25x0.25.nc"
)

NATIVE_THREAD_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "DUCC0_NUM_THREADS",
)
REQUIRED_THREAD_LIMIT_VARIABLES: Final[tuple[str, ...]] = NATIVE_THREAD_VARIABLES[:4]
PACKAGE_NAMES: Final[dict[str, str]] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "xarray": "xarray",
    "dask": "dask",
    "ducc0": "ducc0",
    "numba": "numba",
    "msgspec": "msgspec",
    "PyStormTracker": "PyStormTracker",
}
Backend = Literal["dask", "serial"]


@dataclass(frozen=True, slots=True)
class Case:
    """One scientific case from the completed TRACK benchmark."""

    name: str
    source: Path
    start: np.datetime64
    end: np.datetime64
    frames: int
    source_latitudes: int
    source_longitudes: int
    target: str
    target_latitudes: int
    target_longitudes: int


CASES: Final[dict[str, Case]] = {
    "f320_to_t42_january": Case(
        "f320_to_t42_january",
        F320_JAN,
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-31T18:00:00"),
        124,
        640,
        1280,
        "T42",
        64,
        128,
    ),
    "f320_to_t42_full_year": Case(
        "f320_to_t42_full_year",
        F320_FULL,
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-12-31T18:00:00"),
        1464,
        640,
        1280,
        "T42",
        64,
        128,
    ),
    "f320_to_f320_january": Case(
        "f320_to_f320_january",
        F320_JAN,
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-31T18:00:00"),
        124,
        640,
        1280,
        "F320",
        640,
        1280,
    ),
    "f320_to_f320_full_year": Case(
        "f320_to_f320_full_year",
        F320_FULL,
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-12-31T18:00:00"),
        1464,
        640,
        1280,
        "F320",
        640,
        1280,
    ),
    "regular-2p5-dec": Case(
        "regular-2p5-dec",
        REGULAR_2P5,
        np.datetime64("2025-12-01T00:00:00"),
        np.datetime64("2025-12-31T18:00:00"),
        124,
        73,
        144,
        "T42",
        64,
        128,
    ),
    "regular-2p5-season": Case(
        "regular-2p5-season",
        REGULAR_2P5,
        np.datetime64("2025-12-01T00:00:00"),
        np.datetime64("2026-02-28T18:00:00"),
        360,
        73,
        144,
        "T42",
        64,
        128,
    ),
    "regular-0p25-dec": Case(
        "regular-0p25-dec",
        REGULAR_0P25,
        np.datetime64("2025-12-01T00:00:00"),
        np.datetime64("2025-12-31T18:00:00"),
        124,
        721,
        1440,
        "T42",
        64,
        128,
    ),
    "regular-0p25-season": Case(
        "regular-0p25-season",
        REGULAR_0P25,
        np.datetime64("2025-12-01T00:00:00"),
        np.datetime64("2026-02-28T18:00:00"),
        360,
        721,
        1440,
        "T42",
        64,
        128,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PyStormTracker TRACK-comparison case."
    )
    parser.add_argument("case", nargs="?", choices=sorted(CASES))
    parser.add_argument("run_label", nargs="?")
    parser.add_argument(
        "--result-base",
        type=Path,
        default=Path(os.environ.get("RESULT_BASE", RESULT_ROOT_DEFAULT)),
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run the untimed small T42 and native F320-grid warmup instead.",
    )
    parser.add_argument(
        "--backend",
        choices=("dask", "serial"),
        default="dask",
        help="Execution backend for preprocessing and tracking.",
    )
    parser.add_argument("--frame-workers", type=int, default=None)
    parser.add_argument("--sht-threads", type=int, default=None)
    parser.add_argument("--mge-workers", type=int, default=None)
    args = parser.parse_args()
    if args.warmup:
        if args.case is not None or args.run_label is not None:
            parser.error("--warmup does not accept CASE or RUN_LABEL")
    elif args.case is None or args.run_label is None:
        parser.error("CASE and RUN_LABEL are required unless --warmup is used")
    return args


def _require_thread_limits() -> dict[str, str | None]:
    values = {name: os.environ.get(name) for name in NATIVE_THREAD_VARIABLES}
    missing = [name for name in REQUIRED_THREAD_LIMIT_VARIABLES if values[name] != "1"]
    if missing:
        raise RuntimeError(
            "benchmark requires native thread limits of 1; missing or invalid: "
            + ", ".join(missing)
        )
    return values


def _physical_cpu_count(affinity: list[int]) -> int | None:
    identities: set[tuple[str, str]] = set()
    for cpu in affinity:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = base / "physical_package_id"
        core = base / "core_id"
        try:
            identities.add((package.read_text().strip(), core.read_text().strip()))
        except OSError:
            return None
    return len(identities) if identities else None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or None
    return platform.processor() or None


def _git_sha() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "missing"
    return versions


def _ducc0_thread_pool() -> dict[str, int] | None:
    try:
        from ducc0 import misc

        return {
            "available_hardware_threads": int(misc.available_hardware_threads()),
            "thread_pool_size": int(misc.thread_pool_size()),
        }
    except (ImportError, AttributeError, TypeError):
        return None


def _execution_metadata(
    thread_limits: dict[str, str | None], tracker: HodgesTracker
) -> dict[str, object]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    logical_cpu_count = len(affinity) if affinity else os.cpu_count()
    process_cpu_count = getattr(os, "process_cpu_count", lambda: None)()
    return {
        "git_sha": _git_sha(),
        "python_version": sys.version.replace("\n", " "),
        "python_implementation": platform.python_implementation(),
        "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "package_versions": _package_versions(),
        "cpu_model": _cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "os_process_cpu_count": process_cpu_count,
        "logical_cpu_count": logical_cpu_count,
        "physical_cpu_count": _physical_cpu_count(affinity),
        "cpu_affinity": affinity,
        "available_cpu_count": available_cpu_count(),
        "frame_workers_requested": tracker.frame_workers,
        "resolved_frame_workers": resolve_frame_workers(
            tracker.frame_workers, tracker.backend
        ),
        "sht_threads_requested": tracker.sht_threads,
        "resolved_sht_threads": resolve_sht_threads(
            tracker.sht_threads, tracker.backend
        ),
        "mge_workers_requested": tracker.mge_workers,
        "resolved_mge_workers": resolve_mge_workers(
            tracker.mge_workers, tracker.backend
        ),
        "backend": tracker.backend,
        "native_thread_limits": thread_limits,
        "ducc0_thread_pool_before": _ducc0_thread_pool(),
    }


@contextmanager
def _time_dask_stages(
    workflow_start: float, timings: dict[str, float]
) -> Iterator[None]:
    """Time existing Dask boundaries without adding timers to library code."""
    import dask

    from pystormtracker.hodges import tracker as tracker_module
    from pystormtracker.hodges.segments import merge_segments

    original_compute = dask.compute
    original_merge = merge_segments
    compute_count = 0

    def timed_compute(*args: object, **kwargs: object) -> object:
        nonlocal compute_count
        started = time.perf_counter()
        result = original_compute(*args, **kwargs)
        finished = time.perf_counter()
        if compute_count == 0:
            timings["source_open_graph_preparation_wall_seconds"] = (
                started - workflow_start
            )
            timings["frame_stage_wall_seconds"] = finished - started
        elif compute_count == 1:
            timings["mge_segment_stage_wall_seconds"] = finished - started
        compute_count += 1
        return result

    def timed_merge(
        segment_tracks: Sequence[Tracks],
        segment_plan: Sequence[TrackingSegment],
        **kwargs: object,
    ) -> Tracks:
        started = time.perf_counter()
        result = original_merge(segment_tracks, segment_plan, **kwargs)
        timings["merge_splice_wall_seconds"] = time.perf_counter() - started
        return result

    dask.compute = timed_compute  # type: ignore[assignment]
    tracker_module.merge_segments = timed_merge
    try:
        yield
    finally:
        dask.compute = original_compute  # type: ignore[assignment]
        tracker_module.merge_segments = original_merge
        timings["dask_compute_calls"] = float(compute_count)


def _time_dimension(data: xr.DataArray) -> str:
    for name in ("time", "valid_time"):
        if name in data.dims:
            return name
    raise ValueError(f"could not find a time dimension in {data.dims}")


def _spatial_dimensions(data: xr.DataArray) -> tuple[str, str]:
    for lat_name in ("latitude", "lat"):
        for lon_name in ("longitude", "lon"):
            if lat_name in data.dims and lon_name in data.dims:
                return lat_name, lon_name
    raise ValueError(f"could not find latitude/longitude dimensions in {data.dims}")


def _open_selected_data(
    case: Case, backend: Backend
) -> tuple[xr.Dataset, xr.DataArray, dict[str, object]]:
    if not case.source.is_file():
        raise FileNotFoundError(case.source)
    dataset = xr.open_dataset(case.source)
    if "msl" not in dataset.data_vars:
        raise KeyError(f"{case.source} has no msl data variable")
    data = dataset["msl"]
    time_dim = _time_dimension(data)
    lat_dim, lon_dim = _spatial_dimensions(data)
    if int(data.sizes[lat_dim]) != case.source_latitudes:
        raise ValueError(
            f"{case.name}: expected {case.source_latitudes} latitudes, "
            f"got {data.sizes[lat_dim]}"
        )
    if int(data.sizes[lon_dim]) != case.source_longitudes:
        raise ValueError(
            f"{case.name}: expected {case.source_longitudes} longitudes, "
            f"got {data.sizes[lon_dim]}"
        )

    selected = data.sel({time_dim: slice(case.start, case.end)})
    selected_times = np.asarray(selected[time_dim].values, dtype="datetime64[ms]")
    expected_times = np.arange(
        case.start,
        case.end + np.timedelta64(6, "h"),
        np.timedelta64(6, "h"),
        dtype="datetime64[ms]",
    )
    if not np.array_equal(selected_times, expected_times):
        raise ValueError(
            f"{case.name}: selected timestamps do not equal the expected 6-hourly "
            "case coordinate"
        )
    if selected_times.size != case.frames:
        raise ValueError(
            f"{case.name}: expected {case.frames} selected frames, "
            f"got {selected_times.size}"
        )

    latitude = np.asarray(selected[lat_dim].values, dtype=np.float64)
    longitude = np.asarray(selected[lon_dim].values, dtype=np.float64)
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError(f"{case.name}: degenerate spatial coordinates")
    source_geometry = "GL" if float(np.ptp(np.diff(latitude))) > 1.0e-4 else "CC"
    if not bool(latitude[0] > latitude[-1]):
        raise ValueError(f"{case.name}: expected north-to-south latitude ordering")
    if not np.isclose(longitude[0], 0.0) or np.isclose(longitude[-1], 360.0):
        raise ValueError(f"{case.name}: unexpected longitude convention")

    if backend == "dask":
        selected = selected.chunk({time_dim: 1, lat_dim: -1, lon_dim: -1})
    chunks = None
    if selected.chunks is not None:
        chunks = {
            name: [int(value) for value in chunk_sizes]
            for name, chunk_sizes in zip(selected.dims, selected.chunks, strict=True)
        }
    source_metadata: dict[str, object] = {
        "path": str(case.source),
        "variable": "msl",
        "units": data.attrs.get("units"),
        "source_dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "selected_dimensions": {
            name: int(size) for name, size in selected.sizes.items()
        },
        "time_dimension": time_dim,
        "latitude_dimension": lat_dim,
        "longitude_dimension": lon_dim,
        "first_timestamp": str(selected_times[0]),
        "last_timestamp": str(selected_times[-1]),
        "frame_count": int(selected_times.size),
        "cadence_hours": 6,
        "source_geometry": source_geometry,
        "latitude_first": float(latitude[0]),
        "latitude_last": float(latitude[-1]),
        "longitude_first": float(longitude[0]),
        "longitude_last": float(longitude[-1]),
        "chunks": chunks,
    }
    return dataset, selected, source_metadata


def _tracker(
    backend: Backend,
    *,
    frame_workers: int | None,
    sht_threads: int | None,
    mge_workers: int | None,
) -> HodgesTracker:
    """Construct the public benchmark tracker configuration."""
    return HodgesTracker(
        # Filtering is constructed explicitly with the public SHTFilter below,
        # because the tracker API does not expose a target synthesis grid.
        lmin=None,
        lmax=None,
        taper_points=0,
        spectral_taper=0.1,
        min_object_grid_points=3,
        feature_refinement="bspline",
        track_smoopy_optimization_scale=0.01,
        w1=0.2,
        w2=0.8,
        dmax=6.5,
        phimax=1.0,
        mge_max_iterations=10,
        min_track_points=1,
        dmax_zones=constants.DEFAULT_DMAX_ZONES.copy(),
        adaptive_smoothness=constants.DEFAULT_ADAPTIVE_SMOOTHNESS.copy(),
        missing_frame_parameters=None,
        segment_frames=62,
        projection="global",
        exclude_boundary_extrema=False,
        backend=backend,
        frame_workers=frame_workers,
        sht_threads=sht_threads,
        mge_workers=mge_workers,
    )


def _scientific_metadata(tracker: HodgesTracker) -> dict[str, object]:
    return {
        "variable": "msl",
        "detection_mode": "min",
        "object_threshold_pa": -100.0,
        "lmin": 6,
        "lmax": 42,
        "spectral_taper_at_lmax": 0.1,
        "taper_points": 0,
        "feature_refinement": tracker.feature_refinement,
        "w1": tracker.w1,
        "w2": tracker.w2,
        "dmax": tracker.dmax,
        "dmax_zones": constants.DEFAULT_DMAX_ZONES.tolist(),
        "phimax": tracker.phimax,
        "adaptive_smoothness": constants.DEFAULT_ADAPTIVE_SMOOTHNESS.tolist(),
        "mge_max_iterations": tracker.mge_max_iterations,
        "min_track_points": tracker.min_track_points,
        "segment_frames": tracker.segment_frames,
        "time_step_hours": 6,
        "missing_frame_parameters": None,
        "min_object_grid_points": tracker.min_object_grid_points,
        "exclude_boundary_extrema": tracker.exclude_boundary_extrema,
        "group_adjacent_extrema": tracker.group_adjacent_extrema,
        "public_tracker_refinement_defaults": {
            "bspline_smoothing": tracker.bspline_smoothing,
            "bspline_max_iterations": tracker.bspline_max_iterations,
            "bspline_gradient_tolerance": tracker.bspline_gradient_tolerance,
            "track_smoopy_optimization_scale": tracker.track_smoopy_optimization_scale,
        },
    }


def _filter_data(
    data: xr.DataArray,
    case: Case,
    backend: Backend,
    sht_threads: int | None,
) -> xr.DataArray:
    output_kwargs: dict[str, object] = {}
    if case.target == "T42":
        output_kwargs = {
            "out_geometry": "GL",
            "out_ntheta": case.target_latitudes,
            "out_nphi": case.target_longitudes,
        }
    filtered = SHTFilter(
        lmin=6,
        lmax=42,
        taper_val=0.1,
        geometry="auto",
        sht_threads=sht_threads,
        **output_kwargs,
    ).filter(data, backend=backend)
    if not isinstance(filtered, xr.DataArray):
        raise TypeError("SHTFilter returned a non-xarray result")
    if filtered.sizes.get("latitude") != case.target_latitudes:
        raise ValueError(
            f"{case.name}: filtered latitude size is {filtered.sizes.get('latitude')}, "
            f"expected {case.target_latitudes}"
        )
    if filtered.sizes.get("longitude") != case.target_longitudes:
        raise ValueError(
            f"{case.name}: filtered longitude size is "
            f"{filtered.sizes.get('longitude')}, "
            f"expected {case.target_longitudes}"
        )
    return filtered


def _add_filter_metadata(tracks: Tracks, case: Case) -> Tracks:
    processing = tracks.metadata.processing + (
        ProcessingStep(
            "spectral_filter",
            True,
            {
                "method": "sht-ducc0",
                "input_mode": "PST-native in-memory T6-42 filter",
                "geometry": "auto",
                "lmin": 6,
                "lmax": 42,
                "spectral_taper": 0.1,
            },
        ),
    )
    if case.target == "T42":
        processing += (
            ProcessingStep(
                "sht_synthesis",
                True,
                {
                    "out_geometry": "GL",
                    "out_ntheta": case.target_latitudes,
                    "out_nphi": case.target_longitudes,
                },
            ),
        )
    return tracks.with_metadata(
        tracks.metadata.__class__(
            primary_variable=tracks.metadata.primary_variable,
            mode=tracks.metadata.mode,
            units=tracks.metadata.units,
            bounds=tracks.metadata.bounds,
            processing=processing,
        )
    )


def _run_case(
    case: Case,
    run_label: str,
    result_base: Path,
    backend: Backend,
    args: argparse.Namespace,
) -> None:
    thread_limits = _require_thread_limits()
    result = result_base / case.name / run_label
    if result.exists():
        entries = tuple(result.iterdir())
        precreated = {"run.log", "workflow.time"}
        if any(
            entry.name not in precreated or entry.stat().st_size != 0
            for entry in entries
        ):
            raise FileExistsError(f"refusing to overwrite {result}")
    else:
        result.mkdir(parents=True)
    output_path = result / "output.trackjson"

    workflow_start = time.perf_counter()
    dataset, selected, source_metadata = _open_selected_data(case, backend)
    tracker = _tracker(
        backend,
        frame_workers=args.frame_workers,
        sht_threads=args.sht_threads,
        mge_workers=args.mge_workers,
    )
    execution = _execution_metadata(thread_limits, tracker)
    scientific = _scientific_metadata(tracker)
    timings: dict[str, float] = {}
    try:
        filtered = _filter_data(selected, case, backend, tracker.sht_threads)
        with _time_dask_stages(workflow_start, timings):
            tracks = tracker.track(
                filtered,
                variable="msl",
                detection_mode="min",
                object_threshold=-100.0,
                time_step=np.timedelta64(6, "h"),
            )
        tracks = _add_filter_metadata(tracks, case)
        write_start = time.perf_counter()
        write_trackjson(tracks, output_path)
        write_end = time.perf_counter()
    finally:
        dataset.close()

    timings.setdefault(
        "source_open_graph_preparation_wall_seconds", write_start - workflow_start
    )
    timings["trackjson_write_wall_seconds"] = write_end - write_start
    timings["total_end_to_end_wall_seconds"] = write_end - workflow_start
    execution["ducc0_thread_pool_after"] = _ducc0_thread_pool()

    metadata: dict[str, object] = {
        "case": case.name,
        "run_label": run_label,
        "input": source_metadata,
        "target_grid": {
            "name": case.target,
            "geometry": "GL",
            "latitude_count": case.target_latitudes,
            "longitude_count": case.target_longitudes,
        },
        "execution": execution,
        "scientific_parameters": scientific,
        "output": {
            "path": str(output_path),
            "format": "TrackJSON/1.0",
            "tracks": len(tracks),
            "points": int(tracks.times.size),
            "trackjson_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "timing": timings,
    }
    (result / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_warmup(
    backend: Backend,
    *,
    frame_workers: int | None,
    sht_threads: int | None,
    mge_workers: int | None,
) -> None:
    _require_thread_limits()
    for case_name in ("f320_to_t42_january", "f320_to_f320_january"):
        case = CASES[case_name]
        dataset, selected, _metadata = _open_selected_data(case, backend)
        try:
            warm_data = selected.isel({_time_dimension(selected): slice(0, 4)})
            tracker = _tracker(
                backend,
                frame_workers=frame_workers,
                sht_threads=sht_threads,
                mge_workers=mge_workers,
            )
            filtered = _filter_data(warm_data, case, backend, tracker.sht_threads)
            tracker.track(
                filtered,
                variable="msl",
                detection_mode="min",
                object_threshold=-100.0,
                time_step=np.timedelta64(6, "h"),
            )
        finally:
            dataset.close()


def main() -> None:
    args = _parse_args()
    if args.warmup:
        _run_warmup(
            args.backend,
            frame_workers=args.frame_workers,
            sht_threads=args.sht_threads,
            mge_workers=args.mge_workers,
        )
    else:
        _run_case(
            CASES[args.case],
            args.run_label,
            args.result_base,
            args.backend,
            args,
        )


if __name__ == "__main__":
    main()
