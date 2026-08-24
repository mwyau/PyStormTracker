#!/usr/bin/env python3
"""Profile one warmed PST F320 frame without changing production code.

The helper deliberately calls the same private detector and spline entry
points used by :class:`HodgesTracker`.  It reports the two output resolutions
used by the TRACK comparison, plus a decomposition of the rectangular spline
preparation and the local GDFP calls.  It is a benchmark helper, not a
production timing path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Final

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline

from pystormtracker.hodges.detector import (
    _detect_track_rectangular_candidates,
    detect_hodges_frame,
)
from pystormtracker.preprocessing.spectral import _apply_bandpass_mask_to_alm
from pystormtracker.refinement.bspline import (
    RectangularGridPreparation,
    _solve_cached_rectangular_coefficients,
    build_bspline_surface,
    prepare_rectangular_grid,
    refine_bspline_feature_point,
)

DEFAULT_SOURCE: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/"
    "track_comparison-20260818/inputs/ERA5_mslp_6hr_2024-01_DET.nc"
)
DEFAULT_REPETITIONS: Final[int] = 3
L_MIN: Final[int] = 6
L_MAX: Final[int] = 42
TAPER: Final[float] = 0.1
THRESHOLD: Final[float] = -100.0
OPTIMIZATION_SCALE: Final[float] = 0.01
MAX_ITERATIONS: Final[int] = 100
GRADIENT_TOLERANCE: Final[float] = 1.0e-5

type FloatArray = NDArray[np.float64]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--sht-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frame < 0 or args.repetitions <= 0 or args.sht_threads <= 0:
        parser.error("frame, repetitions, and sht-threads must be positive")
    return args


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _grid_coordinates(nlat: int, nlon: int) -> tuple[FloatArray, FloatArray]:
    latitudes = np.asarray(
        90.0 - np.degrees(ducc0.misc.GL_thetas(nlat)), dtype=np.float64
    )
    longitudes = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    return latitudes, longitudes


def _read_frame(
    source: Path, frame_index: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    with xr.open_dataset(source) as dataset:
        data = dataset["msl"]
        latitudes = np.asarray(data["latitude"].values, dtype=np.float64)
        longitudes = np.asarray(data["longitude"].values, dtype=np.float64)
        frame = np.asarray(data.isel(time=frame_index).values, dtype=np.float64)
    return frame, latitudes, longitudes


def _materialize_once(source: Path, frame_index: int) -> FloatArray:
    with xr.open_dataset(source) as dataset:
        return np.asarray(
            dataset["msl"].isel(time=frame_index).values, dtype=np.float64
        )


def _spectral_frame(
    frame: FloatArray,
    nlat: int,
    nlon: int,
    sht_threads: int,
) -> tuple[FloatArray, dict[str, float]]:
    """Run exactly the 2-D SHT analysis/mask/synthesis sequence."""
    nphi = min(L_MAX, nlon // 2 - 1)
    started = time.perf_counter()
    alm = ducc0.sht.analysis_2d(
        map=np.expand_dims(frame, axis=0),
        spin=0,
        lmax=L_MAX,
        mmax=nphi,
        geometry="GL",
        nthreads=sht_threads,
    )
    analysis_finished = time.perf_counter()
    _apply_bandpass_mask_to_alm(alm, L_MIN, L_MAX, nphi, taper_val=TAPER)
    mask_finished = time.perf_counter()
    output = np.asarray(
        ducc0.sht.synthesis_2d(
            alm=alm,
            spin=0,
            lmax=L_MAX,
            mmax=nphi,
            ntheta=nlat,
            nphi=nlon,
            geometry="GL",
            nthreads=sht_threads,
        )[0],
        dtype=np.float64,
    )
    finished = time.perf_counter()
    return output, {
        "analysis_seconds": analysis_finished - started,
        "mask_seconds": mask_finished - analysis_finished,
        "synthesis_seconds": finished - mask_finished,
        "spectral_seconds": finished - started,
    }


def _rectangular_spline_parts(
    frame: FloatArray,
    latitudes: FloatArray,
    longitudes: FloatArray,
    grid: RectangularGridPreparation,
) -> dict[str, float]:
    """Time FITPACK reference and cached fixed-system spline stages."""
    started = time.perf_counter()
    x = grid.sorted_longitudes
    x_order = grid.longitude_order
    f1_finished = time.perf_counter()

    y = grid.sorted_latitudes
    y_order = grid.latitude_order
    f2_finished = time.perf_counter()

    z = frame[y_order, :][:, x_order]
    f3_finished = time.perf_counter()
    extended_x = np.concatenate((x, [x[0] + 360.0]))
    extended_z = np.concatenate((z, z[:, :1]), axis=1)
    f4_finished = time.perf_counter()

    spline_obj = RectBivariateSpline(
        extended_x,
        y,
        extended_z.T,
        kx=3,
        ky=3,
        s=0.0,
    )
    f5_finished = time.perf_counter()

    fp_tuple = spline_obj.tck
    x_knots = np.asarray(fp_tuple[0], dtype=np.float64)
    y_knots = np.asarray(fp_tuple[1], dtype=np.float64)
    coeffs_raw = np.asarray(fp_tuple[2], dtype=np.float64)
    f6_finished = time.perf_counter()

    reference_finished = time.perf_counter()
    cached_started = time.perf_counter()
    cached_coeffs = _solve_cached_rectangular_coefficients(grid, extended_z)
    cached_finished = time.perf_counter()
    nx_knots = len(x_knots) - 4
    ny_knots = len(y_knots) - 4
    reference_coeffs = coeffs_raw.reshape(nx_knots, ny_knots)
    if cached_coeffs.shape != reference_coeffs.shape:
        raise AssertionError("cached FITPACK coefficient shape is not exact")

    return {
        "f1_longitude_normalization_order_seconds": f1_finished - started,
        "f2_latitude_order_seconds": f2_finished - f1_finished,
        "f3_frame_reorder_seconds": f3_finished - f2_finished,
        "f4_periodic_extension_seconds": f4_finished - f3_finished,
        "f5_fitpack_reference_construction_seconds": f5_finished - f4_finished,
        "f6_tck_extraction_seconds": f6_finished - f5_finished,
        "fitpack_reference_total_seconds": reference_finished - started,
        "cached_coefficient_solve_seconds": cached_finished - cached_started,
        "spline_parts_seconds": cached_finished - started,
    }


def _candidate_count_and_refinements(
    frame: FloatArray,
    latitudes: FloatArray,
    longitudes: FloatArray,
    grid: RectangularGridPreparation | None = None,
) -> tuple[dict[str, object], dict[str, FloatArray]]:
    candidate_started = time.perf_counter()
    candidate_lats, candidate_lons, candidate_values, object_ids = (
        _detect_track_rectangular_candidates(
            frame,
            latitudes,
            longitudes,
            intensity_threshold=THRESHOLD,
            is_min=True,
            min_grid_points=3,
            grid=grid,
        )
    )
    candidate_seconds = time.perf_counter() - candidate_started

    build_started = time.perf_counter()
    surface_result = build_bspline_surface(
        frame,
        latitudes,
        longitudes,
        periodic_x=True,
        smoothing=0.0,
        grid=grid,
    )
    build_seconds = time.perf_counter() - build_started
    if surface_result.surface is None:
        raise RuntimeError(f"spline construction failed: {surface_result.status}")

    refine_started = time.perf_counter()
    success_count = 0
    failure_count = 0
    for index in range(candidate_values.size):
        result = refine_bspline_feature_point(
            surface_result.surface,
            float(candidate_lats[index]),
            float(candidate_lons[index]),
            is_minimum=True,
            initial_value=float(candidate_values[index]),
            optimization_scale=OPTIMIZATION_SCALE,
            max_iterations=MAX_ITERATIONS,
            gradient_tolerance=GRADIENT_TOLERANCE,
        )
        if result.status == "success":
            success_count += 1
        else:
            failure_count += 1
    refine_seconds = time.perf_counter() - refine_started
    return {
        "candidate_seconds": candidate_seconds,
        "candidate_count": int(candidate_values.size),
        "object_count": int(np.unique(object_ids).size),
        "spline_build_seconds": build_seconds,
        "refinement_seconds": refine_seconds,
        "gdfp_calls": int(candidate_values.size),
        "successful_refinements": success_count,
        "failed_refinements": failure_count,
        "average_iterations": None,
    }, {
        "candidate_lats": candidate_lats,
        "candidate_lons": candidate_lons,
        "candidate_values": candidate_values,
        "object_ids": object_ids,
    }


def _profile_target(
    raw_frame: FloatArray,
    target_name: str,
    nlat: int,
    nlon: int,
    sht_threads: int,
    repetitions: int,
) -> dict[str, object]:
    output_latitudes, output_longitudes = _grid_coordinates(nlat, nlon)
    grid = prepare_rectangular_grid(
        output_latitudes,
        output_longitudes,
        periodic_x=True,
    )
    filtered, _ = _spectral_frame(raw_frame, nlat, nlon, sht_threads)

    # Warm DUCC, SciPy, the detector's Numba-independent Python path, and GDFP.
    _candidate_count_and_refinements(
        filtered, output_latitudes, output_longitudes, grid
    )
    detect_hodges_frame(
        filtered,
        np.datetime64("2024-01-01T00:00:00"),
        output_latitudes,
        output_longitudes,
        intensity_threshold=THRESHOLD,
        mode="min",
        min_grid_points=3,
        feature_refinement="bspline",
        track_smoopy_optimization_scale=OPTIMIZATION_SCALE,
        periodic_x=True,
        rectangular_grid=grid,
    )

    stage_samples: dict[str, list[float]] = {}
    full_samples: list[float] = []
    candidate_samples: list[dict[str, object]] = []
    spline_samples: list[dict[str, float]] = []
    refinement_samples: list[dict[str, object]] = []

    for _ in range(repetitions):
        _, spectral = _spectral_frame(raw_frame, nlat, nlon, sht_threads)
        for key, value in spectral.items():
            stage_samples.setdefault(key, []).append(value)
        filtered, _ = _spectral_frame(raw_frame, nlat, nlon, sht_threads)

        candidate_info, _ = _candidate_count_and_refinements(
            filtered, output_latitudes, output_longitudes, grid
        )
        candidate_samples.append(candidate_info)
        spline_samples.append(
            _rectangular_spline_parts(
                filtered,
                output_latitudes,
                output_longitudes,
                grid,
            )
        )
        refinement_samples.append(candidate_info)

        started = time.perf_counter()
        detect_hodges_frame(
            filtered,
            np.datetime64("2024-01-01T00:00:00"),
            output_latitudes,
            output_longitudes,
            intensity_threshold=THRESHOLD,
            mode="min",
            min_grid_points=3,
            feature_refinement="bspline",
            track_smoopy_optimization_scale=OPTIMIZATION_SCALE,
            periodic_x=True,
            rectangular_grid=grid,
        )
        full_samples.append(time.perf_counter() - started)

    def median_info(key: str) -> float:
        return _median([float(item[key]) for item in spline_samples])

    median_candidate = _median(
        [float(item["candidate_seconds"]) for item in candidate_samples]
    )
    median_refinement = _median(
        [float(item["refinement_seconds"]) for item in refinement_samples]
    )
    median_build = _median(
        [float(item["spline_build_seconds"]) for item in refinement_samples]
    )
    median_full = _median(full_samples)
    other = max(0.0, median_full - median_candidate - median_build - median_refinement)

    result: dict[str, object] = {
        "grid": {"name": target_name, "latitudes": nlat, "longitudes": nlon},
        "point_count": nlat * nlon,
        "wall_seconds": {key: _median(values) for key, values in stage_samples.items()},
        "candidate_detection_seconds": median_candidate,
        "spline_seconds": median_build,
        "fitpack_reference_seconds": _median(
            [
                float(item["f5_fitpack_reference_construction_seconds"])
                for item in spline_samples
            ]
        ),
        "cached_coefficient_solve_seconds": _median(
            [float(item["cached_coefficient_solve_seconds"]) for item in spline_samples]
        ),
        "gdfp_refinement_seconds": median_refinement,
        "duplicate_diagnostics_other_seconds": other,
        "complete_detector_refinement_seconds": median_full,
        "candidate_count": int(candidate_samples[-1]["candidate_count"]),
        "object_count": int(candidate_samples[-1]["object_count"]),
        "gdfp_calls": int(refinement_samples[-1]["gdfp_calls"]),
        "successful_refinements": int(refinement_samples[-1]["successful_refinements"]),
        "failed_refinements": int(refinement_samples[-1]["failed_refinements"]),
        "average_iterations": None,
        "spline_parts_seconds": {
            key: median_info(key)
            for key in spline_samples[0]
            if key != "spline_parts_seconds"
        },
    }
    return result


def main() -> None:
    args = _parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    frame, source_latitudes, source_longitudes = _read_frame(args.source, args.frame)
    read_samples = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        _materialize_once(args.source, args.frame)
        read_samples.append(time.perf_counter() - started)

    # The current benchmark source is already an F320 Gaussian grid.  The
    # coordinates are retained in the output to make source selection explicit.
    if frame.shape != (640, 1280):
        raise ValueError(f"expected F320 frame (640, 1280), got {frame.shape}")
    payload: dict[str, object] = {
        "source": str(args.source),
        "frame": args.frame,
        "source_shape": list(frame.shape),
        "source_latitude_endpoints": [
            float(source_latitudes[0]),
            float(source_latitudes[-1]),
        ],
        "source_longitude_endpoints": [
            float(source_longitudes[0]),
            float(source_longitudes[-1]),
        ],
        "sht_threads": args.sht_threads,
        "ducc0_thread_pool_size": int(ducc0.misc.thread_pool_size()),
        "read_materialize_seconds": _median(read_samples),
        "targets": [
            _profile_target(frame, "T42", 64, 128, args.sht_threads, args.repetitions),
            _profile_target(
                frame, "F320", 640, 1280, args.sht_threads, args.repetitions
            ),
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
