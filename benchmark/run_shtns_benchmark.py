"""
SHTns vs ducc0 Spherical Harmonic Transform Benchmark Script.

This standalone benchmark script compares the performance and accuracy of the historical SHTns
implementation against the current ducc0 production engine on ERA5 reference data.

Historical SHTns implementation restored from commits prior to pull request #115 / commit 731301a.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Literal, cast

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray

try:
    import shtns  # type: ignore[import-untyped]

    SHTNS_AVAILABLE = True
except ImportError:
    SHTNS_AVAILABLE = False

# Ensure tests/ is in sys.path to import utils.py for dataset paths
sys.path.insert(0, os.path.abspath("tests"))
try:
    from utils import get_era5_msl_path, get_era5_uv_path, get_era5_vodv_path
except ImportError:
    raise ImportError("Could not import test utils. Ensure script is run from project root.")

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.models.constants import R_EARTH_METERS
from pystormtracker.preprocessing.kinematics import compute_vort_div as compute_vort_div_ducc0
from pystormtracker.preprocessing.spectral import _filter_ducc0_frame

# ==============================================================================
# Historical SHTns Implementation (from PyStormTracker git history)
# ==============================================================================

_thread_local = threading.local()


def _get_shtns_plan(
    nlat: int, nlon: int, lmax: int, polar_opt: float = 0.0
) -> shtns.sht:
    """Retrieves or creates an SHTns plan for the current thread."""
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}

    key = (nlat, nlon, lmax, polar_opt)
    if key not in _thread_local.cache:
        mmax = min(lmax, nlon // 2 - 1)
        sh = shtns.sht(lmax, mmax, norm=shtns.sht_fourpi)
        sh.set_grid(
            nlat,
            nlon,
            flags=shtns.sht_reg_poles | shtns.SHT_PHI_CONTIGUOUS,
            polar_opt=polar_opt,
        )
        _thread_local.cache[key] = sh

    return _thread_local.cache[key]


def filter_shtns_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    lat_reverse: bool = False,
    polar_opt: float = 0.0,
) -> NDArray[np.float64]:
    """Filters a single 2D spatial frame using SHTns."""
    if not SHTNS_AVAILABLE:
        raise ImportError("shtns is requested but not available.")

    if frame.ndim == 3:
        frame = frame[0]

    if lat_reverse:
        frame = frame[::-1, :]

    nlat, nlon = frame.shape
    grid_lmax = (nlat - 1) // 2
    if lmin > grid_lmax:
        raise ValueError(
            f"Unsupported shape for spectral filter: {frame.shape}. "
            f"Grid resolution (lmax={grid_lmax}) is too low for lmin={lmin}."
        )

    actual_lmax = min(lmax, grid_lmax)
    frame = np.ascontiguousarray(frame, dtype=np.float64)
    sh = _get_shtns_plan(nlat, nlon, actual_lmax, polar_opt=polar_opt)

    # Forward transform (Spatial -> Spectral)
    ylm = sh.analys(frame)

    # Apply Bandpass Mask (Zero out coefficients outside [lmin, actual_lmax])
    mask = (sh.l < lmin) | (sh.l > actual_lmax)
    ylm[mask] = 0.0

    # Backward transform (Spectral -> Spatial)
    out = cast(NDArray[np.float64], sh.synth(ylm))

    if lat_reverse:
        out = out[::-1, :]

    return out


def compute_vort_div_shtns(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    lat_reverse: bool = False,
    polar_opt: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes spatial divergence and relative vorticity using SHTns."""
    if not SHTNS_AVAILABLE:
        raise ImportError("shtns is requested but not available.")

    if u.ndim == 3:
        u = u[0]
    if v.ndim == 3:
        v = v[0]

    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u is {u.shape}, v is {v.shape}")

    if lat_reverse:
        u = u[::-1, :]
        v = v[::-1, :]

    ntheta, nphi = u.shape
    grid_lmax = (ntheta - 1) // 2
    if lmax is None:
        actual_lmax = grid_lmax
    else:
        actual_lmax = min(lmax, grid_lmax)

    sh = _get_shtns_plan(ntheta, nphi, actual_lmax, polar_opt=polar_opt)

    # SHTns spat_to_SHsphtor expects (v_theta, v_phi)
    # Theta is colatitude (0 at North Pole, increasing Southward), so v_theta = -v
    v_theta = np.ascontiguousarray(-v, dtype=np.float64)
    v_phi = np.ascontiguousarray(u, dtype=np.float64)

    slm = np.zeros(sh.nlm, dtype=np.complex128)
    tlm = np.zeros(sh.nlm, dtype=np.complex128)
    sh.spat_to_SHsphtor(v_theta, v_phi, slm, tlm)

    # Eigenvalue scaling
    l_arr = sh.l
    eigen = (l_arr * (l_arr + 1.0)) / R
    div_lm = -slm * eigen
    vort_lm = -tlm * eigen

    div = sh.synth(div_lm)
    vort = sh.synth(vort_lm)

    if lat_reverse:
        div = div[::-1, :]
        vort = vort[::-1, :]

    return div, vort


# ==============================================================================
# Benchmark Harness
# ==============================================================================


def compute_metrics(
    calc: NDArray[np.float64], ref: NDArray[np.float64]
) -> dict[str, float]:
    """Calculates RMSE, relative error, and Pearson correlation coefficient."""
    diff = calc.flatten() - ref.flatten()
    rmse = float(np.sqrt(np.mean(diff**2)))
    ref_mean = float(np.mean(np.abs(ref)))
    rel_err = float(rmse / ref_mean) if ref_mean > 0 else 0.0
    corr = float(np.corrcoef(calc.flatten(), ref.flatten())[0, 1])
    return {
        "rmse": rmse,
        "rel_error": rel_err,
        "correlation": corr,
    }


def run_spectral_benchmark(num_runs: int = 10) -> list[dict[str, Any]]:
    """Runs spectral filtering benchmark for MSL across resolutions and engines."""
    results: list[dict[str, Any]] = []

    test_cases = [
        {"res": "2.5x2.5", "lmin": 5, "lmax": 42, "suffix": "t5-42_ncl"},
        {"res": "2.5x2.5", "lmin": 0, "lmax": 42, "suffix": "t0-42_ncl"},
        {"res": "0.25x0.25", "lmin": 5, "lmax": 42, "suffix": "t5-42_ncl"},
        {"res": "0.25x0.25", "lmin": 0, "lmax": 42, "suffix": "t0-42_ncl"},
    ]

    for case in test_cases:
        res = case["res"]
        lmin, lmax = case["lmin"], case["lmax"]
        suffix = case["suffix"]

        src_file = get_era5_msl_path(res)
        ref_file = get_era5_msl_path(res, suffix=suffix)

        if not (os.path.exists(src_file) and os.path.exists(ref_file)):
            print(f"Skipping {res} spectral benchmark: data files missing.")
            continue

        loader_src = DataLoader(src_file)
        loader_ref = DataLoader(ref_file)

        msl = loader_src.ensure_open().msl.load().values
        ref = loader_ref.ensure_open().msl.load().values
        if msl.ndim == 3:
            msl = msl[0]
        if ref.ndim == 3:
            ref = ref[0]

        lat_reversed = loader_src.is_lat_reversed()

        # Benchmark ducc0
        times_ducc0: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            res_ducc0 = _filter_ducc0_frame(
                msl,
                lmin=lmin,
                lmax=lmax,
                lat_reverse=lat_reversed,
                taper_val=1.0,  # Sharp cutoff matching NCL reference
            )
            times_ducc0.append(time.perf_counter() - t0)

        metrics_ducc0 = compute_metrics(res_ducc0, ref)
        metrics_ducc0.update(
            {
                "engine": "ducc0",
                "resolution": res,
                "truncation": f"T{lmin}-{lmax}",
                "mean_time_sec": float(np.mean(times_ducc0)),
                "min_time_sec": float(np.min(times_ducc0)),
            }
        )
        results.append(metrics_ducc0)

        # Benchmark SHTns
        if SHTNS_AVAILABLE:
            times_shtns: list[float] = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                res_shtns = filter_shtns_frame(
                    msl,
                    lmin=lmin,
                    lmax=lmax,
                    lat_reverse=lat_reversed,
                    polar_opt=0.0,
                )
                times_shtns.append(time.perf_counter() - t0)

            metrics_shtns = compute_metrics(res_shtns, ref)
            metrics_shtns.update(
                {
                    "engine": "SHTns",
                    "resolution": res,
                    "truncation": f"T{lmin}-{lmax}",
                    "mean_time_sec": float(np.mean(times_shtns)),
                    "min_time_sec": float(np.min(times_shtns)),
                }
            )
            results.append(metrics_shtns)

    return results


def run_kinematics_benchmark(num_runs: int = 10) -> list[dict[str, Any]]:
    """Runs kinematics benchmark (vorticity/divergence) across engines."""
    results: list[dict[str, Any]] = []

    res = "0.25x0.25"
    wind_file = get_era5_uv_path(res)
    vodiv_file = get_era5_vodv_path(res)

    if not (os.path.exists(wind_file) and os.path.exists(vodiv_file)):
        print(f"Skipping {res} kinematics benchmark: data files missing.")
        return results

    loader_uv = DataLoader(wind_file)
    loader_ref = DataLoader(vodiv_file)

    ds_uv = loader_uv.ensure_open()
    ds_ref = loader_ref.ensure_open()

    u = ds_uv.u.load().values
    v = ds_uv.v.load().values
    vo_ref = ds_ref.vo.load().values
    dv_ref = ds_ref.dv.load().values

    if u.ndim == 3:
        u = u[0]
        v = v[0]
        vo_ref = vo_ref[0]
        dv_ref = dv_ref[0]

    lat_reversed = loader_uv.is_lat_reversed()

    # Benchmark ducc0
    times_ducc0: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        div_ducc0, vort_ducc0 = compute_vort_div_ducc0(
            u, v, lat_reverse=lat_reversed
        )
        times_ducc0.append(time.perf_counter() - t0)

    metrics_vo_ducc0 = compute_metrics(vort_ducc0, vo_ref)
    metrics_vo_ducc0.update(
        {
            "engine": "ducc0",
            "resolution": res,
            "variable": "Vorticity",
            "mean_time_sec": float(np.mean(times_ducc0)),
            "min_time_sec": float(np.min(times_ducc0)),
        }
    )
    results.append(metrics_vo_ducc0)

    # Benchmark SHTns
    if SHTNS_AVAILABLE:
        times_shtns: list[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            div_shtns, vort_shtns = compute_vort_div_shtns(
                u, v, lat_reverse=lat_reversed, polar_opt=0.0
            )
            times_shtns.append(time.perf_counter() - t0)

        metrics_vo_shtns = compute_metrics(vort_shtns, vo_ref)
        metrics_vo_shtns.update(
            {
                "engine": "SHTns",
                "resolution": res,
                "variable": "Vorticity",
                "mean_time_sec": float(np.mean(times_shtns)),
                "min_time_sec": float(np.min(times_shtns)),
            }
        )
        results.append(metrics_vo_shtns)

    return results


def run_polar_opt_comparison() -> dict[str, Any]:
    """Compares SHTns polar_opt=0.0 vs default 1e-10 accuracy and performance."""
    if not SHTNS_AVAILABLE:
        return {}

    res = "0.25x0.25"
    src_file = get_era5_msl_path(res)
    ref_file = get_era5_msl_path(res, suffix="t5-42_ncl")

    if not (os.path.exists(src_file) and os.path.exists(ref_file)):
        return {}

    loader_src = DataLoader(src_file)
    loader_ref = DataLoader(ref_file)

    msl = loader_src.ensure_open().msl.load().values
    ref = loader_ref.ensure_open().msl.load().values
    if msl.ndim == 3:
        msl = msl[0]
    if ref.ndim == 3:
        ref = ref[0]
    lat_reversed = loader_src.is_lat_reversed()

    # polar_opt = 0.0
    t0 = time.perf_counter()
    res_opt0 = filter_shtns_frame(
        msl, lmin=5, lmax=42, lat_reverse=lat_reversed, polar_opt=0.0
    )
    t_opt0 = time.perf_counter() - t0

    # polar_opt = 1e-10
    t0 = time.perf_counter()
    res_opt10 = filter_shtns_frame(
        msl, lmin=5, lmax=42, lat_reverse=lat_reversed, polar_opt=1e-10
    )
    t_opt10 = time.perf_counter() - t0

    diff_rmse = float(np.sqrt(np.mean((res_opt0 - res_opt10) ** 2)))
    m0 = compute_metrics(res_opt0, ref)
    m10 = compute_metrics(res_opt10, ref)

    return {
        "polar_opt_0.0": {"rmse_vs_ncl": m0["rmse"], "time_sec": t_opt0},
        "polar_opt_1e-10": {"rmse_vs_ncl": m10["rmse"], "time_sec": t_opt10},
        "rmse_difference": diff_rmse,
    }


def main() -> None:
    print("=" * 70)
    print("      PyStormTracker: SHTns vs ducc0 Comparison Benchmark")
    print("=" * 70)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"ducc0 Version:  {ducc0.__version__}")
    print(f"SHTns Available: {SHTNS_AVAILABLE}")
    if SHTNS_AVAILABLE:
        print(f"SHTns Module:   {shtns.__file__}")
    print("=" * 70)

    print("\n[1/3] Running Spectral Filtering Benchmark (MSL)...")
    spectral_results = run_spectral_benchmark(num_runs=5)

    print("\n[2/3] Running Kinematic Derivatives Benchmark (Vorticity)...")
    kinematics_results = run_kinematics_benchmark(num_runs=5)

    print("\n[3/3] Evaluating SHTns Polar Optimization Impact...")
    polar_results = run_polar_opt_comparison()

    all_data = {
        "python_version": sys.version.split()[0],
        "ducc0_version": ducc0.__version__,
        "shtns_available": SHTNS_AVAILABLE,
        "spectral_filtering": spectral_results,
        "kinematics": kinematics_results,
        "polar_optimization": polar_results,
    }

    out_file = os.path.join("benchmark", "shtns_vs_ducc0_results.json")
    with open(out_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nResults saved to {out_file}\n")

    # Display Tables
    print("\n### Spectral Filtering (MSL) Accuracy & Performance")
    print(
        f"{'Engine':<8} | {'Resolution':<10} | {'Trunc':<6} | {'RMSE (Pa)':<12} | "
        f"{'Rel. Error':<12} | {'Correlation':<16} | {'Time (s)':<10}"
    )
    print("-" * 88)
    for r in spectral_results:
        print(
            f"{r['engine']:<8} | {r['resolution']:<10} | {r['truncation']:<6} | "
            f"{r['rmse']:<12.8e} | {r['rel_error']:<12.4e} | {r['correlation']:<16.12f} | "
            f"{r['min_time_sec']:<10.6f}"
        )

    print("\n### Kinematic Derivatives (Vorticity) Accuracy & Performance")
    print(
        f"{'Engine':<8} | {'Resolution':<10} | {'RMSE (s^-1)':<14} | "
        f"{'Correlation':<16} | {'Time (s)':<10}"
    )
    print("-" * 64)
    for r in kinematics_results:
        print(
            f"{r['engine']:<8} | {r['resolution']:<10} | {r['rmse']:<14.8e} | "
            f"{r['correlation']:<16.12f} | {r['min_time_sec']:<10.6f}"
        )


if __name__ == "__main__":
    main()
