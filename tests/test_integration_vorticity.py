from __future__ import annotations

import os
import numpy as np
import pytest
import xarray as xr
from pystormtracker.preprocessing.derivatives import apply_wind_derivatives

DATA_DIR = os.path.expanduser("~/PyStormTracker-Data")
UV_FILE = os.path.join(DATA_DIR, "era5_uv850_2025-2026_djf_2.5x2.5.nc")
VO_FILE = os.path.join(DATA_DIR, "era5_vo850_2025-2026_djf_2.5x2.5.nc")

HAS_DATA = os.path.exists(UV_FILE) and os.path.exists(VO_FILE)

@pytest.mark.skipif(not HAS_DATA, reason="ERA5 validation data not found in ~/PyStormTracker-Data")
def test_vorticity_era5_integration() -> None:
    ds_uv = xr.open_dataset(UV_FILE)
    ds_vo = xr.open_dataset(VO_FILE)

    # Use first time step for speed
    u = ds_uv.u.isel(valid_time=0, pressure_level=0)
    v = ds_uv.v.isel(valid_time=0, pressure_level=0)
    vo_ref = ds_vo.vo.isel(valid_time=0, pressure_level=0)

    # Standardize orientation
    if u.latitude[0] < u.latitude[-1]:
        u = u.sortby("latitude", ascending=False)
        v = v.sortby("latitude", ascending=False)
        vo_ref = vo_ref.sortby("latitude", ascending=False)

    engines = ["shtns", "ducc0"]
    results = {}

    for engine in engines:
        if engine == "shtns":
            pytest.importorskip("shtns")
        
        _, vort = apply_wind_derivatives(u, v, engine=engine)
        results[engine] = vort

    # 1. Internal Consistency Check
    # Spectral backends should be very close to each other
    if "shtns" in results and "ducc0" in results:
        diff_internal = results["shtns"].values - results["ducc0"].values
        rmse_int = np.sqrt(np.mean(diff_internal**2))
        # Internal consistency should be significantly better than ERA5 reference match
        assert rmse_int < 2e-5

    # 2. Loose Validation against ERA5 reference
    # We expect a discrepancy due to resolution/interpolation (as discovered in R&D)
    for engine, vo_calc in results.items():
        diff = vo_calc.values - vo_ref.values
        rmse = np.sqrt(np.mean(diff**2))
        # Ensure it's in the expected order of magnitude
        assert rmse < 1e-4
