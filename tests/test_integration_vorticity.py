from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing.kinematics import apply_vort_div

# Use local test data generated from first frame (Generated with NCL 6.6.2)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIND_FILE = os.path.join(BASE_DIR, "data/test/era5/era5_uv850_2025120100_2.5x2.5.nc")
VODIV_FILE = os.path.join(
    BASE_DIR, "data/test/era5/era5_vodv850_2025120100_2.5x2.5_ncl.nc"
)

HAS_DATA = os.path.exists(WIND_FILE) and os.path.exists(VODIV_FILE)


@pytest.mark.skipif(not HAS_DATA, reason="Local integration test data not found.")
def test_vorticity_divergence_parity_integration() -> None:
    """
    Verifies that all SHT backends produce results matching the NCL/Spherepack
    reference data.
    """
    ds_uv = xr.open_dataset(WIND_FILE)
    ds_ref = xr.open_dataset(VODIV_FILE)

    u, v = ds_uv.u, ds_uv.v
    vo_ref, dv_ref = ds_ref.vo, ds_ref.dv

    engines: list[Literal["shtns", "ducc0"]] = ["shtns", "ducc0"]

    for engine in engines:
        if engine == "shtns":
            pytest.importorskip("shtns")

        div, vort = apply_vort_div(u, v, sht_engine=engine)

        if engine == "ducc0":
            # ducc0/pyshtools should be bit-wise identical to NCL Spherepack
            np.testing.assert_allclose(
                vort.values, vo_ref.values, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                div.values, dv_ref.values, rtol=1e-12, atol=1e-12
            )
        else:
            # shtns uses different grid weights for regular grids, loose match
            # but same order of magnitude and general pattern.
            rmse_vo = np.sqrt(np.mean((vort.values - vo_ref.values) ** 2))
            assert rmse_vo < 5e-6  # Tightened from 1e-4, but allows for Fejer weights
            
            # Ensure high structural correlation with NCL reference
            corr = np.corrcoef(vort.values.flatten(), vo_ref.values.flatten())[0, 1]
            assert corr > 0.99


@pytest.mark.skipif(not HAS_DATA, reason="Local integration test data not found.")
def test_vorticity_internal_consistency() -> None:
    """
    Verifies that different backends are consistent with each other.
    """
    ds_uv = xr.open_dataset(WIND_FILE)
    u, v = ds_uv.u, ds_uv.v

    _, vort_ducc = apply_vort_div(u, v, sht_engine="ducc0")
    
    try:
        import pyshtools  # type: ignore[import-untyped]
        _, vort_shtools = apply_vort_div(u, v, sht_engine="shtools")
        
        rmse_shtools = np.sqrt(np.mean((vort_ducc.values - vort_shtools.values) ** 2))
        assert rmse_shtools < 1e-10  # Should be nearly identical
    except ImportError:
        pass

    try:
        import shtns  # type: ignore[import-untyped]

        _ = shtns.sht(31, 31)  # Test import
        _, vort_shtns = apply_vort_div(u, v, sht_engine="shtns")
        
        # Ensure ducc0 and shtns are highly correlated
        corr = np.corrcoef(vort_ducc.values.flatten(), vort_shtns.values.flatten())[0, 1]
        assert corr > 0.99

        # Ensure they are in the same ballpark (RMSE < 5e-6)
        rmse = np.sqrt(np.mean((vort_ducc.values - vort_shtns.values) ** 2))
        assert rmse < 5e-6
    except ImportError:
        pytest.skip("shtns not available for consistency check")
