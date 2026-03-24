from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing.derivatives import apply_wind_derivatives

# Use local test data generated from first frame
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

        div, vort = apply_wind_derivatives(u, v, engine=engine)

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
            assert rmse_vo < 1e-4


@pytest.mark.skipif(not HAS_DATA, reason="Local integration test data not found.")
def test_vorticity_internal_consistency() -> None:
    """
    Verifies that different backends are consistent with each other.
    """
    ds_uv = xr.open_dataset(WIND_FILE)
    u, v = ds_uv.u, ds_uv.v

    _, vort_ducc = apply_wind_derivatives(u, v, engine="ducc0")

    try:
        import shtns  # type: ignore[import-untyped]

        _ = shtns.sht(31, 31)  # Test import
        _, vort_shtns = apply_wind_derivatives(u, v, engine="shtns")
        # Ensure they are in the same ballpark (RMSE < 1e-4)
        rmse = np.sqrt(np.mean((vort_ducc.values - vort_shtns.values) ** 2))
        assert rmse < 1e-4
    except ImportError:
        pytest.skip("shtns not available for consistency check")
