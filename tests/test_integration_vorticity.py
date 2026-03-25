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

    engines: list[Literal["ducc0"]] = ["ducc0"]

    for engine in engines:
        div, vort = apply_vort_div(u, v, sht_engine=engine)

        if engine == "ducc0":
            # ducc0 should be bit-wise identical to NCL Spherepack (on the same grid)
            np.testing.assert_allclose(
                vort.values, vo_ref.values, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                div.values, dv_ref.values, rtol=1e-12, atol=1e-12
            )
