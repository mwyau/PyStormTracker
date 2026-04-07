from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
from testing_utils import get_era5_uv_path, get_era5_vodv_path

from pystormtracker.preprocessing.kinematics import apply_vort_div


@pytest.mark.integration
@pytest.mark.parametrize(
    ("res", "atol"),
    [
        ("2.5x2.5", 0.0),  # Bit-wise identical
        #("0.25x0.25", 5e-11),  # Near machine epsilon for large grid
    ],
)
def test_vorticity_divergence_parity_integration(res: str, atol: float) -> None:
    """
    Verifies that the ducc0 backend produces results matching the NCL/Spherepack
    reference data across different resolutions.
    """
    wind_file = get_era5_uv_path(res)
    vodiv_file = get_era5_vodv_path(res)

    if not (os.path.exists(wind_file) and os.path.exists(vodiv_file)):
        pytest.skip(f"Integration test data for {res} not found.")

    ds_uv = xr.open_dataset(wind_file)
    ds_ref = xr.open_dataset(vodiv_file)

    u, v = ds_uv.u, ds_uv.v
    vo_ref, dv_ref = ds_ref.vo, ds_ref.dv

    div, vort = apply_vort_div(u, v)

    # Validate against NCL Spherepack reference
    np.testing.assert_allclose(vort.values, vo_ref.values, rtol=atol, atol=atol)
    np.testing.assert_allclose(div.values, dv_ref.values, rtol=atol, atol=atol)
