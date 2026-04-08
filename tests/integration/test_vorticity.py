from __future__ import annotations

import os

import numpy as np
import pytest
from testing_utils import get_era5_uv_path, get_era5_vodv_path

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.preprocessing.kinematics import Kinematics


@pytest.mark.integration
@pytest.mark.parametrize(
    ("res", "atol"),
    [
        ("2.5x2.5", 0.0),  # Bit-wise identical
        ("0.25x0.25", 5e-11),  # Near machine epsilon for large grid
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

    # 1. Load Data using DataLoader
    loader_uv = DataLoader(wind_file)
    loader_ref = DataLoader(vodiv_file)

    ds_uv = loader_uv.ensure_open()
    ds_ref = loader_ref.ensure_open()

    # Explicitly load into memory for parity comparison
    u, v = ds_uv.u.load(), ds_uv.v.load()
    vo_ref, dv_ref = ds_ref.vo.load(), ds_ref.dv.load()

    # 2. Compute using Kinematics with auto-detected lat_reverse
    calc = Kinematics(lat_reverse=loader_uv.is_lat_reversed())
    div, vort = calc.compute(u, v)

    # 3. Validate against NCL Spherepack reference
    np.testing.assert_allclose(vort.values, vo_ref.values, rtol=atol, atol=atol)
    np.testing.assert_allclose(div.values, dv_ref.values, rtol=atol, atol=atol)
