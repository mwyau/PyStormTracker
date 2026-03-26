from __future__ import annotations

import os
from typing import Literal, TypedDict

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing import SpectralFilter

# Use local test data (Generated with NCL 6.6.2)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSL_FILE = os.path.join(BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5.nc")


class FilterTestCase(TypedDict):
    lmin: int
    lmax: int
    ref: str


# Test cases for different truncations
TEST_CASES: list[FilterTestCase] = [
    {
        "lmin": 5,
        "lmax": 42,
        "ref": os.path.join(
            BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5_t5-42_ncl.nc"
        ),
    },
    {
        "lmin": 0,
        "lmax": 42,
        "ref": os.path.join(
            BASE_DIR, "data/test/era5/era5_msl_2025120100_2.5x2.5_t0-42_ncl.nc"
        ),
    },
    {
        "lmin": 5,
        "lmax": 42,
        "ref": os.path.join(
            BASE_DIR, "data/test/era5/era5_msl_2025120100_0.25x0.25_t5-42_ncl.nc"
        ),
    },
    {
        "lmin": 0,
        "lmax": 42,
        "ref": os.path.join(
            BASE_DIR, "data/test/era5/era5_msl_2025120100_0.25x0.25_t0-42_ncl.nc"
        ),
    },
]

HAS_BASE_DATA = os.path.exists(MSL_FILE)


@pytest.mark.skipif(not HAS_BASE_DATA, reason="Base MSL test data not found.")
@pytest.mark.parametrize("engine", ["shtns", "ducc0"])
@pytest.mark.parametrize("case", TEST_CASES, ids=["T5-42_2.5", "T0-42_2.5", "T5-42_0.25", "T0-42_0.25"])
def test_spectral_filter_era5_parity_integration(
    case: FilterTestCase, engine: Literal["shtns", "ducc0"]
) -> None:
    """
    Verifies that all SHT backends for filtering produce results matching
    the NCL reference data for MSL across multiple truncations and resolutions.
    """
    if engine == "shtns":
        pytest.importorskip("shtns")

    if not os.path.exists(case["ref"]):
        pytest.skip(f"Reference data not found: {case['ref']}")

    # Determine input file based on reference filename
    if "0.25x0.25" in case["ref"]:
        src_file = os.path.join(BASE_DIR, "data/test/era5/era5_msl_2025120100_0.25x0.25.nc")
    else:
        src_file = MSL_FILE

    if not os.path.exists(src_file):
        pytest.skip(f"Source data not found: {src_file}")

    ds_msl = xr.open_dataset(src_file)
    ds_ref = xr.open_dataset(case["ref"])

    msl = ds_msl.msl
    ref = ds_ref.msl

    filt = SpectralFilter(lmin=case["lmin"], lmax=case["lmax"], sht_engine=engine)
    filtered = filt.filter(msl)

    # Ensure extremely high structural correlation (> 0.9999)
    corr = np.corrcoef(filtered.values.flatten(), ref.values.flatten())[0, 1]
    assert corr > 0.9999, (
        f"Low correlation for {engine} (T{case['lmin']}-{case['lmax']}): {corr}"
    )

    # Ensure RMSE is within acceptable bounds for large-scale field (MSL ~10^5)
    # shtns: ~0.46 Pa RMSE vs NCL (legacy Spherepack)
    # ducc0: ~0.05 Pa RMSE vs NCL (modern implementation consistency)
    rmse = np.sqrt(np.mean((filtered.values - ref.values) ** 2))
    print(
        f"\n{engine} (T{case['lmin']}-{case['lmax']}) RMSE: {rmse:.8f}, Correlation: {corr:.12f}"
    )

    max_rmse = 1.0 if engine == "shtns" else 0.1
    assert rmse < max_rmse, (
        f"High RMSE for {engine} (T{case['lmin']}-{case['lmax']}): {rmse}"
    )
