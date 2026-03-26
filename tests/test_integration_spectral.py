from __future__ import annotations

import os
from typing import TypedDict

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
@pytest.mark.parametrize(
    "case", TEST_CASES, ids=["T0-42_2.5", "T5-42_2.5", "T0-42_0.25", "T5-42_0.25"]
)
def test_spectral_filter_era5_parity_integration(case: FilterTestCase) -> None:
    """
    Verifies that the SHT backend for filtering produces results matching
    the NCL reference data for MSL across multiple truncations and resolutions.
    """
    if not os.path.exists(case["ref"]):
        pytest.skip(f"Reference data not found: {case['ref']}")

    # Determine input file based on reference filename
    if "0.25x0.25" in case["ref"]:
        src_file = os.path.join(
            BASE_DIR, "data/test/era5/era5_msl_2025120100_0.25x0.25.nc"
        )
    else:
        src_file = MSL_FILE

    if not os.path.exists(src_file):
        pytest.skip(f"Source data not found: {src_file}")

    ds_msl = xr.open_dataset(src_file)
    ds_ref = xr.open_dataset(case["ref"])

    msl = ds_msl.msl
    ref = ds_ref.msl

    filt = SpectralFilter(lmin=case["lmin"], lmax=case["lmax"])
    import time

    start_time = time.perf_counter()
    filtered = filt.filter(msl)
    end_time = time.perf_counter()
    duration = end_time - start_time

    # Ensure extremely high structural correlation (> 0.9999)
    corr = np.corrcoef(filtered.values.flatten(), ref.values.flatten())[0, 1]
    assert corr > 0.9999, f"Low correlation for T{case['lmin']}-{case['lmax']}: {corr}"

    # Ensure RMSE is within acceptable bounds for large-scale field (MSL ~10^5)
    # ducc0: ~0.05 Pa RMSE vs NCL (modern implementation consistency)
    rmse = np.sqrt(np.mean((filtered.values - ref.values) ** 2))

    # Relative Error: RMSE / Mean Magnitude of the field
    ref_mean = np.mean(np.abs(ref.values))
    rel_error = rmse / ref_mean if ref_mean > 0 else 0.0

    print(
        f"\nT{case['lmin']}-{case['lmax']} "
        f"RMSE: {rmse:.8f}, RelError: {rel_error:.12f}, "
        f"Corr: {corr:.12f}, Time: {duration:.4f}s"
    )

    # Threshold is slightly higher for aliased 2.5 grid
    max_rmse = 0.1
    assert rmse < max_rmse, f"High RMSE for T{case['lmin']}-{case['lmax']}: {rmse}"
