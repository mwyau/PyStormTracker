from __future__ import annotations

import os
import time
from typing import Literal, TypedDict

import numpy as np
import pytest
from utils import get_era5_msl_path

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.preprocessing import SpectralFilter


class FilterTestCase(TypedDict, total=False):
    lmin: int
    lmax: int
    res: str
    suffix: str
    sht_engine: Literal["ducc0"]


# Test cases for different truncations
TEST_CASES: list[FilterTestCase] = [
    # 2.5-degree cases (ducc0 only)
    {
        "lmin": 5,
        "lmax": 42,
        "res": "2.5x2.5",
        "suffix": "t5-42_ncl",
        "sht_engine": "ducc0",
    },
    {
        "lmin": 0,
        "lmax": 42,
        "res": "2.5x2.5",
        "suffix": "t0-42_ncl",
        "sht_engine": "ducc0",
    },
    # 0.25-degree cases (ducc0)
    {
        "lmin": 5,
        "lmax": 42,
        "res": "0.25x0.25",
        "suffix": "t5-42_ncl",
        "sht_engine": "ducc0",
    },
    {
        "lmin": 0,
        "lmax": 42,
        "res": "0.25x0.25",
        "suffix": "t0-42_ncl",
        "sht_engine": "ducc0",
    },
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[
        "T5-42_2.5_ducc0",
        "T0-42_2.5_ducc0",
        "T5-42_0.25_ducc0",
        "T0-42_0.25_ducc0",
    ],
)
def test_spectral_filter_era5_parity_integration(case: FilterTestCase) -> None:
    """
    Verifies that the SHT backend for filtering produces results matching
    the NCL reference data for MSL across multiple truncations and resolutions.
    """
    sht_engine: Literal["ducc0"] = case.get("sht_engine", "ducc0")

    res = case["res"]
    suffix = case["suffix"]

    src_file = get_era5_msl_path(res)
    ref_file = get_era5_msl_path(res, suffix=suffix)

    if not os.path.exists(src_file):
        pytest.skip(f"Source data not found: {src_file}")
    if not os.path.exists(ref_file):
        pytest.skip(f"Reference data not found: {ref_file}")

    # 1. Load Data using DataLoader
    loader_src = DataLoader(src_file)
    loader_ref = DataLoader(ref_file)

    ds_msl = loader_src.ensure_open()
    ds_ref = loader_ref.ensure_open()

    msl = ds_msl.msl.load()
    ref = ds_ref.msl.load()

    # 2. Filter using SpectralFilter with auto-detected lat_reverse
    filt = SpectralFilter(
        lmin=case["lmin"],
        lmax=case["lmax"],
        lat_reverse=loader_src.is_lat_reversed(),
    )

    start_time = time.perf_counter()
    filtered = filt.filter(msl)
    end_time = time.perf_counter()
    duration = end_time - start_time

    # 3. Validation
    # Ensure extremely high structural correlation (> 0.9999)
    corr = np.corrcoef(filtered.values.flatten(), ref.values.flatten())[0, 1]
    assert corr > 0.9999, (
        f"Low correlation for T{case['lmin']}-{case['lmax']} ({sht_engine}): {corr}"
    )

    # Ensure RMSE is within acceptable bounds for large-scale field (MSL ~10^5)
    # ducc0/jax: ~0.05 Pa RMSE vs NCL (modern implementation consistency)
    rmse = np.sqrt(np.mean((filtered.values - ref.values) ** 2))

    # Relative Error: RMSE / Mean Magnitude of the field
    ref_mean = np.mean(np.abs(ref.values))
    rel_error = rmse / ref_mean if ref_mean > 0 else 0.0

    print(
        f"\nT{case['lmin']}-{case['lmax']} ({sht_engine}) "
        f"RMSE: {rmse:.8f}, RelError: {rel_error:.12f}, "
        f"Corr: {corr:.12f}, Time: {duration:.4f}s"
    )

    # Threshold is slightly higher for aliased 2.5 grid
    max_rmse = 0.1
    assert rmse < max_rmse, (
        f"High RMSE for T{case['lmin']}-{case['lmax']} ({sht_engine}): {rmse}"
    )
