from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.preprocessing import SHTFilter
from tests.utils import get_integration_msl_path

pytestmark = pytest.mark.parity

REFERENCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "data"
    / "ncl"
    / "era5_msl_2025-12-01_0000_2.5x2.5_t5-42.nc"
)


def test_sht_filter_matches_ncl_t5_42() -> None:
    """Compare the bundled T5-42 spectral filter case with NCL output."""
    source_path = get_integration_msl_path()
    assert source_path.is_file(), f"NCL parity source data missing: {source_path}"
    assert REFERENCE_PATH.is_file(), (
        f"NCL parity reference data missing: {REFERENCE_PATH}"
    )

    source_loader = DataLoader(source_path)
    reference_loader = DataLoader(REFERENCE_PATH)
    source = (
        source_loader.ensure_open()
        .msl.sel(valid_time=np.datetime64("2025-12-01T00:00:00"))
        .load()
    )
    reference = reference_loader.ensure_open().msl.load()

    filtered = SHTFilter(
        lmin=5,
        lmax=42,
        lat_reverse=source_loader.is_lat_reversed(),
        taper_val=1.0,
    ).filter(source)

    correlation = np.corrcoef(filtered.values.ravel(), reference.values.ravel())[0, 1]
    rmse = np.sqrt(np.mean((filtered.values - reference.values) ** 2))

    assert correlation > 0.9999
    assert rmse < 0.1
