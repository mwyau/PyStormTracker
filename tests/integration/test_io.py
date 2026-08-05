from __future__ import annotations

import pytest
import xarray as xr
from utils import RAW_CONTENT_URL, fetch_era5_msl

from pystormtracker.io.data_loader import DataLoader


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


@pytest.mark.integration
@pytest.mark.parametrize(
    "url, expected_engine",
    [
        (f"{RAW_CONTENT_URL}era5_msl_2025-2026_djf_2.5x2.5.zarr", "zarr"),
    ],
)
def test_dataloader_remote_autodetection(url: str, expected_engine: str) -> None:
    """Integration test for remote Zarr loading with auto-detection."""
    pytest.importorskip("zarr")

    loader = DataLoader(url)
    ds = loader.ensure_open()
    assert isinstance(ds, xr.Dataset)
    assert loader.engine is None  # Auto-detection was used
    # Check that it was cached
    assert url in DataLoader._ds_cache


@pytest.mark.integration
def test_dataloader_local_zarr_archive() -> None:
    """Open the release Zarr archive after local extraction."""
    pytest.importorskip("zarr")

    zarr_path = fetch_era5_msl(format="zarr", local=True)
    loader = DataLoader(zarr_path)
    ds = loader.ensure_open()

    assert isinstance(ds, xr.Dataset)
    assert "msl" in ds.data_vars
    time_name, latitude_name, longitude_name = loader.get_coords()
    assert {time_name, latitude_name, longitude_name} <= set(ds.coords)
