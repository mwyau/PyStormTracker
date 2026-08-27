from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from pystormtracker.io.data_loader import DataLoader
from tests.utils import fetch_era5_msl, get_integration_msl_path


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


@pytest.mark.integration
@pytest.mark.data
@pytest.mark.parametrize(
    ("url", "expected_engine"),
    [
        (
            fetch_era5_msl(format="zarr"),
            "zarr",
        ),
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
    assert any(k[0] == url for k in DataLoader._ds_cache)


@pytest.mark.integration
def test_dataloader_local_zarr_roundtrip(tmp_path: Path) -> None:
    """Write the committed MSL input to temporary Zarr and load it back."""
    pytest.importorskip("zarr")

    zarr_path = tmp_path / "era5_msl.zarr"
    with xr.open_dataset(get_integration_msl_path(), engine="h5netcdf") as dataset:
        dataset.to_zarr(zarr_path, consolidated=False)

    loader = DataLoader(str(zarr_path))
    ds = loader.ensure_open()

    assert isinstance(ds, xr.Dataset)
    assert "msl" in ds.data_vars
    time_name, latitude_name, longitude_name = loader.get_coords()
    assert {time_name, latitude_name, longitude_name} <= set(ds.coords)
