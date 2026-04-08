from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from utils import RAW_CONTENT_URL

from pystormtracker.io.data_loader import DataLoader


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


def test_dataloader_init() -> None:
    loader = DataLoader("test.nc")
    assert loader.pathname == Path("test.nc")
    assert loader.engine is None


def test_dataloader_init_with_engine() -> None:
    loader = DataLoader("test.nc", engine="h5netcdf")
    assert loader.engine == "h5netcdf"


@patch("xarray.open_dataset")
def test_ensure_open_netcdf(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader = DataLoader("test.nc")
    ds = loader.ensure_open()
    assert ds == mock_ds
    # Now defaults to None for standard xarray detection
    mock_open.assert_called_once_with(Path("test.nc"), engine=None, chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_grib(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader = DataLoader("test.grib")
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(Path("test.grib"), engine="cfgrib", chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_zarr(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader = DataLoader("test.zarr")
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(Path("test.zarr"), engine="zarr", chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_zarr_dir(mock_open: MagicMock, tmp_path: Path) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    zarr_dir = tmp_path / "test_data"
    zarr_dir.mkdir()
    (zarr_dir / ".zmetadata").touch()
    loader = DataLoader(zarr_dir)
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(zarr_dir, engine="zarr", chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_caching(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader1 = DataLoader("test.nc")
    loader2 = DataLoader("test.nc")
    loader1.ensure_open()
    loader2.ensure_open()
    mock_open.assert_called_once()


@patch("xarray.open_dataset")
def test_get_coords_mapping(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.coords = ["time", "lat", "lon"]
    mock_open.return_value = mock_ds
    loader = DataLoader("test.nc")
    time, lat, lon = loader.get_coords()
    assert time == "time"
    assert lat == "lat"
    assert lon == "lon"


def test_is_lat_reversed_direct() -> None:
    """Test is_lat_reversed with direct DataArray input."""
    lats = xr.DataArray([90, 80, 70], dims="lat", coords={"lat": [90, 80, 70]})
    ds = xr.Dataset({"var": lats})
    loader = DataLoader(ds)
    assert loader.is_lat_reversed() is True

    lats_asc = xr.DataArray([-90, -80], dims="lat", coords={"lat": [-90, -80]})
    ds_asc = xr.Dataset({"var": lats_asc})
    loader_asc = DataLoader(ds_asc)
    assert loader_asc.is_lat_reversed() is False


@pytest.mark.integration
@pytest.mark.parametrize(
    "url, expected_engine",  # noqa: PT006
    [
        (f"{RAW_CONTENT_URL}era5_msl_2025-2026_djf_2.5x2.5.zarr", "zarr"),
    ],
)
def test_dataloader_remote_autodetection(url: str, expected_engine: str) -> None:
    """Integration test for remote Zarr loading with auto-detection."""
    loader = DataLoader(url)
    ds = loader.ensure_open()
    assert isinstance(ds, xr.Dataset)
    assert loader.engine is None  # Auto-detection was used
    # Check that it was cached
    assert url in DataLoader._ds_cache


@patch("xarray.open_dataset")
@patch("importlib.util.find_spec")
def test_dataloader_grib_missing_dependency(
    mock_find_spec: MagicMock, mock_open: MagicMock
) -> None:
    """Test that DataLoader raises ValueError if cfgrib is not
    installed for GRIB files.
    """
    mock_find_spec.return_value = None  # Simulate cfgrib not found
    loader = DataLoader("test.grib")
    with pytest.raises(
        ValueError,
        match=r"cfgrib is required to open GRIB files. Please install it",
    ):
        loader.ensure_open()
    mock_open.assert_not_called()
