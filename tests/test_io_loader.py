from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from pystormtracker.io.loader import DataLoader


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
    mock_open.assert_called_once_with(Path("test.nc"), engine="h5netcdf", chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_grib(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds

    loader = DataLoader("test.grib")
    ds = loader.ensure_open()

    assert ds == mock_ds
    mock_open.assert_called_once_with(Path("test.grib"), engine="cfgrib", chunks={})


@patch("xarray.open_dataset")
def test_ensure_open_caching(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds

    loader1 = DataLoader("test.nc")
    loader2 = DataLoader("test.nc")

    loader1.ensure_open()
    loader2.ensure_open()

    # Should only call open_dataset once for the same path
    mock_open.assert_called_once()


@patch("xarray.open_dataset")
def test_get_coords_mapping(mock_open: MagicMock) -> None:
    # Mock dataset with specific coordinate names
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.coords = ["time", "lat", "lon"]
    mock_open.return_value = mock_ds

    loader = DataLoader("test.nc")
    time, lat, lon = loader.get_coords()

    assert time == "time"
    assert lat == "lat"
    assert lon == "lon"


@patch("xarray.open_dataset")
def test_get_coords_mapping_aliases(mock_open: MagicMock) -> None:
    # Mock dataset with alias coordinate names
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.coords = ["valid_time", "latitude", "longitude"]
    mock_open.return_value = mock_ds

    loader = DataLoader("test.nc")
    time, lat, lon = loader.get_coords()

    assert time == "valid_time"
    assert lat == "latitude"
    assert lon == "longitude"


@patch("xarray.open_dataset")
def test_get_coords_mapping_defaults(mock_open: MagicMock) -> None:
    # Mock dataset with no matching coordinate names
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_ds.coords = []
    mock_open.return_value = mock_ds

    loader = DataLoader("test.nc")
    time, lat, lon = loader.get_coords()

    assert time == "time"
    assert lat == "latitude"
    assert lon == "longitude"
