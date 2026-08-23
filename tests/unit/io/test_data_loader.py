from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from pystormtracker.io.data_loader import DataLoader, normalize_tracking_data


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
    mock_open.assert_called_once_with(
        Path("test.nc"), engine="h5netcdf", decode_times=False
    )


@patch("xarray.open_dataset")
def test_ensure_open_grib(mock_open: MagicMock) -> None:
    pytest.importorskip("cfgrib")

    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader = DataLoader("test.grib")
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(
        Path("test.grib"), engine="cfgrib", decode_times=False
    )


@patch("xarray.open_dataset")
def test_ensure_open_zarr(mock_open: MagicMock) -> None:
    pytest.importorskip("zarr")

    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    loader = DataLoader("test.zarr")
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(
        Path("test.zarr"),
        engine="zarr",
        decode_times=False,
        backend_kwargs={"consolidated": False},
    )


@patch("xarray.open_dataset")
def test_ensure_open_zarr_dir(mock_open: MagicMock, tmp_path: Path) -> None:
    pytest.importorskip("zarr")

    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds
    zarr_dir = tmp_path / "test_data"
    zarr_dir.mkdir()
    (zarr_dir / ".zmetadata").touch()
    loader = DataLoader(zarr_dir)
    ds = loader.ensure_open()
    assert ds == mock_ds
    mock_open.assert_called_once_with(
        zarr_dir,
        engine="zarr",
        decode_times=False,
        backend_kwargs={"consolidated": False},
    )


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


def test_normalize_tracking_data_selects_dataarray_and_dataset_inputs() -> None:
    times = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2000-01-02"),
            np.datetime64("2000-01-03"),
        ]
    )
    data = xr.DataArray(
        np.arange(3.0)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
        name="msl",
    )

    selected_array = normalize_tracking_data(
        data,
        "msl",
        start_time="2000-01-02",
        end_time="2000-01-03",
    )
    selected_dataset = normalize_tracking_data(
        data.to_dataset(),
        "msl",
        start_time="2000-01-02",
        end_time="2000-01-03",
    )
    selected_processed = normalize_tracking_data(
        data.rename("msl_spectral_filtered"),
        "msl",
    )

    assert selected_array.name == "msl"
    assert selected_dataset.name == "msl"
    np.testing.assert_array_equal(selected_array.time, selected_dataset.time)
    np.testing.assert_array_equal(selected_array.values, np.array([[[1.0]], [[2.0]]]))
    assert selected_processed.name == "msl_spectral_filtered"


@pytest.mark.parametrize(
    ("lon_name", "longitudes", "expected"),
    [
        ("lon", np.arange(0.0, 360.0, 60.0), True),
        ("longitude", np.arange(300.0, -60.0, -60.0), True),
        ("lon", np.arange(-180.0, 180.0, 60.0), True),
        ("lon", np.arange(0.0, 180.0, 30.0), False),
        ("x", np.arange(-3000.0, 3001.0, 1000.0), False),
    ],
)
def test_is_global_longitude(
    lon_name: str, longitudes: np.ndarray, expected: bool
) -> None:
    data = xr.DataArray(
        np.zeros((1, 3, len(longitudes))),
        dims=("time", "lat", lon_name),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": [-1.0, 0.0, 1.0],
            lon_name: longitudes,
        },
        name="msl",
    )

    assert DataLoader(data).is_global_longitude() is expected


@patch("xarray.open_dataset")
@patch("pystormtracker.io.data_loader.find_spec")
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


def test_reduced_gaussian_metadata(reduced_gaussian_data: xr.DataArray) -> None:
    loader = DataLoader(reduced_gaussian_data)

    assert loader.is_reduced_gaussian("msl")
    np.testing.assert_array_equal(
        loader.get_reduced_grid_pl("msl"), [4, 8, 12, 16, 16, 12, 8, 4]
    )
    metadata = loader.get_grid_metadata("msl")
    assert set(metadata) == {"theta", "nphi", "phi0", "ringstart"}
    np.testing.assert_array_equal(metadata["ringstart"], [0, 4, 12, 24, 40, 56, 68, 76])
