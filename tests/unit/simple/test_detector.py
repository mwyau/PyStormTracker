from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.simple.detector import SimpleDetector


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


@patch("xarray.open_dataset")
def test_simple_detector_init(mock_open: MagicMock) -> None:
    # Use a real dataset to avoid mock complexity
    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), np.ones((1, 3, 3)))},
        coords={"time": [0], "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )
    mock_open.return_value = ds

    detector = SimpleDetector(pathname="test.nc", variable_name="msl")
    detector._ensure_open()

    mock_open.assert_called_once_with(
        Path("test.nc"), engine=None, chunks={}, decode_times=False
    )


@patch("xarray.open_dataset")
def test_simple_detector_detect_mock(mock_open: MagicMock) -> None:
    # Create real xarray data for reliable behavior
    data: NDArray[np.float64] = np.ones((1, 7, 7)) * 1000
    data[0, 3, 3] = 950  # Minimum at index 3,3

    times: NDArray[np.datetime64] = np.array(["2025-12-01"], dtype="datetime64[ns]")
    lats: NDArray[np.float64] = np.arange(7, dtype=float)
    lons: NDArray[np.float64] = np.arange(7, dtype=float)

    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    mock_open.return_value = ds

    detector = SimpleDetector(pathname="test2.nc", variable_name="msl")
    raw_results = detector.detect(search_window_size=5, intensity_threshold=0.0)

    assert len(raw_results) == 1
    _time_val, lats_out, lons_out, values = raw_results[0]

    assert len(lats_out) == 1
    assert lats_out[0] == 3.0
    assert lons_out[0] == 3.0
    assert values[0] == 950.0


def test_simple_detector_optional_quadratic_refinement() -> None:
    y, x = np.meshgrid(np.arange(7.0), np.arange(7.0), indexing="ij")
    field = (y - 3.25) ** 2 + (x - 3.2) ** 2
    data = xr.DataArray(
        field[np.newaxis, :, :],
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": np.arange(7.0),
            "lon": np.arange(7.0),
        },
        name="msl",
    )

    detector = SimpleDetector.from_xarray(data)
    raw = detector.detect(
        search_window_size=5, intensity_threshold=0.0, feature_point_method="grid"
    )[0]
    refined = detector.detect(
        search_window_size=5, intensity_threshold=0.0, feature_point_method="quadratic"
    )[0]

    assert raw[1][0] == 3.0
    assert raw[2][0] == 3.0
    assert refined[1][0] == pytest.approx(3.25)
    assert refined[2][0] == pytest.approx(3.2)
    assert refined[3][0] < raw[3][0]
