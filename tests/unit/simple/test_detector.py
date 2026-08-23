from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.io.data_loader import DataLoader
from pystormtracker.simple.detector import (
    SimpleDetector,
    _compute_masked_laplacian,
    _extract_centers,
    _filter_extrema,
    _remove_duplicate_extrema,
)


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
        Path("test.nc"), engine="h5netcdf", decode_times=False
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
        search_window_size=5, intensity_threshold=0.0, feature_refinement="grid"
    )[0]
    refined = detector.detect(
        search_window_size=5, intensity_threshold=0.0, feature_refinement="quadratic"
    )[0]

    assert raw[1][0] == 3.0
    assert raw[2][0] == 3.0
    assert refined[1][0] == pytest.approx(3.25)
    assert refined[2][0] == pytest.approx(3.2)
    assert refined[3][0] < raw[3][0]


def test_filter_extrema() -> None:
    # Create a 10x10 data with a clear minimum
    data: NDArray[np.float64] = np.ones((10, 10), dtype=np.float64) * 100.0
    data[5, 5] = 90.0

    # Test for local minimum with size 3, threshold 5
    out = _filter_extrema(data, size=3, threshold=5.0, is_min=True)
    assert out[5, 5] == 1.0
    assert np.sum(out) == 1.0

    # Test for local maximum (should be empty as data[5,5] is a minimum)
    out_max = _filter_extrema(data, size=3, threshold=5.0, is_min=False)
    assert np.sum(out_max) == 0.0


def test_filter_extrema_plateau() -> None:
    # Plateaus should be handled (rank filtering)
    data: NDArray[np.float64] = np.ones((10, 10), dtype=np.float64) * 100.0
    data[5, 5] = 90.0
    data[5, 6] = 90.0  # Plateau

    out = _filter_extrema(data, size=3, threshold=5.0, is_min=True)
    assert np.sum(out) == 2.0
    assert out[5, 5] == 1.0
    assert out[5, 6] == 1.0


def test_filter_extrema_does_not_wrap_projected_x() -> None:
    data = np.full((7, 7), 100.0, dtype=np.float64)
    data[3, 0] = 90.0

    global_result = _filter_extrema(
        data, size=3, threshold=5.0, is_min=True, periodic_x=True
    )
    projected_result = _filter_extrema(
        data, size=3, threshold=5.0, is_min=True, periodic_x=False
    )

    assert global_result[3, 0] == 1.0
    assert projected_result.sum() == 0.0


def test_compute_masked_laplacian() -> None:
    data: NDArray[np.float64] = np.zeros((5, 5), dtype=np.float64)
    data[2, 2] = -1.0  # Minimum

    mask: NDArray[np.float64] = np.zeros((5, 5), dtype=np.float64)
    mask[2, 2] = 1.0

    # Laplace: up + down + left + right - 4*center
    # neighbors are 0, center is -1 -> 0 + 0 + 0 + 0 - 4*(-1) = 4
    out = _compute_masked_laplacian(data, mask, is_min=True)
    assert out[2, 2] == 4.0
    assert np.sum(out) == 4.0


def test_remove_duplicate_extrema_tie_breaking() -> None:
    # Create two duplicate intensity points
    laplacian: NDArray[np.float64] = np.zeros((10, 10), dtype=np.float64)
    laplacian[5, 5] = 10.0
    laplacian[5, 6] = 10.0

    # Lower index wins: (5,5) should win over (5,6)
    out = _remove_duplicate_extrema(laplacian, size=3)
    assert out[5, 5] == 1.0
    assert out[5, 6] == 0.0
    assert np.sum(out) == 1.0


def test_extract_centers() -> None:
    extrema: NDArray[np.float64] = np.zeros((10, 10), dtype=np.float64)
    extrema[2, 2] = 1.0
    extrema[8, 8] = 1.0

    frame: NDArray[np.float64] = np.random.default_rng().random((10, 10))

    r, c, vals = _extract_centers(extrema, frame)
    assert len(r) == 2
    assert r[0] == 2
    assert c[0] == 2
    assert r[1] == 8
    assert c[1] == 8
    assert vals[0] == frame[2, 2]
