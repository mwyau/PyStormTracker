from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.hodges.detector import HodgesDetector
from pystormtracker.io.loader import DataLoader


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


@patch("xarray.open_dataset")
def test_hodges_detector_init(mock_open: MagicMock) -> None:
    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), np.ones((1, 3, 3)))},
        coords={"time": [0], "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )
    mock_open.return_value = ds

    detector = HodgesDetector(pathname="test.nc", varname="msl")
    detector._ensure_open()

    mock_open.assert_called_once_with(Path("test.nc"), engine="h5netcdf", chunks={})


@patch("xarray.open_dataset")
def test_hodges_detector_detect_mock(mock_open: MagicMock) -> None:
    # Create real xarray data for reliable behavior
    data: NDArray[np.float64] = np.ones((1, 7, 7)) * 1000
    # Create a nice quadratic peak for sub-grid refinement
    # f(y,x) = 1000 - (y-3)^2 - (x-3.2)^2
    # Grid peak will be at (3,3), refined should be at (3, 3.2)
    for i in range(7):
        for j in range(7):
            data[0, i, j] = 1000 - (i - 3) ** 2 - (j - 3.2) ** 2

    times: NDArray[np.datetime64] = np.array(["2025-12-01"], dtype="datetime64[ns]")
    lats: NDArray[np.float64] = np.arange(7, dtype=float)
    lons: NDArray[np.float64] = np.arange(7, dtype=float)

    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    mock_open.return_value = ds

    detector = HodgesDetector(pathname="test2.nc", varname="msl")
    # We want max because we built a peak
    raw_results = detector.detect(size=5, threshold=0.0, minmaxmode="max")

    assert len(raw_results) == 1
    _time_val, lats_out, lons_out, vars_dict = raw_results[0]

    assert len(lats_out) == 1
    assert lats_out[0] == 3.0
    # Refined lon should be close to 3.2
    assert np.allclose(lons_out[0], 3.2)
    assert vars_dict["msl"][0] == 1000.0
