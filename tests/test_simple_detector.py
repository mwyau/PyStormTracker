from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from pystormtracker.simple.detector import SimpleDetector


@patch("xarray.open_dataset")
def test_simple_detector_init(mock_open: MagicMock) -> None:
    mock_ds = MagicMock(spec=xr.Dataset)
    mock_open.return_value = mock_ds

    mock_data = MagicMock(spec=xr.DataArray)
    mock_ds.__getitem__.return_value = mock_data
    mock_ds.coords = ["time", "latitude", "longitude"]
    mock_ds.variables = {"slp": mock_data}

    detector = SimpleDetector(pathname="test.nc", varname="slp")
    detector._ensure_open()

    mock_open.assert_called_once_with(Path("test.nc"), mask_and_scale=False)


@patch("xarray.open_dataset")
def test_simple_detector_detect_mock(mock_open: MagicMock) -> None:
    # Create real xarray data for reliable behavior
    data = np.ones((1, 7, 7)) * 1000
    data[0, 3, 3] = 950  # Minimum at index 3,3

    times = np.array(["2025-12-01"], dtype="datetime64[ns]")
    lats = np.arange(7, dtype=float)
    lons = np.arange(7, dtype=float)

    ds = xr.Dataset(
        data_vars={"slp": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    mock_open.return_value = ds

    detector = SimpleDetector(pathname="test.nc", varname="slp")
    centers = detector.detect(size=5, threshold=0.0)

    assert len(centers) == 1
    assert len(centers[0]) == 1
    assert centers[0][0].lat == 3.0
    assert centers[0][0].lon == 3.0
    assert centers[0][0].var == 950.0
