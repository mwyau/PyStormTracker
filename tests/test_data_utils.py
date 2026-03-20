from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pystormtracker.utils.data_utils import fetch_era5_msl, fetch_era5_vo850


@patch("pystormtracker.utils.data_utils.GOOD_DATA")
def test_fetch_era5_msl_valid(mock_pooch):
    mock_pooch.fetch.return_value = "/path/to/data.nc"

    path = fetch_era5_msl(resolution="2.5x2.5", season="djf", format="nc")

    assert path == "/path/to/data.nc"
    mock_pooch.fetch.assert_called_once_with("era5_msl_2025-2026_djf_2.5x2.5.nc")


@patch("pystormtracker.utils.data_utils.GOOD_DATA")
def test_fetch_era5_msl_grib(mock_pooch):
    mock_pooch.fetch.return_value = "/path/to/data.grib"

    path = fetch_era5_msl(resolution="0.25x0.25", season="djf", format="grib")

    assert path == "/path/to/data.grib"
    mock_pooch.fetch.assert_called_once_with("era5_msl_2025-2026_djf_0.25x0.25.grib")


def test_fetch_era5_msl_invalid_res():
    with pytest.raises(ValueError, match="Resolution must be either"):
        fetch_era5_msl(resolution="invalid")


def test_fetch_era5_msl_invalid_season():
    with pytest.raises(ValueError, match="Season 'mam' not available"):
        fetch_era5_msl(season="mam")


def test_fetch_era5_msl_invalid_format():
    with pytest.raises(ValueError, match="Format must be either"):
        fetch_era5_msl(format="txt")


@patch("pystormtracker.utils.data_utils.GOOD_DATA")
def test_fetch_era5_vo850_valid(mock_pooch):
    mock_pooch.fetch.return_value = "/path/to/vo850.nc"

    path = fetch_era5_vo850(resolution="2.5x2.5", season="djf", format="nc")

    assert path == "/path/to/vo850.nc"
    mock_pooch.fetch.assert_called_once_with("era5_vo850_2025-2026_djf_2.5x2.5.nc")
