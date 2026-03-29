from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from testing_utils import (
    RAW_CONTENT_URL,
    fetch_era5_msl,
    fetch_era5_uv850,
    fetch_era5_vo850,
    get_base_dir,
)


def test_get_base_dir() -> None:
    """Test get_base_dir returns a path pointing to PyStormTracker root."""
    base_dir = get_base_dir()
    assert base_dir.is_dir()
    assert (base_dir / "pyproject.toml").exists()
    assert (base_dir / "tests").is_dir()


@patch("testing_utils.CACHED_DATA")
def test_fetch_era5_msl_valid(mock_pooch: MagicMock) -> None:
    mock_pooch.fetch.return_value = "/path/to/data.nc"
    path = fetch_era5_msl(resolution="2.5x2.5", season="djf", format="nc")
    assert path == "/path/to/data.nc"
    mock_pooch.fetch.assert_called_once_with("era5_msl_2025-2026_djf_2.5x2.5.nc")


@patch("testing_utils.CACHED_DATA")
def test_fetch_era5_msl_grib(mock_pooch: MagicMock) -> None:
    mock_pooch.fetch.return_value = "/path/to/data.grib"
    path = fetch_era5_msl(resolution="0.25x0.25", season="djf", format="grib")
    assert path == "/path/to/data.grib"
    mock_pooch.fetch.assert_called_once_with("era5_msl_2025-2026_djf_0.25x0.25.grib")


def test_fetch_era5_zarr() -> None:
    """Test that fetching zarr format returns a URL."""
    url = fetch_era5_msl(resolution="2.5x2.5", format="zarr")
    expected_url = f"{RAW_CONTENT_URL}era5_msl_2025-2026_djf_2.5x2.5.zarr"
    assert url == expected_url


def test_fetch_era5_msl_invalid_res() -> None:
    with pytest.raises(ValueError, match="Resolution must be either"):
        fetch_era5_msl(resolution="invalid")


def test_fetch_era5_msl_invalid_season() -> None:
    with pytest.raises(ValueError, match="Season 'mam' not available"):
        fetch_era5_msl(season="mam")


def test_fetch_era5_msl_invalid_format() -> None:
    with pytest.raises(ValueError, match="Format must be"):
        fetch_era5_msl(format="txt")


@patch("testing_utils.CACHED_DATA")
def test_fetch_era5_vo850_valid(mock_pooch: MagicMock) -> None:
    mock_pooch.fetch.return_value = "/path/to/vo850.nc"
    path = fetch_era5_vo850(resolution="2.5x2.5", season="djf", format="nc")
    assert path == "/path/to/vo850.nc"
    mock_pooch.fetch.assert_called_once_with("era5_vo850_2025-2026_djf_2.5x2.5.nc")


def test_fetch_era5_uv850_valid() -> None:
    """Test that fetching uv850 zarr format returns a URL."""
    url = fetch_era5_uv850(resolution="2.5x2.5", season="djf", format="zarr")
    expected_url = f"{RAW_CONTENT_URL}era5_uv850_2025-2026_djf_2.5x2.5.zarr"
    assert url == expected_url
