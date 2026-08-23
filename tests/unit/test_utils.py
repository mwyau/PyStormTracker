from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests import utils
from tests.utils import (
    DATA_VERSION,
    RAW_BASE,
    RELEASE_BASE,
    f320_month_filenames,
    fetch_era5_msl,
    fetch_f320_month,
    fetch_release_asset,
    fetch_repo_file,
)


@patch("tests.utils.pooch.retrieve")
def test_fetch_repo_file_uses_the_pinned_raw_path(
    mock_retrieve: MagicMock,
    tmp_path: Path,
) -> None:
    with patch("tests.utils.pooch.os_cache", return_value=tmp_path):
        mock_retrieve.return_value = str(tmp_path / "reference.txt")
        path = fetch_repo_file("parity/ncl/reference.txt")

    assert path.endswith("reference.txt")
    mock_retrieve.assert_called_once_with(
        url=f"{RAW_BASE}parity/ncl/reference.txt",
        fname="reference.txt",
        path=tmp_path / "data" / DATA_VERSION / "parity" / "ncl",
    )


@patch("tests.utils.pooch.retrieve")
def test_fetch_release_asset_uses_the_pinned_release_path(
    mock_retrieve: MagicMock,
    tmp_path: Path,
) -> None:
    with patch("tests.utils.pooch.os_cache", return_value=tmp_path):
        mock_retrieve.return_value = str(tmp_path / "asset.nc")
        path = fetch_release_asset("era5_msl_2024-01_f320.nc")

    assert path.endswith("asset.nc")
    mock_retrieve.assert_called_once_with(
        url=f"{RELEASE_BASE}era5_msl_2024-01_f320.nc",
        fname="era5_msl_2024-01_f320.nc",
        path=tmp_path / "data" / DATA_VERSION / ".",
    )


@pytest.mark.parametrize(
    "path",
    ["", "/absolute/file", "../file", "parity/../file", r"parity\file"],
)
def test_fetch_repo_file_rejects_non_relative_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        fetch_repo_file(path)


def test_f320_month_names_are_explicit() -> None:
    assert f320_month_filenames() == tuple(
        f"era5_msl_2024-{month:02d}_f320.nc" for month in range(1, 13)
    )
    assert f320_month_filenames("vo850")[-1] == "era5_vo850_2024-12_f320.nc"


@patch("tests.utils.fetch_release_asset")
def test_fetch_f320_month_constructs_a_release_filename(
    mock_fetch: MagicMock,
) -> None:
    mock_fetch.return_value = "/cache/era5_msl_2024-01_f320.nc"

    assert fetch_f320_month("msl", 1) == "/cache/era5_msl_2024-01_f320.nc"
    mock_fetch.assert_called_once_with("era5_msl_2024-01_f320.nc")


@patch("tests.utils.fetch_release_asset")
def test_era5_release_wrapper_does_not_consult_a_catalog(
    mock_fetch: MagicMock,
) -> None:
    mock_fetch.return_value = "/cache/era5.nc"

    assert fetch_era5_msl(format="nc") == "/cache/era5.nc"
    mock_fetch.assert_called_once_with("era5_msl_2025-2026_djf_2.5x2.5.nc")


def test_zarr_uses_its_pinned_raw_store_path() -> None:
    url = utils.fetch_era5_msl(format="zarr")
    assert url == (f"{RAW_BASE}integration/era5_msl_2025-2026_djf_2.5x2.5.zarr")


def test_zarr_local_extraction_is_not_part_of_the_contract() -> None:
    with pytest.raises(ValueError, match="raw URL"):
        fetch_era5_msl(format="zarr", local=True)


def test_invalid_era5_format_and_resolution() -> None:
    with pytest.raises(ValueError, match="Format"):
        fetch_era5_msl(format="txt")
    with pytest.raises(ValueError, match="Resolution"):
        fetch_era5_msl(resolution="invalid")
