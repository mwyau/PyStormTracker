from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import utils
from utils import (
    DATA_RELEASE_VERSION,
    RAW_CONTENT_URL,
    fetch_era5_msl,
    fetch_era5_uv850,
    fetch_era5_vo850,
    get_base_dir,
    get_cached_data,
    list_release_files,
    parse_sha256sums,
)


def _configure_asset(cache: MagicMock, filename: str) -> None:
    cache.registry = {filename: "sha256:0123456789abcdef"}


def test_get_base_dir() -> None:
    """Test get_base_dir returns a path pointing to PyStormTracker root."""
    base_dir = get_base_dir()
    assert base_dir.is_dir()
    assert (base_dir / "pyproject.toml").exists()
    assert (base_dir / "tests").is_dir()


def test_parse_sha256sums(tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS"
    msl_checksum = "a" * 64
    vo_checksum = "B" * 64
    manifest.write_text(
        "# Release data\n"
        f"{msl_checksum}  era5_msl_2025-2026_djf_n320.grib\n"
        f"{vo_checksum} era5_vo850_2025-2026_djf_n320.grib\n",
        encoding="utf-8",
    )

    registry = parse_sha256sums(manifest)

    assert registry == {
        "era5_msl_2025-2026_djf_n320.grib": f"sha256:{msl_checksum}",
        "era5_vo850_2025-2026_djf_n320.grib": f"sha256:{vo_checksum.lower()}",
    }


@pytest.mark.parametrize(
    ("manifest_text", "match"),
    [
        ("not-a-valid-entry\n", "Invalid SHA256SUMS entry"),
        (
            f"{'z' * 64} era5_msl_2025-2026_djf_n320.grib\n",
            "Invalid SHA-256 checksum",
        ),
        (
            f"{'a' * 64} nested/era5_msl_2025-2026_djf_n320.grib\n",
            "Invalid release filename",
        ),
        ("# no assets\n", "does not contain any data assets"),
    ],
)
def test_parse_sha256sums_rejects_invalid_entries(
    tmp_path: Path, manifest_text: str, match: str
) -> None:
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        parse_sha256sums(manifest)


@patch("utils.pooch.create")
@patch("utils.pooch.retrieve")
def test_get_cached_data_loads_release_manifest(
    mock_retrieve: MagicMock,
    mock_create: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "SHA256SUMS"
    checksum = "a" * 64
    manifest.write_text(
        f"{checksum} era5_msl_2025-2026_djf_n320.grib\n", encoding="utf-8"
    )
    cache = MagicMock()
    mock_retrieve.return_value = str(manifest)
    mock_create.return_value = cache
    monkeypatch.setattr(utils, "CACHED_DATA", None)

    assert get_cached_data() is cache

    retrieve_kwargs = mock_retrieve.call_args.kwargs
    assert retrieve_kwargs["url"] == utils.SHA256SUMS_URL
    assert retrieve_kwargs["known_hash"] is None
    assert retrieve_kwargs["fname"] == utils.SHA256SUMS_FILENAME
    assert retrieve_kwargs["path"]
    assert mock_create.call_args.kwargs["registry"] == {
        "era5_msl_2025-2026_djf_n320.grib": f"sha256:{checksum}"
    }


@patch("utils.get_cached_data")
def test_list_release_files(mock_get_cached_data: MagicMock) -> None:
    mock_get_cached_data.return_value.registry = {
        "z.nc": "sha256:1",
        "a.grib": "sha256:2",
    }

    assert list_release_files() == ("a.grib", "z.nc")


@patch("utils.get_cached_data")
def test_fetch_era5_msl_valid(mock_get_cached_data: MagicMock) -> None:
    filename = "era5_msl_2025-2026_djf_2.5x2.5.nc"
    cache = mock_get_cached_data.return_value
    _configure_asset(cache, filename)
    cache.fetch.return_value = "/path/to/data.nc"

    path = fetch_era5_msl(resolution="2.5x2.5", season="djf", format="nc")

    assert path == "/path/to/data.nc"
    cache.fetch.assert_called_once_with(filename)


@patch("utils.get_cached_data")
def test_fetch_era5_msl_n320_grib(mock_get_cached_data: MagicMock) -> None:
    filename = "era5_msl_2025-2026_djf_n320.grib"
    cache = mock_get_cached_data.return_value
    _configure_asset(cache, filename)
    cache.fetch.return_value = "/path/to/data.grib"

    path = fetch_era5_msl(resolution="n320", format="grib")

    assert path == "/path/to/data.grib"
    cache.fetch.assert_called_once_with(filename)


def test_fetch_era5_zarr_remote() -> None:
    url = fetch_era5_msl(resolution="2.5x2.5", format="zarr")

    assert url == f"{RAW_CONTENT_URL}era5_msl_2025-2026_djf_2.5x2.5.zarr"


@patch("utils.get_cached_data")
def test_fetch_era5_zarr_local(mock_get_cached_data: MagicMock) -> None:
    archive_filename = "era5_msl_2025-2026_djf_2.5x2.5.zarr.tar.gz"
    cache = mock_get_cached_data.return_value
    _configure_asset(cache, archive_filename)
    cache.fetch.return_value = [
        "/cache/archive.untar/era5_msl.zarr/zarr.json",
        "/cache/archive.untar/era5_msl.zarr/msl/c/0/0",
    ]

    path = fetch_era5_msl(resolution="2.5x2.5", format="zarr", local=True)

    assert path == "/cache/archive.untar/era5_msl.zarr"
    call_args = cache.fetch.call_args
    assert call_args.args == (archive_filename,)
    assert call_args.kwargs["processor"].__class__.__name__ == "Untar"


@patch("utils.get_cached_data")
def test_fetch_era5_vo850_valid(mock_get_cached_data: MagicMock) -> None:
    filename = "era5_vo850_2025-2026_djf_2.5x2.5.nc"
    cache = mock_get_cached_data.return_value
    _configure_asset(cache, filename)
    cache.fetch.return_value = "/path/to/vo850.nc"

    path = fetch_era5_vo850(resolution="2.5x2.5", season="djf", format="nc")

    assert path == "/path/to/vo850.nc"
    cache.fetch.assert_called_once_with(filename)


def test_fetch_era5_uv850_zarr_remote() -> None:
    url = fetch_era5_uv850(resolution="2.5x2.5", format="zarr")

    assert url == f"{RAW_CONTENT_URL}era5_uv850_2025-2026_djf_2.5x2.5.zarr"


@patch("utils.get_cached_data")
def test_fetch_era5_invalid_resolution(mock_get_cached_data: MagicMock) -> None:
    mock_get_cached_data.return_value.registry = {}

    with pytest.raises(ValueError, match="not available in release"):
        fetch_era5_msl(resolution="invalid")


def test_fetch_era5_invalid_season() -> None:
    with pytest.raises(ValueError, match="Season 'mam' not available"):
        fetch_era5_msl(season="mam")


def test_fetch_era5_invalid_format() -> None:
    with pytest.raises(ValueError, match="Format must be"):
        fetch_era5_msl(format="txt")


def test_fetch_era5_local_requires_zarr() -> None:
    with pytest.raises(ValueError, match="only supported when format='zarr'"):
        fetch_era5_msl(local=True)


def test_data_release_version() -> None:
    assert DATA_RELEASE_VERSION == "v0.1.4-data"
