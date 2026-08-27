from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final

import pooch

DATA_VERSION: Final[str] = "v0.2.0-data"
RAW_BASE: Final[str] = (
    f"https://raw.githubusercontent.com/mwyau/PyStormTracker-Data/{DATA_VERSION}/"
)
RELEASE_BASE: Final[str] = (
    f"https://github.com/mwyau/PyStormTracker-Data/releases/download/{DATA_VERSION}/"
)
DECEMBER_2025_START: Final[str] = "2025-12-01T00:00:00"
DECEMBER_2025_END: Final[str] = "2025-12-31T18:00:00"


def get_base_dir() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.absolute()


def _repo_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or "\\" in path
        or ".." in candidate.parts
        or candidate == PurePosixPath(".")
    ):
        raise ValueError(f"repository path must be relative and normalized: {path!r}")
    return candidate.as_posix()


def _release_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename:
        raise ValueError(f"release asset must be a filename, got {filename!r}")
    return filename


def _fetch(url: str, relative_name: str) -> str:
    relative = Path(relative_name)
    cache_dir = (
        Path(pooch.os_cache("pystormtracker")) / "data" / DATA_VERSION / relative.parent
    )
    return str(
        pooch.retrieve(
            url=url,
            fname=relative.name,
            path=cache_dir,
        )
    )


def raw_repo_url(path: str) -> str:
    """Return the pinned raw-Git URL for a repository path."""
    return RAW_BASE + _repo_path(path)


def fetch_repo_file(path: str) -> str:
    """Fetch one small Git-tracked Data-repository file into the Pooch cache."""
    relative = _repo_path(path)
    return _fetch(raw_repo_url(relative), relative)


def fetch_release_asset(filename: str) -> str:
    """Fetch one large release asset by its exact filename."""
    filename = _release_filename(filename)
    return _fetch(RELEASE_BASE + filename, filename)


def f320_month_filenames(
    variable: str = "msl",
    *,
    year: int = 2024,
) -> tuple[str, ...]:
    """Return the twelve canonical monthly F320 filenames."""
    if variable not in {"msl", "vo850"}:
        raise ValueError(f"unknown F320 variable: {variable}")
    return tuple(
        f"era5_{variable}_{year}-{month:02d}_f320.nc" for month in range(1, 13)
    )


def fetch_f320_month(variable: str, month: int, *, year: int = 2024) -> str:
    """Fetch one canonical monthly F320 release asset."""
    if variable not in {"msl", "vo850"}:
        raise ValueError(f"unknown F320 variable: {variable}")
    if month not in range(1, 13):
        raise ValueError(f"month must be in 1..12, got {month}")
    return fetch_release_asset(f"era5_{variable}_{year}-{month:02d}_f320.nc")


def _fetch_era5(
    variable: str,
    resolution: str,
    season: str,
    format: str,
    local: bool,
) -> str:
    if season != "djf":
        raise ValueError(f"Season {season!r} not available. Options: 'djf'")
    if format not in {"nc", "grib", "zarr"}:
        raise ValueError("Format must be 'nc', 'grib', or 'zarr'")
    if resolution not in {"0.25x0.25", "2.5x2.5", "n320"}:
        raise ValueError(f"Resolution {resolution!r} is not available")

    if format == "zarr":
        if local:
            raise ValueError(
                "local Zarr extraction is not supported; use the pinned raw URL"
            )
        if resolution != "2.5x2.5" or variable not in {"msl", "vo850"}:
            raise ValueError(
                f"No Git-tracked Zarr store is available for {variable}/{resolution}"
            )
        return raw_repo_url(f"integration/era5_{variable}_2025-2026_djf_2.5x2.5.zarr")

    return fetch_release_asset(f"era5_{variable}_2025-2026_djf_{resolution}.{format}")


def fetch_era5_msl(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """Fetch a release-backed ERA5 mean-sea-level-pressure asset."""
    return _fetch_era5("msl", resolution, season, format, local)


def fetch_era5_vo850(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """Fetch a release-backed ERA5 850 hPa vorticity asset."""
    return _fetch_era5("vo850", resolution, season, format, local)


def fetch_era5_uv850(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """Fetch a release-backed ERA5 850 hPa wind asset."""
    return _fetch_era5("uv850", resolution, season, format, local)


# --- Local integration test data helpers ---

BASE_DIR = get_base_dir()
ERA5_TEST_DATA_DIR = BASE_DIR / "tests" / "data" / "era5"
TRACKS_TEST_DATA_DIR = BASE_DIR / "tests" / "data" / "tracks"

INTEGRATION_MSL_FILENAME: Final[str] = "era5_msl_2025-12_2.5x2.5.nc"


def get_integration_msl_path() -> Path:
    """Return the one committed real-data integration input."""
    return ERA5_TEST_DATA_DIR / INTEGRATION_MSL_FILENAME


def get_legacy_track_path(var: str = "msl") -> Path:
    """Fetch and return a historical PyStormTracker v0.0.2 reference path."""
    if var == "msl":
        path = "parity/legacy/v0.0.2/era5_msl_2025-2026_djf_2.5x2.5_imilast.txt"
    elif var == "vo":
        path = "parity/legacy/v0.0.2/era5_vo850_2025-2026_djf_2.5x2.5_1e-4_imilast.txt"
    else:
        raise ValueError(f"Unknown legacy variable: {var}")
    return Path(fetch_repo_file(path))
