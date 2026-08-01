from __future__ import annotations

from pathlib import Path
from typing import cast

import pooch  # type: ignore[import-untyped]

DATA_RELEASE_VERSION = "v0.1.4-data"
RELEASE_URL = f"https://github.com/mwyau/PyStormTracker-Data/releases/download/{DATA_RELEASE_VERSION}/"
RAW_CONTENT_URL = f"https://raw.githubusercontent.com/mwyau/PyStormTracker-Data/{DATA_RELEASE_VERSION}/"
SHA256SUMS_URL = f"{RELEASE_URL}SHA256SUMS"
SHA256SUMS_FILENAME = f"{DATA_RELEASE_VERSION}-SHA256SUMS"
SHA256SUMS_HASH = (
    "sha256:4f221867b111ec5411c58859da825b13111f9aab8a492a50172cad45fddb3ad9"
)


def get_base_dir() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.absolute()


CACHED_DATA: pooch.Pooch | None = None


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse a release checksum manifest into a Pooch registry."""
    registry: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid SHA256SUMS entry on line {line_number}: {raw_line!r}"
            )

        checksum, filename = parts
        is_sha256 = len(checksum) == 64 and all(
            char in "0123456789abcdef" for char in checksum.lower()
        )
        if not is_sha256:
            raise ValueError(
                f"Invalid SHA-256 checksum on line {line_number}: {checksum!r}"
            )
        if Path(filename).name != filename:
            raise ValueError(
                f"Invalid release filename on line {line_number}: {filename!r}"
            )
        registry[filename] = f"sha256:{checksum.lower()}"

    if not registry:
        raise ValueError("SHA256SUMS does not contain any data assets")
    return registry


def get_cached_data() -> pooch.Pooch:
    """Return the cache backed by the current release checksum manifest."""
    global CACHED_DATA

    if CACHED_DATA is None:
        manifest_path = Path(
            pooch.retrieve(
                url=SHA256SUMS_URL,
                known_hash=SHA256SUMS_HASH,
                fname=SHA256SUMS_FILENAME,
                path=pooch.os_cache("pystormtracker"),
            )
        )
        CACHED_DATA = pooch.create(
            path=pooch.os_cache("pystormtracker"),
            base_url=RELEASE_URL,
            registry=parse_sha256sums(manifest_path),
        )
    return CACHED_DATA


def list_release_files() -> tuple[str, ...]:
    """Return the data asset filenames published by the current release."""
    return tuple(sorted(get_cached_data().registry))


def fetch_release_file(filename: str) -> str:
    """Fetch a checksum-verified data asset from the current release."""
    return str(get_cached_data().fetch(filename))


def _release_filename(
    variable: str,
    resolution: str,
    season: str,
    format: str,
) -> str:
    suffix = ".zarr.tar.gz" if format == "zarr" else f".{format}"
    return f"era5_{variable}_2025-2026_{season}_{resolution}{suffix}"


def _validate_release_asset(filename: str) -> None:
    if filename not in get_cached_data().registry:
        raise ValueError(
            f"Data asset '{filename}' is not available in release "
            f"{DATA_RELEASE_VERSION}"
        )


def _fetch_local_zarr(filename: str) -> str:
    extracted_files = cast(
        list[str],
        get_cached_data().fetch(filename, processor=pooch.Untar()),
    )
    stores = {
        parent
        for extracted_file in extracted_files
        for parent in Path(extracted_file).parents
        if parent.name.endswith(".zarr")
    }
    if len(stores) != 1:
        raise ValueError(
            f"Expected one Zarr store in archive '{filename}', found {len(stores)}"
        )
    return str(next(iter(stores)))


def _fetch_era5(
    variable: str,
    resolution: str,
    season: str,
    format: str,
    local: bool,
) -> str:
    if season != "djf":
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    if format not in ("nc", "grib", "zarr"):
        raise ValueError("Format must be 'nc', 'grib', or 'zarr'")
    if local and format != "zarr":
        raise ValueError("local=True is only supported when format='zarr'")

    if format == "zarr":
        filename = _release_filename(variable, resolution, season, format)
        _validate_release_asset(filename)
        if not local:
            return RAW_CONTENT_URL + filename.removesuffix(".tar.gz")
        return _fetch_local_zarr(filename)

    filename = _release_filename(variable, resolution, season, format)
    _validate_release_asset(filename)
    return fetch_release_file(filename)


def fetch_era5_msl(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """
    Fetches the ERA5 mean sea level pressure sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution published by the data release.
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc" (default), "grib", or "zarr".
        local (bool): Extract a local Zarr store when ``format="zarr"``.

    Returns:
        str: Absolute path to the downloaded local file, local Zarr store, or URL.
    """
    return _fetch_era5("msl", resolution, season, format, local)


def fetch_era5_vo850(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """
    Fetches the ERA5 850hPa relative vorticity sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution published by the data release.
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc" (default), "grib", or "zarr".
        local (bool): Extract a local Zarr store when ``format="zarr"``.

    Returns:
        str: Absolute path to the downloaded local file, local Zarr store, or URL.
    """
    return _fetch_era5("vo850", resolution, season, format, local)


def fetch_era5_uv850(
    resolution: str = "2.5x2.5",
    season: str = "djf",
    format: str = "nc",
    local: bool = False,
) -> str:
    """
    Fetches the ERA5 850hPa u- and v-component of wind sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution published by the data release.
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc", "grib", or "zarr".
        local (bool): Extract a local Zarr store when ``format="zarr"``.

    Returns:
        str: Absolute path to the downloaded local file, local Zarr store, or URL.
    """
    return _fetch_era5("uv850", resolution, season, format, local)


# --- Local Integration Test Data Helpers ---

BASE_DIR = get_base_dir()
ERA5_TEST_DIR = BASE_DIR / "tests" / "data" / "era5"
TRACKS_TEST_DIR = BASE_DIR / "tests" / "data" / "tracks"


def get_era5_msl_path(res: str = "2.5x2.5", suffix: str = "") -> Path:
    """
    Returns the path to the ERA5 MSL test data.

    Args:
        res: Resolution (e.g., '2.5x2.5' or '0.25x0.25').
        suffix: Optional suffix for filtered data (e.g., 't5-42_ncl').
    """
    name = f"era5_msl_2025120100_{res}"
    if suffix:
        name += f"_{suffix}"
    return ERA5_TEST_DIR / f"{name}.nc"


def get_era5_uv_path(res: str = "2.5x2.5") -> Path:
    """Returns the path to the ERA5 UV test data."""
    return ERA5_TEST_DIR / f"era5_uv850_2025120100_{res}.nc"


def get_era5_vodv_path(res: str = "2.5x2.5", suffix: str = "ncl") -> Path:
    """Returns the path to the ERA5 VODV test data."""
    return ERA5_TEST_DIR / f"era5_vodv850_2025120100_{res}_{suffix}.nc"


def get_legacy_track_path(var: str = "msl") -> Path:
    """Returns the path to legacy regression track files."""
    if var == "msl":
        return TRACKS_TEST_DIR / "era5_msl_2025-2026_djf_2.5x2.5_v0.0.2_imilast.txt"
    if var == "vo":
        return TRACKS_TEST_DIR / "era5_vo_2025-2026_djf_2.5x2.5_1e-4_v0.0.2_imilast.txt"
    raise ValueError(f"Unknown legacy variable: {var}")
