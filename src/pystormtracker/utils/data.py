from __future__ import annotations

import pooch  # type: ignore[import-untyped]

DATA_RELEASE_VERSION = "v0.1.3-data"
RELEASE_URL = f"https://github.com/mwyau/PyStormTracker-Data/releases/download/{DATA_RELEASE_VERSION}/"
RAW_CONTENT_URL = f"https://raw.githubusercontent.com/mwyau/PyStormTracker-Data/{DATA_RELEASE_VERSION}/"


# Define a central repository of data files
CACHED_DATA = pooch.create(
    path=pooch.os_cache("pystormtracker"),
    base_url=RELEASE_URL,
    registry={
        "era5_msl_2025-2026_djf_0.25x0.25.nc": (
            "sha256:a1847093356472303585eb9acdbfb8c993795a2e643e80d5f7cc803d0919216d"
        ),
        "era5_msl_2025-2026_djf_2.5x2.5.nc": (
            "sha256:19477e18e4239b9f8ea5a7b7a56c2f3790fbc661bbff1a949e59ebda1a61fc40"
        ),
        "era5_msl_2025-2026_djf_2.5x2.5.grib": (
            "sha256:74213e8fb335e4ad17d2c07ae4719427bad788e9ac0e00637fd5523a82362e38"
        ),
        "era5_vo850_2025-2026_djf_0.25x0.25.nc": (
            "sha256:907f1d94bebc87a83207d36cd395ce2051829d3a2669ab64befc1243ddb9058d"
        ),
        "era5_vo850_2025-2026_djf_2.5x2.5.nc": (
            "sha256:46ce78cd3b065d3777c2d628cdc2311d68a9fcb4d3a3b9948db7c7376ae7a6aa"
        ),
        "era5_vo850_2025-2026_djf_2.5x2.5.grib": (
            "sha256:9b92f82bb252fbafa90b507df75faa61721062cd70915b8f42da8d803a5a86d7"
        ),
        "era5_uv850_2025-2026_djf_0.25x0.25.nc": (
            "sha256:0e35d002edc8bdbdbe9bc5f965db76dcf983d97b613ba4e217d0a4f5d84c3e51"
        ),
        "era5_uv850_2025-2026_djf_2.5x2.5.nc": (
            "sha256:43cbc346a52c5230ac34eb22c7a640800fbffad40da4058686c8042a76bc5965"
        ),
    },
)


def fetch_era5_msl(
    resolution: str = "2.5x2.5", season: str = "djf", format: str = "nc"
) -> str:
    """
    Fetches the ERA5 mean sea level pressure sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution of the dataset.
            Options: "2.5x2.5" or "0.25x0.25".
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc" (default), "grib", or "zarr".

    Returns:
        str: Absolute path to the downloaded local file or URL for Zarr.
    """
    if resolution not in ["2.5x2.5", "0.25x0.25"]:
        raise ValueError("Resolution must be either '2.5x2.5' or '0.25x0.25'")
    if season not in ["djf"]:
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    if format not in ["nc", "grib", "zarr"]:
        raise ValueError("Format must be 'nc', 'grib', or 'zarr'")

    fname = f"era5_msl_2025-2026_{season}_{resolution}.{format}"
    if format == "zarr":
        return RAW_CONTENT_URL + fname
    return str(CACHED_DATA.fetch(fname))


def fetch_era5_vo850(
    resolution: str = "2.5x2.5", season: str = "djf", format: str = "nc"
) -> str:
    """
    Fetches the ERA5 850hPa relative vorticity sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution of the dataset.
            Options: "2.5x2.5" or "0.25x0.25".
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc" (default), "grib", or "zarr".

    Returns:
        str: Absolute path to the downloaded local file or URL for Zarr.
    """
    if resolution not in ["2.5x2.5", "0.25x0.25"]:
        raise ValueError("Resolution must be either '2.5x2.5' or '0.25x0.25'")
    if season not in ["djf"]:
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    if format not in ["nc", "grib", "zarr"]:
        raise ValueError("Format must be 'nc', 'grib', or 'zarr'")

    fname = f"era5_vo850_2025-2026_{season}_{resolution}.{format}"
    if format == "zarr":
        return RAW_CONTENT_URL + fname
    return str(CACHED_DATA.fetch(fname))


def fetch_era5_uv850(
    resolution: str = "2.5x2.5", season: str = "djf", format: str = "nc"
) -> str:
    """
    Fetches the ERA5 850hPa u- and v-component of wind sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution of the dataset.
            Options: "2.5x2.5" or "0.25x0.25".
        season (str): Season of the dataset. Currently only "djf" is available.
        format (str): File format. Options: "nc" (default) or "zarr".

    Returns:
        str: Absolute path to the downloaded local file or URL for Zarr.
    """
    if resolution not in ["2.5x2.5", "0.25x0.25"]:
        raise ValueError("Resolution must be either '2.5x2.5' or '0.25x0.25'")
    if season not in ["djf"]:
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    if format not in ["nc", "zarr"]:  # UV data has no GRIB for now
        raise ValueError("Format must be 'nc' or 'zarr'")

    fname = f"era5_uv850_2025-2026_{season}_{resolution}.{format}"
    if format == "zarr":
        return RAW_CONTENT_URL + fname
    return str(CACHED_DATA.fetch(fname))
