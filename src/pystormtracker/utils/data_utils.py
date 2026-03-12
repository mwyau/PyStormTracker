from __future__ import annotations

import pooch  # type: ignore[import-untyped]

DATA_RELEASE_VERSION = "v0.1.0-data"

# Define a central repository of data files
GOOD_DATA = pooch.create(
    path=pooch.os_cache("pystormtracker"),
    base_url=(
        f"https://github.com/mwyau/PyStormTracker-Data/"
        f"releases/download/{DATA_RELEASE_VERSION}/"
    ),
    registry={
        "era5_msl_2025-2026_djf_0.25x0.25.nc": (
            "sha256:a1847093356472303585eb9acdbfb8c993795a2e643e80d5f7cc803d0919216d"
        ),
        "era5_msl_2025-2026_djf_2.5x2.5.nc": (
            "sha256:19477e18e4239b9f8ea5a7b7a56c2f3790fbc661bbff1a949e59ebda1a61fc40"
        ),
        "era5_vo850_2025-2026_djf_0.25x0.25.nc": (
            "sha256:907f1d94bebc87a83207d36cd395ce2051829d3a2669ab64befc1243ddb9058d"
        ),
        "era5_vo850_2025-2026_djf_2.5x2.5.nc": (
            "sha256:46ce78cd3b065d3777c2d628cdc2311d68a9fcb4d3a3b9948db7c7376ae7a6aa"
        ),
    },
)


def fetch_era5_msl(resolution: str = "2.5x2.5", season: str = "djf") -> str:
    """
    Fetches the ERA5 mean sea level pressure sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution of the dataset.
            Options: "2.5x2.5" or "0.25x0.25".
        season (str): Season of the dataset. Currently only "djf" is available.

    Returns:
        str: Absolute path to the downloaded local NetCDF file.
    """
    if resolution not in ["2.5x2.5", "0.25x0.25"]:
        raise ValueError("Resolution must be either '2.5x2.5' or '0.25x0.25'")
    if season not in ["djf"]:
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    return str(GOOD_DATA.fetch(f"era5_msl_2025-2026_{season}_{resolution}.nc"))


def fetch_era5_vo850(resolution: str = "2.5x2.5", season: str = "djf") -> str:
    """
    Fetches the ERA5 850hPa relative vorticity sample dataset.
    Downloads the data on the first call and returns the path to the cached local file.

    Args:
        resolution (str): Spatial resolution of the dataset.
            Options: "2.5x2.5" or "0.25x0.25".
        season (str): Season of the dataset. Currently only "djf" is available.

    Returns:
        str: Absolute path to the downloaded local NetCDF file.
    """
    if resolution not in ["2.5x2.5", "0.25x0.25"]:
        raise ValueError("Resolution must be either '2.5x2.5' or '0.25x0.25'")
    if season not in ["djf"]:
        raise ValueError(f"Season '{season}' not available. Options: 'djf'")
    return str(GOOD_DATA.fetch(f"era5_vo850_2025-2026_{season}_{resolution}.nc"))
