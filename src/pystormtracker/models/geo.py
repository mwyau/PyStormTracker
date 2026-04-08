from __future__ import annotations

from typing import TypeAlias

import numba as nb
import numpy as np

from ..models.constants import R_EARTH_KM

# Type alias for a map bounding box (xmin, xmax, ymin, ymax) in km
MapExtent: TypeAlias = tuple[float, float, float, float]

DEGTORAD = np.pi / 180.0


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def geod_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great circle distance (angular separation) in radians."""
    phi1 = lat1 * DEGTORAD
    phi2 = lat2 * DEGTORAD
    lam1 = lon1 * DEGTORAD
    lam2 = lon2 * DEGTORAD

    # Dot product of unit vectors
    dot = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(
        lam1 - lam2
    )

    # Clamp for precision
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0

    return float(np.arccos(dot))


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def geod_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great circle distance in kilometers."""
    return float(geod_dist(lat1, lon1, lat2, lon2) * R_EARTH_KM)


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def stereo_to_latlon(
    x: float, y: float, hemisphere: int, lon_0: float = 0.0
) -> tuple[float, float]:
    """
    Converts (x, y) coordinates on a polar stereographic projection (in km)
    back to (lat, lon) in degrees.

    Args:
        x: X coordinate in km.
        y: Y coordinate in km.
        hemisphere: 1 for Northern Hemisphere, -1 for Southern Hemisphere.
        lon_0: Central longitude in degrees.

    Returns:
        (lat, lon) in degrees.
    """
    rho = np.sqrt(x**2 + y**2)
    if rho == 0.0:
        return 90.0 * hemisphere, lon_0

    if hemisphere == 1:
        theta = 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
        phi = (np.radians(lon_0) + np.arctan2(x, -y)) % (2 * np.pi)
    else:
        theta = np.pi - 2.0 * np.arctan(rho / (2.0 * R_EARTH_KM))
        phi = (np.radians(lon_0) + np.arctan2(x, y)) % (2 * np.pi)

    lat = 90.0 - np.degrees(theta)
    lon = np.degrees(phi) % 360.0
    return lat, lon
