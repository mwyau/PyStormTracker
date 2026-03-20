from __future__ import annotations

from typing import cast

import numba as nb
import numpy as np

R_EARTH = 6367.0  # Radius of Earth in km
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

    return cast(float, np.arccos(dot))


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def geod_dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great circle distance in kilometers."""
    return cast(float, geod_dist(lat1, lon1, lat2, lon2) * R_EARTH)
