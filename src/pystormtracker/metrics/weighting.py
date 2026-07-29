from __future__ import annotations

from enum import IntEnum

import numba as nb
import numpy as np


class WeightType(IntEnum):
    """Supported weight types for spherical kernel estimation."""

    CONSTANT = 0  # Threshold method (Yau and Chang 2020)
    FISHER = 1  # Exponential kernel (Hodges 1999)
    CRESSMAN = 2  # Rational kernel (Cressman 1959)
    LINEAR = 3  # Linear decay
    QUADRATIC = 4  # Quadratic decay


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def calculate_spherical_weight(
    dist_km: float,
    radius_km: float,
    weight_type: int,
    kappa: float,
) -> float:
    """
    Computes weight based on kernel type.

    Args:
        dist_km: Geodesic distance in kilometers.
        radius_km: Radius of influence in kilometers.
        weight_type: Integer ID from WeightType enum.
        kappa: Smoothing parameter for Fisher kernel.

    Returns:
        float: Computed weight.
    """
    if weight_type == 0:  # CONSTANT
        return 1.0 if dist_km <= radius_km else 0.0

    if weight_type == 1:  # FISHER
        # weight = exp(kappa * (cos(theta) - 1))
        # theta = dist / R_earth (6371.22 km)
        theta = dist_km / 6371.22
        return float(np.exp(kappa * (np.cos(theta) - 1.0)))

    if weight_type == 2:  # CRESSMAN
        if dist_km > radius_km:
            return 0.0
        r2 = radius_km**2
        d2 = dist_km**2
        return (r2 - d2) / (r2 + d2)

    if weight_type == 3:  # LINEAR
        if dist_km > radius_km:
            return 0.0
        return 1.0 - (dist_km / radius_km)

    if weight_type == 4:  # QUADRATIC
        if dist_km > radius_km:
            return 0.0
        return 1.0 - (dist_km / radius_km) ** 2

    return 0.0
