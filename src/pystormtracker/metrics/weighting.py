from __future__ import annotations

from enum import IntEnum

import numba as nb


class _WeightType(IntEnum):
    """Supported weight types for spherical kernel estimation.

    The constant kernel represents the hard accumulation radius used by Yau
    and Chang (2020).  ``CRESSMAN`` follows Cressman (1959).  The linear and
    quadratic compact kernels are PyStormTracker generalizations.

    References:
        Yau, A. M.-W., and E. K.-M. Chang (2020). Finding Storm Track
            Activity Metrics That Are Highly Correlated with Weather Impacts.
            Part I. *Journal of Climate*, 33(23), 10169--10186.
            https://doi.org/10.1175/JCLI-D-20-0393.1
        Hodges, K. I. (1996). Spherical Nonparametric Estimators Applied to
            the UGAMP Model Integration for AMIP. *Monthly Weather Review*,
            124(12), 2914--2932.
            https://doi.org/10.1175/1520-0493(1996)124<2914:SNEATT>2.0.CO;2
        Cressman, G. P. (1959). An Operational Objective Analysis System.
            *Monthly Weather Review*, 87(10), 367--374.
            https://doi.org/10.1175/1520-0493(1959)087<0367:AOOAS>2.0.CO;2
    """

    CONSTANT = 0  # Hard radius corresponding to the Yau--Chang rule.
    CRESSMAN = 1  # Cressman (1959) compact rational weighting.
    LINEAR = 2  # PyStormTracker compact linear generalization.
    QUADRATIC = 3  # PyStormTracker compact quadratic generalization.


@nb.njit(cache=True, nogil=True)
def calculate_spherical_weight(
    dist_km: float,
    radius_km: float,
    weight_type: int,
) -> float:
    """
    Compute a spherical distance weight using the selected kernel.

    The formulas are standard or project-specific kernel choices; this helper
    is the numerical implementation used by the gridded metrics.

    Args:
        dist_km: Geodesic distance in kilometers.
        radius_km: Radius of influence in kilometers.
        weight_type: Integer ID from _WeightType enum.

    Returns:
        float: Computed weight.
    """
    if weight_type == _WeightType.CONSTANT:
        return 1.0 if dist_km <= radius_km else 0.0

    if weight_type == _WeightType.CRESSMAN:
        if dist_km > radius_km:
            return 0.0
        r2 = radius_km**2
        d2 = dist_km**2
        return (r2 - d2) / (r2 + d2)

    if weight_type == _WeightType.LINEAR:
        if dist_km > radius_km:
            return 0.0
        return 1.0 - (dist_km / radius_km)

    if weight_type == _WeightType.QUADRATIC:
        if dist_km > radius_km:
            return 0.0
        return 1.0 - (dist_km / radius_km) ** 2

    return 0.0
