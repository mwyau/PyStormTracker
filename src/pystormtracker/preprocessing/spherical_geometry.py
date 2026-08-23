"""Shared local geometry for feature refinement on the unit sphere."""

from __future__ import annotations

from typing import NamedTuple

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(cache=True, nogil=True)
def sphere_point_and_basis(
    theta: float, phi: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a unit point and the colatitudinal/eastward tangent basis."""
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    point = np.array([st * cp, st * sp, ct], dtype=np.float64)
    e_theta = np.array([ct * cp, ct * sp, -st], dtype=np.float64)
    e_phi = np.array([-sp, cp, 0.0], dtype=np.float64)
    return point, e_theta, e_phi


class SphericalLocalizationRegion(NamedTuple):
    """Candidate-centred tangent rectangle derived from the detector footprint."""

    origin: NDArray[np.float64]
    origin_e_theta: NDArray[np.float64]
    origin_e_phi: NDArray[np.float64]
    theta_half_width: float
    phi_half_width: float


def spherical_localization_region(
    sample_latitudes: NDArray[np.float64],
    sample_longitudes: NDArray[np.float64],
    latitude: float,
    longitude: float,
    search_window_size: int,
) -> SphericalLocalizationRegion | None:
    """Derive the fixed physical local region used by spherical refinements.

    The meridional half-width uses actual latitude spacing.  The zonal
    half-width is its local tangent-space physical length, so neither signed
    longitudes nor a longitude seam require a special case.
    """
    if search_window_size <= 1 or search_window_size % 2 == 0:
        return None
    if sample_latitudes.size == 0 or sample_longitudes.size == 0:
        return None

    theta = float(np.deg2rad(90.0 - latitude))
    phi = float(np.deg2rad(longitude) % (2.0 * np.pi))
    latitude_index = int(np.argmin(np.abs(sample_latitudes - latitude)))
    longitude_differences = np.mod(sample_longitudes - longitude + 180.0, 360.0) - 180.0
    longitude_index = int(np.argmin(np.abs(longitude_differences)))
    half_window = search_window_size // 2

    first_latitude = max(0, latitude_index - half_window)
    last_latitude = min(sample_latitudes.size - 1, latitude_index + half_window)
    theta_half_width = float(
        np.max(
            np.abs(
                np.deg2rad(90.0 - sample_latitudes[first_latitude : last_latitude + 1])
                - theta
            )
        )
    )

    max_longitude_difference = 0.0
    for offset in range(-half_window, half_window + 1):
        index = (longitude_index + offset) % sample_longitudes.size
        difference = (sample_longitudes[index] - longitude + 180.0) % 360.0 - 180.0
        max_longitude_difference = max(max_longitude_difference, abs(difference))
    phi_half_width = float(np.sin(theta) * np.deg2rad(max_longitude_difference))
    if theta_half_width <= 1.0e-12 or phi_half_width <= 1.0e-12:
        return None

    origin, origin_e_theta, origin_e_phi = sphere_point_and_basis(theta, phi)
    return SphericalLocalizationRegion(
        origin=origin,
        origin_e_theta=origin_e_theta,
        origin_e_phi=origin_e_phi,
        theta_half_width=theta_half_width,
        phi_half_width=phi_half_width,
    )


@nb.njit(cache=True, nogil=True)
def inside_spherical_localization_region(
    point: NDArray[np.float64],
    origin: NDArray[np.float64],
    origin_e_theta: NDArray[np.float64],
    origin_e_phi: NDArray[np.float64],
    theta_half_width: float,
    phi_half_width: float,
) -> bool:
    """Return whether a point belongs to a fixed tangent-space rectangle."""
    dot = point[0] * origin[0] + point[1] * origin[1] + point[2] * origin[2]
    dot = min(max(dot, -1.0), 1.0)
    distance = np.arccos(dot)
    if distance <= 1.0e-14:
        return True
    sin_distance = np.sin(distance)
    if abs(sin_distance) <= 1.0e-14:
        return False
    tangent = (point - dot * origin) / sin_distance
    xi_theta = distance * (
        tangent[0] * origin_e_theta[0]
        + tangent[1] * origin_e_theta[1]
        + tangent[2] * origin_e_theta[2]
    )
    xi_phi = distance * (
        tangent[0] * origin_e_phi[0]
        + tangent[1] * origin_e_phi[1]
        + tangent[2] * origin_e_phi[2]
    )
    return bool(
        abs(xi_theta) <= theta_half_width + 1.0e-12
        and abs(xi_phi) <= phi_half_width + 1.0e-12
    )
