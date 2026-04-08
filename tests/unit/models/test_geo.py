from __future__ import annotations

import numpy as np

from pystormtracker.models.constants import R_EARTH_KM
from pystormtracker.models.geo import stereo_to_latlon


def test_stereo_to_latlon_nh() -> None:
    # Test North Pole
    lat, lon = stereo_to_latlon(0.0, 0.0, hemisphere=1)
    assert np.isclose(lat, 90.0)

    # Test a point on the equator (radius = 2 * R * tan(pi/4) = 2 * R)
    r_eq = 2.0 * R_EARTH_KM

    # Point at x=0, y=r_eq => phi = arctan2(0, -r_eq) = arctan2(0, -1) = pi => lon=180
    lat, lon = stereo_to_latlon(0.0, r_eq, hemisphere=1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 180.0, atol=1e-7)

    # Point at x=r_eq, y=0 => phi = arctan2(r_eq, 0) = pi/2 => lon=90
    lat, lon = stereo_to_latlon(r_eq, 0.0, hemisphere=1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 90.0, atol=1e-7)


def test_stereo_to_latlon_sh() -> None:
    # Test South Pole
    lat, lon = stereo_to_latlon(0.0, 0.0, hemisphere=-1)
    assert np.isclose(lat, -90.0)

    # Test a point on the equator
    r_eq = 2.0 * R_EARTH_KM

    # Point at x=0, y=r_eq => phi = arctan2(0, r_eq) = 0 => lon=0
    lat, lon = stereo_to_latlon(0.0, r_eq, hemisphere=-1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 0.0, atol=1e-7)

    # Point at x=r_eq, y=0 => phi = arctan2(r_eq, 0) = pi/2 => lon=90
    lat, lon = stereo_to_latlon(r_eq, 0.0, hemisphere=-1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 90.0, atol=1e-7)
