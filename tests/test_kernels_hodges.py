from __future__ import annotations

import numpy as np

from pystormtracker.hodges.kernels import (
    geod_dev,
    get_adaptive_phimax,
    get_regional_dmax,
    subgrid_refine,
)
from pystormtracker.utils.geo import geod_dist


def test_geod_dist() -> None:
    # 0 distance
    assert geod_dist(0.0, 0.0, 0.0, 0.0) == 0.0
    # 90 degrees
    assert np.allclose(geod_dist(0.0, 0.0, 90.0, 0.0), np.pi / 2)
    # 180 degrees
    assert np.allclose(geod_dist(0.0, 0.0, 0.0, 180.0), np.pi)


def test_geod_dev() -> None:
    # Straight line, constant speed -> cost should be 0
    # p0=(0,0), p1=(0,1), p2=(0,2)
    cost = geod_dev(0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.2, 0.8)
    assert np.allclose(cost, 0.0, atol=1e-7)

    # Sharp turn (90 degrees) -> directional cost should be high
    # p0=(0,0), p1=(0,1), p2=(1,1)
    cost = geod_dev(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    # Tangent at p1: T1=(0,1), T2=(1,0) -> dot=0 -> cost=0.5*1.0*(1-0)=0.5
    assert np.allclose(cost, 0.5)

    # Speed change -> speed cost should be high
    # p0=(0,0), p1=(0,1), p2=(0,1.1)
    # alpha1=1, alpha2=0.1 -> w2*(1 - 2*sqrt(0.1)/1.1)
    cost = geod_dev(0.0, 0.0, 0.0, 1.0, 0.0, 1.1, 0.0, 1.0)
    expected = 1.0 - 2.0 * np.sqrt(1.0 * 0.1) / 1.1
    assert np.allclose(cost, expected)


def test_get_regional_dmax() -> None:
    zones = np.array(
        [
            [0.0, 360.0, -90.0, -20.0, 6.5],
            [0.0, 360.0, -20.0, 20.0, 3.0],
            [0.0, 360.0, 20.0, 90.0, 6.5],
        ]
    )
    # Tropics
    assert get_regional_dmax(0.0, 180.0, zones, 5.0) == 3.0
    # Extratropics
    assert get_regional_dmax(45.0, 180.0, zones, 5.0) == 6.5
    # Default fallback (if zones empty)
    assert get_regional_dmax(0.0, 0.0, np.zeros((0, 5)), 5.0) == 5.0


def test_get_adaptive_phimax() -> None:
    adapt_params = np.array([[1.0, 2.0, 5.0, 8.0], [1.0, 0.3, 0.1, 0.0]])

    # Below min
    assert get_adaptive_phimax(0.5, adapt_params, 0.5) == 1.0
    # Above max
    assert get_adaptive_phimax(10.0, adapt_params, 0.5) == 0.0
    # On threshold
    assert get_adaptive_phimax(2.0, adapt_params, 0.5) == 0.3
    # Interpolated
    # Between 1.0 and 2.0, mean is 1.5 -> (1.0 + 0.3)/2 = 0.65
    assert np.allclose(get_adaptive_phimax(1.5, adapt_params, 0.5), 0.65)


def test_subgrid_refine() -> None:
    # Create a 3x3 with a peak at center
    frame = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([100.0, 101.0, 102.0])

    rlat, rlon, rval = subgrid_refine(frame, 1, 1, lat, lon)
    assert rlat == 11.0
    assert rlon == 101.0
    assert rval == 1.0

    # Peak shifted slightly
    frame = np.array([[0.0, 0.6, 0.0], [0.5, 1.0, 0.5], [0.0, 0.4, 0.0]])
    rlat, rlon, rval = subgrid_refine(frame, 1, 1, lat, lon)
    assert rlat < 11.0  # Peak is between 10 and 11
    assert rlon == 101.0
