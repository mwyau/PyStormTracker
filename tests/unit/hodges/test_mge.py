from __future__ import annotations

import numpy as np

from pystormtracker.hodges.mge import (
    _compute_adaptive_phimax,
    _mge_iteration,
    _select_regional_dmax,
    geod_dev,
)
from pystormtracker.models.geo import geod_dist


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


def test_select_regional_dmax() -> None:
    zones = np.array(
        [
            [0.0, 360.0, -90.0, -20.0, 6.5],
            [0.0, 360.0, -20.0, 20.0, 3.0],
            [0.0, 360.0, 20.0, 90.0, 6.5],
        ]
    )
    # Tropics
    assert _select_regional_dmax(0.0, 180.0, zones, 5.0) == 3.0
    # Extratropics
    assert _select_regional_dmax(45.0, 180.0, zones, 5.0) == 6.5
    # Default fallback (if zones empty)
    assert _select_regional_dmax(0.0, 0.0, np.zeros((0, 5)), 5.0) == 5.0


def test_select_regional_dmax_normalizes_signed_longitudes_for_360_zones() -> None:
    zones = np.array(
        [[300.0, 360.0, -90.0, 90.0, 2.0], [0.0, 300.0, -90.0, 90.0, 4.0]],
        dtype=np.float64,
    )

    assert _select_regional_dmax(0.0, -10.0, zones, 5.0) == 2.0


def test_compute_adaptive_phimax() -> None:
    adaptive_smoothness = np.array([[1.0, 2.0, 5.0, 8.0], [1.0, 0.3, 0.1, 0.0]])

    # Below min
    assert _compute_adaptive_phimax(0.5, adaptive_smoothness, 0.5) == 1.0
    # Above max
    assert _compute_adaptive_phimax(10.0, adaptive_smoothness, 0.5) == 0.0
    # On threshold
    assert _compute_adaptive_phimax(2.0, adaptive_smoothness, 0.5) == 0.3
    assert _compute_adaptive_phimax(1.0, adaptive_smoothness, 0.5) == 1.0
    assert _compute_adaptive_phimax(5.0, adaptive_smoothness, 0.5) == 0.1
    assert _compute_adaptive_phimax(8.0, adaptive_smoothness, 0.5) == 0.0
    # Interpolated
    # Between 1.0 and 2.0, mean is 1.5 -> (1.0 + 0.3)/2 = 0.65
    assert np.allclose(_compute_adaptive_phimax(1.5, adaptive_smoothness, 0.5), 0.65)


def test_mge_iteration_accepts_a_beneficial_swap_and_restores_scan_state() -> None:
    tracks = np.array([[0, 2, 5], [1, 3, 4]], dtype=np.int64)
    original = tracks.copy()
    features_lat = np.zeros(6, dtype=np.float64)
    features_lon = np.array([0.0, 10.0, 1.0, 9.0, 2.0, 8.0], dtype=np.float64)

    result = _mge_iteration(
        tracks,
        features_lat,
        features_lon,
        1,
        True,
        0.5,
        0.5,
        np.array([20.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.zeros(3, dtype=np.int64),
        np.empty((0, 5), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
    )

    assert result == (0, 1)
    np.testing.assert_array_equal(tracks, original)


def test_mge_iteration_rejects_a_swap_over_the_displacement_limit() -> None:
    tracks = np.array([[0, 2, 5], [1, 3, 4]], dtype=np.int64)
    features_lat = np.zeros(6, dtype=np.float64)
    features_lon = np.array([0.0, 10.0, 1.0, 9.0, 8.0, 2.0], dtype=np.float64)

    result = _mge_iteration(
        tracks,
        features_lat,
        features_lon,
        1,
        True,
        0.5,
        0.5,
        np.array([5.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.zeros(3, dtype=np.int64),
        np.empty((0, 5), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
    )

    assert result == (-1, -1)


def test_mge_iteration_rejects_a_swap_over_adaptive_smoothness_limit() -> None:
    tracks = np.array([[0, 2, 5], [1, 3, 4]], dtype=np.int64)
    features_lat = np.array([0.0, 0.0, 0.0, 0.0, 2.0, -2.0], dtype=np.float64)
    features_lon = np.array([0.0, 10.0, 2.0, 8.0, 4.0, 2.0], dtype=np.float64)
    dmax_parameters = np.array([10.0], dtype=np.float64)
    phimax_parameters = np.array([1.0], dtype=np.float64)
    missing_counts = np.zeros(3, dtype=np.int64)
    zones = np.empty((0, 5), dtype=np.float64)

    beneficial_without_adaptation = _mge_iteration(
        tracks.copy(),
        features_lat,
        features_lon,
        1,
        True,
        0.5,
        0.5,
        dmax_parameters,
        phimax_parameters,
        missing_counts,
        zones,
        np.zeros((2, 0), dtype=np.float64),
    )
    rejected_with_adaptation = _mge_iteration(
        tracks.copy(),
        features_lat,
        features_lon,
        1,
        True,
        0.5,
        0.5,
        dmax_parameters,
        phimax_parameters,
        missing_counts,
        zones,
        np.array(
            [[0.0, 10.0, 20.0, 30.0], [0.05, 0.05, 0.05, 0.05]],
            dtype=np.float64,
        ),
    )

    assert beneficial_without_adaptation == (0, 1)
    assert rejected_with_adaptation == (-1, -1)
