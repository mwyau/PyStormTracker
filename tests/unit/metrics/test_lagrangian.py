from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.metrics.tracks import compute_track_metrics
from pystormtracker.models.tracks import Tracks


@pytest.fixture
def equator_crossing_track() -> Tracks:
    """Track crossing the equator and prime meridian."""
    times = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T06:00:00"], dtype="datetime64[ns]"
    )
    lats = np.array([-1.0, 1.0])
    lons = np.array([359.0, 1.0])
    track_ids = np.array([1, 1], dtype=np.int64)
    variables = {"intensity": np.array([100.0, 120.0])}
    return Tracks(track_ids, times, lats, lons, variables)


@pytest.fixture
def polar_track() -> Tracks:
    """Track very close to the North Pole."""
    times = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T06:00:00"], dtype="datetime64[ns]"
    )
    lats = np.array([89.0, 89.5])
    lons = np.array([0.0, 180.0])  # Crosses over the pole
    track_ids = np.array([1, 1], dtype=np.int64)
    variables = {"intensity": np.array([100.0, 100.0])}
    return Tracks(track_ids, times, lats, lons, variables)


def test_equator_wrapping(equator_crossing_track: Tracks) -> None:
    # Grid point exactly at 0,0
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    # Both points are within ~111km of 0,0.
    # Radius 500km should capture both.
    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, radius_km=500.0, kernel="constant"
    )

    # Track frequency should be 1 (one track passed through)
    # Cyclone frequency should be 2 (two points passed through)
    assert ds.track_frequency.values[0, 0, 0] == 1.0
    assert ds.cyclone_frequency.values[0, 0, 0] == 2.0
    # ACA = sum of intensities = 100 + 120 = 220
    assert ds.aca.values[0, 0, 0] == 220.0
    # ATA = max intensity * 1 = 120 * 1 = 120
    assert ds.ata.values[0, 0, 0] == 120.0


def test_polar_distance(polar_track: Tracks) -> None:
    # Grid point exactly at North Pole
    grid_lat = np.array([90.0])
    grid_lon = np.array([0.0])

    # Points are within ~111km and ~55km of the pole
    ds = compute_track_metrics(
        polar_track, grid_lat, grid_lon, radius_km=500.0, kernel="constant"
    )

    assert ds.track_frequency.values[0, 0, 0] == 1.0
    assert ds.cyclone_frequency.values[0, 0, 0] == 2.0


def test_weighted_metrics_consistency(equator_crossing_track: Tracks) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    # With Cressman kernel, weights will be < 1.0
    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, radius_km=500.0, kernel="cressman"
    )

    cf = ds.cyclone_frequency.values[0, 0, 0]
    tf = ds.track_frequency.values[0, 0, 0]
    aca = ds.aca.values[0, 0, 0]
    amp = ds.cyclone_amplitude.values[0, 0, 0]

    # Weight should be between 0 and 1
    assert 0.0 < cf < 2.0
    assert 0.0 < tf < 1.0
    # Amplitude should be weighted average: aca / cf
    assert np.allclose(amp, aca / cf)


def test_fisher_lagrangian(equator_crossing_track: Tracks) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, kernel="fisher", kappa=50.0
    )

    # Fisher kernel values at ~1deg distance (~111km) with kappa=50 should be ~0.4
    # theta = 111 / 6371 = 0.017 rad. cos(theta) = 0.9998. exp(50 * (0.9998 - 1)) = 0.99
    # Actually at 1 deg, weight is very close to 1.
    assert ds.track_frequency.values[0, 0, 0] > 0.5
    assert ds.aca.values[0, 0, 0] > 100.0


def test_linear_lagrangian(equator_crossing_track: Tracks) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    # R = 200km.
    # Points are at (-1, 359) and (1, 1).
    # Distance to (0,0) is approx 1.41 deg ~ 157km.
    # Weight = 1 - 157/200 = 0.215
    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, radius_km=200.0, kernel="linear"
    )

    assert 0.2 < ds.track_frequency.values[0, 0, 0] < 0.25


def test_quadratic_lagrangian(equator_crossing_track: Tracks) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    # R = 200km. dist ~ 157km.
    # Weight = 1 - (157/200)^2 = 1 - 0.616 = 0.384
    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, radius_km=200.0, kernel="quadratic"
    )

    assert 0.35 < ds.track_frequency.values[0, 0, 0] < 0.45


def test_ata_max_logic_multipoint() -> None:
    """Verify that ATA correctly uses the maximum amplitude within the radius."""
    # A track that gets stronger as it moves towards the center, then weaker
    times = np.arange(3) * np.timedelta64(6, "h") + np.datetime64("2020-01-01")
    lats = np.array([2.0, 0.0, -2.0])
    lons = np.array([0.0, 0.0, 0.0])
    track_ids = np.array([1, 1, 1], dtype=np.int64)
    # Amplitudes: 100 at 2deg (~222km), 150 at center, 80 at -2deg (~222km)
    variables = {"intensity": np.array([100.0, 150.0, 80.0])}
    tracks = Tracks(track_ids, times, lats, lons, variables)

    grid_lat, grid_lon = np.array([0.0]), np.array([0.0])

    # Constant kernel
    ds = compute_track_metrics(
        tracks, grid_lat, grid_lon, radius_km=500.0, kernel="constant"
    )

    # ATA should be 150.0 (the maximum amplitude seen by this track in this radius)
    assert ds.ata.values[0, 0, 0] == 150.0
    # ACA should be sum = 100 + 150 + 80 = 330
    assert ds.aca.values[0, 0, 0] == 330.0
