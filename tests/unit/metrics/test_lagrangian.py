from __future__ import annotations

import numpy as np
import pytest
from xarray.testing import assert_identical

from pystormtracker.metrics.lagrangian import (
    ATAInterpolation,
    _interpolate_ata_track,
    compute_track_metrics,
)
from pystormtracker.models.tracks import Tracks, TracksMetadata


def _packed(
    track_ids: np.ndarray,
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variables: dict[str, np.ndarray],
) -> Tracks:
    return Tracks(
        ids=np.array([track_ids[0]], dtype=np.int64),
        offsets=np.array([0, len(track_ids)], dtype=np.int64),
        times=times,
        lats=lats,
        lons=lons,
        variables=variables,
        metadata=TracksMetadata("intensity", "max", {"intensity": "1"}),
    )


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
    return _packed(track_ids, times, lats, lons, variables)


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
    return _packed(track_ids, times, lats, lons, variables)


def test_equator_wrapping(equator_crossing_track: Tracks) -> None:
    # Grid point exactly at 0,0
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    # Both points are about 157 km from 0,0.
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


def test_fisher_kernel_weighting(equator_crossing_track: Tracks) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])

    ds = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, kernel="fisher", kappa=50.0
    )

    # At about 157 km, theta = 157 / 6371 = 0.025 rad, so
    # exp(50 * (cos(theta) - 1)) is close to 1.
    assert ds.track_frequency.values[0, 0, 0] > 0.5
    assert ds.aca.values[0, 0, 0] > 100.0


def test_linear_kernel_weighting(equator_crossing_track: Tracks) -> None:
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


def test_quadratic_kernel_weighting(equator_crossing_track: Tracks) -> None:
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
    tracks = _packed(track_ids, times, lats, lons, variables)

    grid_lat, grid_lon = np.array([0.0]), np.array([0.0])

    # Constant kernel
    ds = compute_track_metrics(
        tracks, grid_lat, grid_lon, radius_km=500.0, kernel="constant"
    )

    # ATA should be 150.0 (the maximum amplitude seen by this track in this radius)
    assert ds.ata.values[0, 0, 0] == 150.0
    # ACA should be sum = 100 + 150 + 80 = 330
    assert ds.aca.values[0, 0, 0] == 330.0


def _hours(*values: float) -> np.ndarray:
    """Return elapsed-hour values in the packed canonical millisecond unit."""
    return np.asarray([int(value * 3_600_000) for value in values], dtype=np.int64)


def test_linear_hourly_time_and_amplitude_interpolation() -> None:
    times, _, _, amplitudes = _interpolate_ata_track(
        _hours(0, 6),
        np.array([0.0, 0.0]),
        np.array([0.0, 6.0]),
        np.array([0.0, 6.0]),
        "linear",
    )

    np.testing.assert_array_equal(times, _hours(0, 1, 2, 3, 4, 5, 6))
    np.testing.assert_allclose(amplitudes, np.arange(7.0), rtol=0.0, atol=1e-14)


def test_hourly_generation_preserves_non_hourly_endpoint() -> None:
    times, _, _, _ = _interpolate_ata_track(
        _hours(0, 2.5),
        np.array([0.0, 0.0]),
        np.array([0.0, 2.0]),
        np.array([0.0, 2.0]),
        "linear",
    )

    np.testing.assert_array_equal(times, _hours(0, 1, 2, 2.5))


def test_linear_position_interpolation_on_equator() -> None:
    _, latitudes, longitudes, _ = _interpolate_ata_track(
        _hours(0, 6),
        np.array([0.0, 0.0]),
        np.array([0.0, 6.0]),
        np.array([0.0, 0.0]),
        "linear",
    )

    np.testing.assert_allclose(latitudes, 0.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(longitudes, np.arange(7.0), rtol=0.0, atol=1e-14)


def test_linear_position_interpolation_crosses_antimeridian() -> None:
    _, _, longitudes, _ = _interpolate_ata_track(
        _hours(0, 2),
        np.array([0.0, 0.0]),
        np.array([179.0, -179.0]),
        np.array([0.0, 0.0]),
        "linear",
    )

    np.testing.assert_allclose(
        longitudes,
        np.array([179.0, -180.0, -179.0]),
        rtol=0.0,
        atol=1e-14,
    )
    assert np.all((longitudes >= -180.0) & (longitudes < 180.0))


@pytest.mark.parametrize("interpolation", ["linear", "linear_pchip"])
def test_interpolation_preserves_sharp_turn_at_observed_knot(
    interpolation: ATAInterpolation,
) -> None:
    _, latitudes, longitudes, _ = _interpolate_ata_track(
        _hours(0, 6, 12),
        np.array([0.0, 0.0, 30.0]),
        np.array([0.0, 60.0, 60.0]),
        np.array([0.0, 1.0, 2.0]),
        interpolation,
    )

    assert latitudes[6] == 0.0
    assert longitudes[6] == 60.0
    np.testing.assert_allclose(latitudes[:7], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(longitudes[7:], 60.0, rtol=0.0, atol=1e-12)
    assert latitudes[-1] == 30.0


@pytest.mark.parametrize("interpolation", ["linear", "linear_pchip"])
def test_interpolation_preserves_backtracking(
    interpolation: ATAInterpolation,
) -> None:
    _, _, longitudes, _ = _interpolate_ata_track(
        _hours(0, 6, 12),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 60.0, 0.0]),
        np.array([0.0, 1.0, 2.0]),
        interpolation,
    )

    assert longitudes[6] == 60.0
    assert np.all(np.diff(longitudes[:7]) > 0.0)
    assert np.all(np.diff(longitudes[6:]) < 0.0)
    assert longitudes[-1] == 0.0


def test_pchip_preserves_observed_amplitude_knots() -> None:
    _, _, _, amplitudes = _interpolate_ata_track(
        _hours(0, 6, 12),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
        "linear_pchip",
    )

    np.testing.assert_allclose(amplitudes[[0, 6, 12]], [0.0, 10.0, 0.0])


def test_pchip_does_not_overshoot_a_local_extremum() -> None:
    _, _, _, amplitudes = _interpolate_ata_track(
        _hours(0, 6, 12),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
        "linear_pchip",
    )

    assert np.min(amplitudes) >= 0.0
    assert np.max(amplitudes) <= 10.0
    assert amplitudes[6] == np.max(amplitudes)


def test_pchip_preserves_monotonic_amplitude() -> None:
    _, _, _, monotonic = _interpolate_ata_track(
        _hours(0, 6, 12),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 5.0, 10.0]),
        "linear_pchip",
    )
    assert np.all(np.diff(monotonic) >= 0.0)


def test_pchip_matches_linear_for_a_two_point_track() -> None:
    _, pchip_lats, pchip_lons, two_point_amplitudes = _interpolate_ata_track(
        _hours(0, 6),
        np.array([0.0, 1.0]),
        np.array([0.0, 90.0]),
        np.array([0.0, 6.0]),
        "linear_pchip",
    )
    _, linear_lats, linear_lons, linear_amplitudes = _interpolate_ata_track(
        _hours(0, 6),
        np.array([0.0, 1.0]),
        np.array([0.0, 90.0]),
        np.array([0.0, 6.0]),
        "linear",
    )
    np.testing.assert_array_equal(pchip_lats, linear_lats)
    np.testing.assert_array_equal(pchip_lons, linear_lons)
    np.testing.assert_allclose(
        two_point_amplitudes, linear_amplitudes, rtol=0.0, atol=1e-14
    )


def test_linear_pchip_changes_only_amplitude_between_knots() -> None:
    inputs = (
        _hours(0, 6, 12),
        np.array([0.0, 5.0, 10.0]),
        np.array([170.0, -175.0, -160.0]),
        np.array([0.0, 10.0, 0.0]),
    )
    linear = _interpolate_ata_track(*inputs, "linear")
    linear_pchip = _interpolate_ata_track(*inputs, "linear_pchip")

    np.testing.assert_array_equal(linear[0], linear_pchip[0])
    np.testing.assert_array_equal(linear[1], linear_pchip[1])
    np.testing.assert_array_equal(linear[2], linear_pchip[2])
    assert not np.array_equal(linear[3], linear_pchip[3])
    np.testing.assert_array_equal(linear_pchip[3][[0, 6, 12]], inputs[3])
    assert np.min(linear_pchip[3]) >= 0.0
    assert np.max(linear_pchip[3]) <= 10.0


def test_non_increasing_track_times_are_rejected() -> None:
    with pytest.raises(ValueError, match="track times must be strictly increasing"):
        _interpolate_ata_track(
            _hours(0, 6, 6),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            "linear",
        )


def test_single_point_track_is_preserved() -> None:
    inputs = (
        _hours(3),
        np.array([12.0]),
        np.array([-45.0]),
        np.array([8.0]),
    )

    output = _interpolate_ata_track(*inputs, "linear_pchip")

    for actual, expected in zip(output, inputs, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_compute_track_metrics_rejects_unknown_interpolation(
    equator_crossing_track: Tracks,
) -> None:
    with pytest.raises(ValueError, match="Unknown ATA interpolation"):
        compute_track_metrics(
            equator_crossing_track,
            np.array([0.0]),
            np.array([0.0]),
            monthly=False,
            interpolation="invalid",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )


def test_default_ata_interpolation_matches_explicit_linear(
    equator_crossing_track: Tracks,
) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])
    default = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, monthly=False
    )
    explicit = compute_track_metrics(
        equator_crossing_track,
        grid_lat,
        grid_lon,
        monthly=False,
        interpolation="linear",
    )

    assert_identical(default, explicit)


def test_interpolation_modes_leave_non_ata_metrics_unchanged(
    equator_crossing_track: Tracks,
) -> None:
    grid_lat = np.array([0.0])
    grid_lon = np.array([0.0])
    baseline = compute_track_metrics(
        equator_crossing_track, grid_lat, grid_lon, monthly=False
    )

    alternate = compute_track_metrics(
        equator_crossing_track,
        grid_lat,
        grid_lon,
        monthly=False,
        interpolation="linear_pchip",
    )
    for variable in (
        "cyclone_amplitude",
        "cyclone_frequency",
        "track_frequency",
        "aca",
    ):
        np.testing.assert_array_equal(
            baseline[variable].values, alternate[variable].values
        )


def test_hourly_interpolation_detects_fast_cyclone_encounter() -> None:
    tracks = _packed(
        np.array([1, 1], dtype=np.int64),
        np.array(
            ["2020-01-01T00:00:00", "2020-01-01T06:00:00"], dtype="datetime64[ns]"
        ),
        np.array([-3.0, 3.0]),
        np.array([0.0, 0.0]),
        {"intensity": np.array([10.0, 20.0])},
    )
    ds = compute_track_metrics(
        tracks,
        np.array([0.0]),
        np.array([0.0]),
        radius_km=200.0,
        monthly=False,
    )

    assert ds.cyclone_frequency.values[0, 0] == 0.0
    assert ds.track_frequency.values[0, 0] == 0.0
    assert ds.ata.values[0, 0] == pytest.approx(50.0 / 3.0, abs=1e-12)


def test_antimeridian_representation_invariance() -> None:
    first = _packed(
        np.array([1, 1], dtype=np.int64),
        np.array(
            ["2020-01-01T00:00:00", "2020-01-01T06:00:00"], dtype="datetime64[ns]"
        ),
        np.array([0.0, 0.0]),
        np.array([179.0, -179.0]),
        {"intensity": np.array([10.0, 20.0])},
    )
    equivalent = _packed(
        np.array([1, 1], dtype=np.int64),
        np.array(
            ["2020-01-01T00:00:00", "2020-01-01T06:00:00"], dtype="datetime64[ns]"
        ),
        np.array([0.0, 0.0]),
        np.array([179.0, 181.0]),
        {"intensity": np.array([10.0, 20.0])},
    )

    for interpolation in ("linear", "linear_pchip"):
        first_result = compute_track_metrics(
            first,
            np.array([0.0]),
            np.array([180.0]),
            monthly=False,
            interpolation=interpolation,
        )
        equivalent_result = compute_track_metrics(
            equivalent,
            np.array([0.0]),
            np.array([180.0]),
            monthly=False,
            interpolation=interpolation,
        )
        np.testing.assert_allclose(first_result.ata, equivalent_result.ata)


def test_ata_interpolation_does_not_mutate_tracks(
    equator_crossing_track: Tracks,
) -> None:
    times = equator_crossing_track.times.copy()
    lats = equator_crossing_track.lats.copy()
    lons = equator_crossing_track.lons.copy()
    variables = {
        name: values.copy() for name, values in equator_crossing_track.variables.items()
    }
    metadata = equator_crossing_track.metadata

    compute_track_metrics(
        equator_crossing_track,
        np.array([0.0]),
        np.array([0.0]),
        monthly=False,
        interpolation="linear_pchip",
    )

    np.testing.assert_array_equal(equator_crossing_track.times, times)
    np.testing.assert_array_equal(equator_crossing_track.lats, lats)
    np.testing.assert_array_equal(equator_crossing_track.lons, lons)
    for name, values in variables.items():
        np.testing.assert_array_equal(equator_crossing_track.variables[name], values)
    assert equator_crossing_track.metadata == metadata
