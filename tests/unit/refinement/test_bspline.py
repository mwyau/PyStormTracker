from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.hodges.detector import _detect_track_rectangular_candidates
from pystormtracker.refinement import (
    build_bspline_surface,
    build_spherical_bspline_surface,
    refine_bspline_feature_point,
    refine_spherical_bspline_feature_point,
)
from pystormtracker.refinement.bspline import (
    _eval_bspline_2d,
    build_bspline_surface_reference,
    prepare_rectangular_grid,
)


def _spherical_peak(
    theta: NDArray[np.float64],
    phi: NDArray[np.float64],
    theta_center: float,
    phi_center: float,
    concentration: float,
) -> NDArray[np.float64]:
    values = concentration * (
        np.cos(theta) * np.cos(theta_center)
        + np.sin(theta) * np.sin(theta_center) * np.cos(phi - phi_center)
        - 1.0
    )
    return np.asarray(np.exp(values), dtype=np.float64)


def test_spherical_spline_refines_a_spherical_off_grid_maximum() -> None:
    latitudes = np.linspace(-80.0, 80.0, 19)
    longitudes = np.arange(0.0, 360.0, 10.0)
    theta = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi = np.deg2rad(longitudes)[np.newaxis, :]
    frame = np.maximum(
        _spherical_peak(
            theta,
            phi,
            np.deg2rad(103.4),
            np.deg2rad(216.7),
            45.0,
        ),
        _spherical_peak(theta, phi, 1.0, 0.0, 1.0),
    )

    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )

    assert built.status == "success"
    assert built.surface is not None
    refined = refine_spherical_bspline_feature_point(
        built.surface,
        -17.77777777777778,
        220.0,
        is_minimum=False,
    )

    assert refined.status == "success"
    assert refined.latitude == pytest.approx(-13.4, abs=0.5)
    assert refined.longitude == pytest.approx(216.7, abs=0.5)
    assert refined.value > 0.96


def test_spherical_spline_reports_unsupported_regional_construction() -> None:
    built = build_spherical_bspline_surface(
        np.ones((4, 4), dtype=np.float64),
        np.linspace(-45.0, 45.0, 4),
        np.arange(4.0),
        periodic_x=False,
    )

    assert built.surface is None
    assert built.status == "spline_construction_failure"


def test_rectangular_spline_refines_a_rectangular_off_grid_minimum() -> None:
    latitudes = np.linspace(-60.0, 60.0, 13)
    longitudes = np.linspace(0.0, 360.0, 25, endpoint=False)
    target_latitude = 12.5
    target_longitude = 152.0
    frame = (latitudes[:, None] - target_latitude) ** 2 + (
        longitudes[None, :] - target_longitude
    ) ** 2

    built = build_bspline_surface(
        frame,
        latitudes,
        longitudes,
        periodic_x=True,
    )

    assert built.status == "success"
    assert built.surface is not None
    refined = refine_bspline_feature_point(
        built.surface,
        10.0,
        158.4,
        is_minimum=True,
        max_iterations=100,
    )

    assert refined.status == "success"
    assert refined.latitude == pytest.approx(target_latitude, abs=1.0e-3)
    assert refined.longitude == pytest.approx(target_longitude, abs=1.0e-3)


def test_cached_rectangular_grid_preparation_is_exact() -> None:
    from scipy.interpolate import RectBivariateSpline

    latitudes = np.linspace(-60.0, 60.0, 17, dtype=np.float64)
    unsigned_longitudes = np.linspace(0.0, 360.0, 36, endpoint=False)
    order = np.argsort((unsigned_longitudes + 180.0) % 360.0 - 180.0)
    longitudes = (unsigned_longitudes[order] + 180.0) % 360.0 - 180.0
    frame = np.sin(np.radians(latitudes[:, None])) + np.cos(
        np.radians(longitudes[None, :])
    )
    grid = prepare_rectangular_grid(latitudes, longitudes, periodic_x=True)
    reference = build_bspline_surface_reference(
        frame, latitudes, longitudes, periodic_x=True
    )
    cached = build_bspline_surface(
        frame,
        latitudes,
        longitudes,
        periodic_x=True,
        grid=grid,
    )
    assert reference.status == cached.status == "success"
    assert reference.surface is not None
    assert cached.surface is not None
    np.testing.assert_array_equal(cached.surface.x_knots, reference.surface.x_knots)
    np.testing.assert_array_equal(cached.surface.y_knots, reference.surface.y_knots)
    np.testing.assert_allclose(
        cached.surface.coeffs,
        reference.surface.coeffs,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    assert cached.surface.x_lower == reference.surface.x_lower
    assert cached.surface.x_upper == reference.surface.x_upper
    assert cached.surface.y_lower == reference.surface.y_lower
    assert cached.surface.y_upper == reference.surface.y_upper
    assert cached.surface.first_sample_x == reference.surface.first_sample_x
    assert cached.surface.last_sample_x == reference.surface.last_sample_x
    assert cached.surface.periodic_x == reference.surface.periodic_x

    sorted_x = grid.sorted_longitudes
    sorted_y = grid.sorted_latitudes
    sorted_frame = frame[grid.latitude_order, :][:, grid.longitude_order]
    extended_x = np.concatenate((sorted_x, [sorted_x[0] + 360.0]))
    extended_frame = np.concatenate((sorted_frame, sorted_frame[:, :1]), axis=1)
    scipy_surface = RectBivariateSpline(
        extended_x, sorted_y, extended_frame.T, kx=3, ky=3, s=0.0
    )
    raw_coefficients = np.asarray(scipy_surface.tck[2], dtype=np.float64)
    expected_coefficients = raw_coefficients.reshape(
        len(scipy_surface.tck[0]) - 4,
        len(scipy_surface.tck[1]) - 4,
    )
    np.testing.assert_allclose(
        cached.surface.coeffs,
        expected_coefficients,
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_cached_rectangular_system_depends_only_on_grid() -> None:
    latitudes = np.linspace(-75.0, 75.0, 17, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float64)
    grid = prepare_rectangular_grid(latitudes, longitudes, periodic_x=True)
    same_grid = prepare_rectangular_grid(latitudes, longitudes, periodic_x=True)
    frame_a = np.sin(np.deg2rad(latitudes[:, None])) + np.cos(
        np.deg2rad(longitudes[None, :])
    )
    frame_b = 2.0 * frame_a + np.sin(np.deg2rad(3.0 * longitudes))[None, :]

    first = build_bspline_surface(
        frame_a, latitudes, longitudes, periodic_x=True, grid=grid
    )
    second = build_bspline_surface(
        frame_b, latitudes, longitudes, periodic_x=True, grid=grid
    )

    assert first.surface is not None
    assert second.surface is not None
    np.testing.assert_array_equal(first.surface.x_knots, second.surface.x_knots)
    np.testing.assert_array_equal(first.surface.y_knots, second.surface.y_knots)
    for first_factor, second_factor in (
        (grid.x_factor, same_grid.x_factor),
        (grid.y_factor, same_grid.y_factor),
    ):
        np.testing.assert_array_equal(first_factor.upper, second_factor.upper)
        np.testing.assert_array_equal(first_factor.cosines, second_factor.cosines)
        np.testing.assert_array_equal(first_factor.sines, second_factor.sines)
        np.testing.assert_array_equal(first_factor.active, second_factor.active)
        np.testing.assert_array_equal(
            first_factor.row_numbers, second_factor.row_numbers
        )
    assert not np.array_equal(first.surface.coeffs, second.surface.coeffs)
    assert not grid.x_factor.upper.flags.writeable
    assert not grid.y_factor.upper.flags.writeable


def test_cached_and_reference_rectangular_refinement_agree_at_periodic_seam() -> None:
    latitudes = np.linspace(-60.0, 60.0, 25, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 10.0, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    target_latitude = 11.5
    target_longitude = 359.0
    longitude_distance = np.minimum(
        np.abs(lon_mesh - target_longitude),
        360.0 - np.abs(lon_mesh - target_longitude),
    )
    frame = (lat_mesh - target_latitude) ** 2 + longitude_distance**2
    grid = prepare_rectangular_grid(latitudes, longitudes, periodic_x=True)
    reference = build_bspline_surface_reference(
        frame, latitudes, longitudes, periodic_x=True, grid=grid
    )
    cached = build_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True, grid=grid
    )
    assert reference.surface is not None
    assert cached.surface is not None

    reference_result = refine_bspline_feature_point(
        reference.surface,
        target_latitude,
        0.0,
        is_minimum=True,
        initial_value=float(frame[12, 0]),
        optimization_scale=0.01,
        max_iterations=100,
    )
    cached_result = refine_bspline_feature_point(
        cached.surface,
        target_latitude,
        0.0,
        is_minimum=True,
        initial_value=float(frame[12, 0]),
        optimization_scale=0.01,
        max_iterations=100,
    )
    assert reference_result.status == cached_result.status == "success"
    assert cached_result.latitude == pytest.approx(
        reference_result.latitude, abs=1.0e-9
    )
    assert cached_result.longitude == pytest.approx(
        reference_result.longitude, abs=1.0e-9
    )
    assert cached_result.value == pytest.approx(reference_result.value, abs=1.0e-8)


def test_cached_rectangular_values_derivatives_and_candidates_match_reference() -> None:
    latitudes = np.linspace(-60.0, 60.0, 25, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 15.0, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)
    first_distance = np.minimum(
        np.abs(lon_mesh - 25.0), 360.0 - np.abs(lon_mesh - 25.0)
    )
    second_distance = np.minimum(
        np.abs(lon_mesh - 250.0), 360.0 - np.abs(lon_mesh - 250.0)
    )
    frame = -200.0 * np.exp(
        -(((lat_mesh - 15.0) / 12.0) ** 2) - (first_distance / 25.0) ** 2
    ) - 150.0 * np.exp(
        -(((lat_mesh + 25.0) / 16.0) ** 2) - (second_distance / 30.0) ** 2
    )
    grid = prepare_rectangular_grid(latitudes, longitudes, periodic_x=True)
    reference = build_bspline_surface_reference(
        frame, latitudes, longitudes, periodic_x=True, grid=grid
    )
    cached = build_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True, grid=grid
    )
    assert reference.surface is not None
    assert cached.surface is not None

    points = np.asarray(
        [
            [cached.surface.x_lower, cached.surface.y_lower],
            [cached.surface.x_upper, cached.surface.y_lower],
            [cached.surface.x_lower, cached.surface.y_upper],
            [cached.surface.x_upper, cached.surface.y_upper],
            [25.0, 15.0],
            [250.0, -25.0],
        ],
        dtype=np.float64,
    )
    cached_evaluations = np.asarray(
        [
            _eval_bspline_2d(
                cached.surface.x_knots,
                cached.surface.y_knots,
                cached.surface.coeffs,
                float(longitude),
                float(latitude),
            )
            for longitude, latitude in points
        ]
    )
    reference_evaluations = np.asarray(
        [
            _eval_bspline_2d(
                reference.surface.x_knots,
                reference.surface.y_knots,
                reference.surface.coeffs,
                float(longitude),
                float(latitude),
            )
            for longitude, latitude in points
        ]
    )
    np.testing.assert_allclose(
        cached_evaluations,
        reference_evaluations,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    cached_candidates = _detect_track_rectangular_candidates(
        frame,
        latitudes,
        longitudes,
        intensity_threshold=-1.0,
        is_min=True,
        min_grid_points=3,
        grid=grid,
    )
    reference_candidates = _detect_track_rectangular_candidates(
        frame,
        latitudes,
        longitudes,
        intensity_threshold=-1.0,
        is_min=True,
        min_grid_points=3,
        grid=grid,
    )
    for cached_values, reference_values in zip(
        cached_candidates, reference_candidates, strict=True
    ):
        np.testing.assert_array_equal(cached_values, reference_values)

    for latitude, longitude, value in zip(
        cached_candidates[0],
        cached_candidates[1],
        cached_candidates[2],
        strict=True,
    ):
        cached_result = refine_bspline_feature_point(
            cached.surface,
            float(latitude),
            float(longitude),
            is_minimum=True,
            initial_value=float(value),
            optimization_scale=0.01,
            max_iterations=100,
        )
        reference_result = refine_bspline_feature_point(
            reference.surface,
            float(latitude),
            float(longitude),
            is_minimum=True,
            initial_value=float(value),
            optimization_scale=0.01,
            max_iterations=100,
        )
        assert cached_result.status == reference_result.status
        assert cached_result.latitude == pytest.approx(
            reference_result.latitude, abs=1.0e-9
        )
        assert cached_result.longitude == pytest.approx(
            reference_result.longitude, abs=1.0e-9
        )
        assert cached_result.value == pytest.approx(reference_result.value, abs=1.0e-8)


def test_cached_rectangular_signed_and_unsigned_longitudes_are_equivalent() -> None:
    latitudes = np.linspace(-60.0, 60.0, 17, dtype=np.float64)
    unsigned = np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float64)
    signed = (unsigned + 180.0) % 360.0 - 180.0
    signed_order = np.argsort(signed)
    signed = signed[signed_order]

    def values(longitude: NDArray[np.float64]) -> NDArray[np.float64]:
        physical_longitude = (longitude + 180.0) % 360.0 - 180.0
        return (physical_longitude - 17.0) ** 2

    frame_unsigned = (latitudes[:, None] - 8.0) ** 2 + values(unsigned)[None, :]
    frame_signed = (latitudes[:, None] - 8.0) ** 2 + values(signed)[None, :]
    unsigned_grid = prepare_rectangular_grid(latitudes, unsigned, periodic_x=True)
    signed_grid = prepare_rectangular_grid(latitudes, signed, periodic_x=True)

    unsigned_surface = build_bspline_surface(
        frame_unsigned,
        latitudes,
        unsigned,
        periodic_x=True,
        grid=unsigned_grid,
    )
    signed_surface = build_bspline_surface(
        frame_signed,
        latitudes,
        signed,
        periodic_x=True,
        grid=signed_grid,
    )
    assert unsigned_surface.surface is not None
    assert signed_surface.surface is not None
    np.testing.assert_array_equal(
        unsigned_surface.surface.x_knots, signed_surface.surface.x_knots
    )
    np.testing.assert_allclose(
        unsigned_surface.surface.coeffs,
        signed_surface.surface.coeffs,
        rtol=5.0e-13,
        atol=5.0e-12,
    )


def test_rectangular_spline_optimization_scale_preserves_field_value_units() -> None:
    latitudes = np.linspace(-60.0, 60.0, 13)
    longitudes = np.linspace(0.0, 360.0, 25, endpoint=False)
    frame = (latitudes[:, None] - 12.5) ** 2 + (longitudes[None, :] - 152.0) ** 2
    built = build_bspline_surface(frame, latitudes, longitudes, periodic_x=True)

    assert built.surface is not None
    refined = refine_bspline_feature_point(
        built.surface,
        10.0,
        158.4,
        is_minimum=True,
        initial_value=float(frame[7, 11]),
        optimization_scale=0.01,
        max_iterations=100,
    )

    assert refined.status == "success"
    assert refined.latitude == pytest.approx(12.5, abs=1.0e-3)
    assert refined.longitude == pytest.approx(152.0, abs=1.0e-3)
    assert refined.value == pytest.approx(0.0, abs=5.0e-5)


@pytest.mark.parametrize("optimization_scale", [0.0, -1.0, np.inf])
def test_rectangular_spline_rejects_invalid_optimization_scale(
    optimization_scale: float,
) -> None:
    latitudes = np.linspace(-30.0, 30.0, 5)
    longitudes = np.linspace(0.0, 360.0, 8, endpoint=False)
    built = build_bspline_surface(
        latitudes[:, None] ** 2 + longitudes[None, :] ** 2,
        latitudes,
        longitudes,
        periodic_x=True,
    )

    assert built.surface is not None
    refined = refine_bspline_feature_point(
        built.surface,
        0.0,
        0.0,
        is_minimum=True,
        optimization_scale=optimization_scale,
    )

    assert refined.status == "invalid_neighborhood"


def test_spherical_spline_periodic_seam_invariance() -> None:
    latitudes = np.linspace(-60.0, 60.0, 31, dtype=np.float64)
    longitudes = np.arange(360.0, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

    # Minimum near the seam at longitude 359.5 / -0.5
    target_lat = 10.0
    target_lon = 359.5
    dlon = np.minimum(
        np.abs(lon_mesh - target_lon),
        np.abs(lon_mesh - target_lon + 360.0),
    )
    dlon = np.minimum(dlon, np.abs(lon_mesh - target_lon - 360.0))
    frame = (lat_mesh - target_lat) ** 2 + dlon**2

    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    assert built.status == "success"

    # Refine starting from 0.0 vs starting from 359.0
    r1 = refine_spherical_bspline_feature_point(
        built.surface,
        10.0,
        0.0,
        is_minimum=True,
    )
    r2 = refine_spherical_bspline_feature_point(
        built.surface,
        10.0,
        359.0,
        is_minimum=True,
    )

    assert r1.status == "success"
    assert r2.status == "success"
    assert r1.latitude == pytest.approx(r2.latitude, abs=1.0e-3)
    assert r1.longitude == pytest.approx(r2.longitude, abs=1.0e-3)
    assert r1.value == pytest.approx(r2.value, abs=1.0e-3)


def test_numba_bspline_eval_direct_parity_against_scipy() -> None:
    """Verify Numba B-spline kernel produces direct numerical parity with SciPy."""
    from scipy.interpolate import RectBivariateSpline

    from pystormtracker.refinement.bspline import (
        _eval_bspline_2d,
        _eval_bspline_val,
    )

    latitudes = np.linspace(-60.0, 60.0, 15, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 25, endpoint=False, dtype=np.float64)
    frame = np.sin(np.deg2rad(latitudes[:, None])) * np.cos(
        np.deg2rad(longitudes[None, :])
    )

    built = build_bspline_surface(frame, latitudes, longitudes, periodic_x=True)
    assert built.surface is not None
    surface = built.surface

    # Construct local SciPy oracle directly
    extended_x = np.concatenate((longitudes, [longitudes[0] + 360.0]))
    extended_z = np.concatenate((frame, frame[:, :1]), axis=1)
    oracle = RectBivariateSpline(extended_x, latitudes, extended_z.T, kx=3, ky=3, s=0.0)

    test_lats = np.linspace(-55.0, 55.0, 10)
    test_lons = np.linspace(5.0, 355.0, 10)

    for lat_val in test_lats:
        for lon_val in test_lons:
            val_numba, dx_numba, dy_numba = _eval_bspline_2d(
                surface.x_knots, surface.y_knots, surface.coeffs, lon_val, lat_val
            )
            val_only = _eval_bspline_val(
                surface.x_knots, surface.y_knots, surface.coeffs, lon_val, lat_val
            )
            val_scipy = float(
                np.asarray(oracle(lon_val, lat_val, grid=False)).reshape(-1)[0]
            )
            dx_scipy = float(
                np.asarray(oracle(lon_val, lat_val, dx=1, dy=0, grid=False)).reshape(
                    -1
                )[0]
            )
            dy_scipy = float(
                np.asarray(oracle(lon_val, lat_val, dx=0, dy=1, grid=False)).reshape(
                    -1
                )[0]
            )

            assert val_numba == pytest.approx(val_scipy, rel=1e-12, abs=1e-12)
            assert val_only == pytest.approx(val_scipy, rel=1e-12, abs=1e-12)
            assert dx_numba == pytest.approx(dx_scipy, rel=1e-11, abs=1e-11)
            assert dy_numba == pytest.approx(dy_scipy, rel=1e-11, abs=1e-11)


def test_numba_rect_sphere_bspline_eval_direct_parity_against_scipy() -> None:
    """Verify Numba B-spline kernel parity with SciPy RectSphereBivariateSpline."""
    from scipy.interpolate import RectSphereBivariateSpline

    from pystormtracker.refinement.bspline import (
        _eval_bspline_2d,
        _eval_bspline_val,
    )

    latitudes = np.linspace(-75.0, 75.0, 16, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 32, endpoint=False, dtype=np.float64)
    lon_2d, lat_2d = np.meshgrid(longitudes, latitudes)
    frame = np.sin(np.deg2rad(lat_2d)) * np.cos(np.deg2rad(lon_2d)) + np.cos(
        3.0 * np.deg2rad(lat_2d)
    )

    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    surface = built.surface

    # Construct local SciPy spherical oracle directly
    theta = np.deg2rad(90.0 - latitudes)
    theta_order = np.argsort(theta)
    clamped_theta = np.clip(theta[theta_order], 1.0e-7, np.pi - 1.0e-7)
    phi = np.deg2rad(longitudes) % (2.0 * np.pi)
    phi_order = np.argsort(phi)
    ordered_frame = frame[theta_order, :][:, phi_order]
    oracle = RectSphereBivariateSpline(
        clamped_theta,
        phi[phi_order],
        ordered_frame,
        s=0.0,
        pole_continuity=False,
        pole_values=None,
        pole_exact=False,
    )

    test_thetas = np.deg2rad(np.linspace(20.0, 160.0, 10))
    test_phis = np.deg2rad(np.linspace(10.0, 350.0, 10))

    for th_val in test_thetas:
        for ph_val in test_phis:
            val_numba, dth_numba, dph_numba = _eval_bspline_2d(
                surface.theta_knots,
                surface.phi_knots,
                surface.coeffs,
                th_val,
                ph_val,
            )
            val_only = _eval_bspline_val(
                surface.theta_knots,
                surface.phi_knots,
                surface.coeffs,
                th_val,
                ph_val,
            )
            val_scipy = float(np.asarray(oracle.ev(th_val, ph_val)).item())
            dth_scipy = float(
                np.asarray(oracle.ev(th_val, ph_val, dtheta=1, dphi=0)).item()
            )
            dph_scipy = float(
                np.asarray(oracle.ev(th_val, ph_val, dtheta=0, dphi=1)).item()
            )

            assert val_numba == pytest.approx(val_scipy, rel=1e-12, abs=1e-12)
            assert val_only == pytest.approx(val_scipy, rel=1e-12, abs=1e-12)
            assert dth_numba == pytest.approx(dth_scipy, rel=1e-11, abs=1e-11)
            assert dph_numba == pytest.approx(dph_scipy, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("scale_factor", [1.0e-3, 1.0e-1, 1.0, 10.0, 1000.0])
def test_spherical_spline_scale_invariance(scale_factor: float) -> None:
    """Verify spherical optimizer is scale invariant across 6 orders of magnitude."""
    latitudes = np.linspace(-70.0, 70.0, 21, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 10.0, dtype=np.float64)
    theta = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi = np.deg2rad(longitudes)[np.newaxis, :]

    target_lat = 25.4
    target_lon = 143.7
    target_theta = np.deg2rad(90.0 - target_lat)
    target_phi = np.deg2rad(target_lon)

    # Base peak
    base_frame = _spherical_peak(theta, phi, target_theta, target_phi, 30.0)

    # Unscaled build
    built_base = build_spherical_bspline_surface(
        base_frame, latitudes, longitudes, periodic_x=True
    )
    assert built_base.surface is not None
    res_base = refine_spherical_bspline_feature_point(
        built_base.surface,
        20.0,
        140.0,
        is_minimum=False,
    )
    assert res_base.status == "success"

    # Scaled build
    scaled_frame = base_frame * scale_factor
    built_scaled = build_spherical_bspline_surface(
        scaled_frame, latitudes, longitudes, periodic_x=True
    )
    assert built_scaled.surface is not None
    res_scaled = refine_spherical_bspline_feature_point(
        built_scaled.surface,
        20.0,
        140.0,
        is_minimum=False,
    )
    assert res_scaled.status == "success"

    # Coordinates must match to high precision
    assert res_scaled.latitude == pytest.approx(res_base.latitude, abs=1.0e-2)
    assert res_scaled.longitude == pytest.approx(res_base.longitude, abs=1.0e-2)
    assert res_scaled.value == pytest.approx(res_base.value * scale_factor, rel=1.0e-4)


def test_spherical_spline_longitude_origin_invariance() -> None:
    """Verify shifting longitudes produces invariant spherical coordinates."""
    latitudes = np.linspace(-60.0, 60.0, 25, dtype=np.float64)
    longitudes_1 = np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float64)
    longitudes_2 = (longitudes_1 + 90.0) % 360.0
    sort_idx = np.argsort(longitudes_2)
    longitudes_2_sorted = longitudes_2[sort_idx]

    theta_1 = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi_1 = np.deg2rad(longitudes_1)[np.newaxis, :]
    frame_1 = _spherical_peak(theta_1, phi_1, np.deg2rad(70.0), np.deg2rad(45.0), 30.0)

    theta_2 = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi_2 = np.deg2rad(longitudes_2_sorted)[np.newaxis, :]
    frame_2 = _spherical_peak(theta_2, phi_2, np.deg2rad(70.0), np.deg2rad(45.0), 30.0)

    built_1 = build_spherical_bspline_surface(
        frame_1, latitudes, longitudes_1, periodic_x=True
    )
    built_2 = build_spherical_bspline_surface(
        frame_2, latitudes, longitudes_2_sorted, periodic_x=True
    )
    assert built_1.surface is not None
    assert built_2.surface is not None

    r1 = refine_spherical_bspline_feature_point(
        built_1.surface, 20.0, 40.0, is_minimum=False
    )
    r2 = refine_spherical_bspline_feature_point(
        built_2.surface, 20.0, 40.0, is_minimum=False
    )

    assert r1.status == "success"
    assert r2.status == "success"
    assert r1.latitude == pytest.approx(r2.latitude, abs=1.0e-3)
    assert r1.longitude == pytest.approx(r2.longitude, abs=1.0e-3)
    assert r1.value == pytest.approx(r2.value, abs=1.0e-3)


def test_sphere_parallel_transport_and_basis_properties() -> None:
    """Verify S^2 parallel transport tangency, isometry, and SPD preservation."""
    from pystormtracker.preprocessing.spherical_geometry import sphere_point_and_basis
    from pystormtracker.refinement.bspline import (
        _parallel_transport_sphere,
    )

    rng = np.random.default_rng(12345)
    for _ in range(50):
        theta0 = float(rng.uniform(0.1, np.pi - 0.1))
        phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
        r0, eth0, eph0 = sphere_point_and_basis(theta0, phi0)

        # Tangent search direction at r0
        p2d = rng.normal(size=2).astype(np.float64)
        p2d /= np.linalg.norm(p2d)
        v0 = p2d[0] * eth0 + p2d[1] * eph0
        u0 = v0 / np.linalg.norm(v0)

        # Geodesic step
        alpha = float(rng.uniform(0.01, 0.08))
        r1 = np.cos(alpha) * r0 + np.sin(alpha) * u0
        u1 = -np.sin(alpha) * r0 + np.cos(alpha) * u0

        rz = np.clip(r1[2], -1.0, 1.0)
        theta1 = float(np.arccos(rz))
        phi1 = float(np.arctan2(r1[1], r1[0]) % (2.0 * np.pi))
        _, eth1, eph1 = sphere_point_and_basis(theta1, phi1)

        # Transport basis
        eth0_tr = _parallel_transport_sphere(eth0, u0, u1)
        eph0_tr = _parallel_transport_sphere(eph0, u0, u1)

        # A. Tangency at r1
        assert abs(np.dot(eth0_tr, r1)) < 1.0e-13
        assert abs(np.dot(eph0_tr, r1)) < 1.0e-13

        # B. Norm preservation
        assert np.linalg.norm(eth0_tr) == pytest.approx(1.0, abs=1.0e-13)
        assert np.linalg.norm(eph0_tr) == pytest.approx(1.0, abs=1.0e-13)

        # C. Inner-product preservation (orthogonality of transported basis)
        assert abs(np.dot(eth0_tr, eph0_tr)) < 1.0e-13

        # D. Round trip along reversed geodesic
        eth0_rec = _parallel_transport_sphere(eth0_tr, -u1, -u0)
        assert np.allclose(eth0_rec, eth0, atol=1.0e-13)

        # E. Q change-of-basis matrix orthogonality
        q_mat = np.array(
            [
                [np.dot(eth1, eth0_tr), np.dot(eth1, eph0_tr)],
                [np.dot(eph1, eth0_tr), np.dot(eph1, eph0_tr)],
            ],
            dtype=np.float64,
        )
        assert np.allclose(q_mat.T @ q_mat, np.eye(2), atol=1.0e-13)
        assert np.allclose(q_mat @ q_mat.T, np.eye(2), atol=1.0e-13)

        # F. Symmetric positive-definite inverse-Hessian transport
        rnd_mat = rng.normal(size=(2, 2)).astype(np.float64)
        h_orig = rnd_mat.T @ rnd_mat + np.eye(2)
        h_tr = q_mat @ h_orig @ q_mat.T
        assert np.allclose(h_tr, h_tr.T, atol=1.0e-13)
        orig_eigs = np.sort(np.linalg.eigvalsh(h_orig))
        tr_eigs = np.sort(np.linalg.eigvalsh(h_tr))
        assert np.allclose(orig_eigs, tr_eigs, atol=1.0e-13)


def test_spherical_optimizer_max_iterations_returns_no_convergence() -> None:
    """Verify exhausting max_iterations returns status 2, not 0."""
    from pystormtracker.refinement.bspline import (
        _eval_bspline_2d,
        _spherical_geodesic_gdfp_search,
    )

    latitudes = np.linspace(-60.0, 60.0, 25, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 36, endpoint=False, dtype=np.float64)
    theta = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi = np.deg2rad(longitudes)[np.newaxis, :]
    frame = _spherical_peak(theta, phi, np.deg2rad(70.0), np.deg2rad(45.0), 30.0)

    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    s = built.surface

    # Choose a starting point away from the peak with non-zero gradient
    th_0 = float(np.deg2rad(50.0))
    ph_0 = float(np.deg2rad(20.0))

    # Verify starting point has non-negligible gradient
    _, dth, dph = _eval_bspline_2d(s.theta_knots, s.phi_knots, s.coeffs, th_0, ph_0)
    assert np.sqrt(dth**2 + dph**2) > 1.0e-3

    # Run with max_iterations=0 or 1 with tight gradient tolerance so it cannot converge
    status, _, _, _ = _spherical_geodesic_gdfp_search(
        theta_0=th_0,
        phi_0=ph_0,
        sign=-1.0,
        tx=s.theta_knots,
        ty=s.phi_knots,
        c=s.coeffs,
        max_iterations=1,
        gradient_tolerance=1.0e-12,
    )
    assert status == 2  # optimizer_no_convergence


def test_spherical_localization_boundary_contains_every_geodesic_trial() -> None:
    """The first boundary intersection keeps the complete trial segment local."""
    from pystormtracker.preprocessing.spherical_geometry import (
        inside_spherical_localization_region,
        sphere_point_and_basis,
    )
    from pystormtracker.refinement.bspline import (
        _spherical_localization_boundary_distance,
        _spherical_localization_region,
    )

    latitudes = np.linspace(-60.0, 60.0, 25, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 5.0, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    frame = (lat_grid + 10.0) ** 2 + ((lon_grid - 45.0 + 180.0) % 360.0 - 180.0) ** 2
    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    region = _spherical_localization_region(built.surface, -10.0, 45.0, 3)
    assert region is not None

    point, e_theta, e_phi = sphere_point_and_basis(np.deg2rad(100.0), np.deg2rad(45.0))
    direction = e_theta + 0.4 * e_phi
    direction /= np.linalg.norm(direction)
    boundary = _spherical_localization_boundary_distance(
        point,
        direction,
        region.origin,
        region.origin_e_theta,
        region.origin_e_phi,
        region.theta_half_width,
        region.phi_half_width,
    )

    assert boundary > 0.0
    for step in np.linspace(0.0, boundary, 21):
        trial = np.cos(step) * point + np.sin(step) * direction
        assert inside_spherical_localization_region(
            trial,
            region.origin,
            region.origin_e_theta,
            region.origin_e_phi,
            region.theta_half_width,
            region.phi_half_width,
        )
    outside = np.cos(boundary + 1.0e-5) * point + np.sin(boundary + 1.0e-5) * direction
    assert not inside_spherical_localization_region(
        outside,
        region.origin,
        region.origin_e_theta,
        region.origin_e_phi,
        region.theta_half_width,
        region.phi_half_width,
    )


def test_spherical_refinement_stays_in_original_two_basin_neighbourhood() -> None:
    """A deeper remote spline minimum cannot capture a local feature centre."""
    latitudes = np.linspace(-60.0, 60.0, 49, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 5.0, dtype=np.float64)
    theta = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi = np.deg2rad(longitudes)[np.newaxis, :]

    def spherical_bowl(
        latitude: float, longitude: float, amplitude: float, sharpness: float
    ) -> NDArray[np.float64]:
        target_theta = np.deg2rad(90.0 - latitude)
        target_phi = np.deg2rad(longitude)
        dot = np.cos(theta) * np.cos(target_theta) + np.sin(theta) * np.sin(
            target_theta
        ) * np.cos(phi - target_phi)
        return np.asarray(
            -amplitude * np.exp(sharpness * (dot - 1.0)), dtype=np.float64
        )

    frame = spherical_bowl(-20.0, 50.0, 1.0, 150.0) + spherical_bowl(
        25.0, 200.0, 5.0, 120.0
    )
    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    refined = refine_spherical_bspline_feature_point(
        built.surface,
        -17.5,
        50.0,
        is_minimum=True,
    )

    assert refined.status == "success"
    assert refined.latitude == pytest.approx(-20.0, abs=1.0e-3)
    assert refined.longitude == pytest.approx(50.0, abs=1.0e-3)


def test_spherical_success_requires_intrinsic_stationarity() -> None:
    """A successful local refinement satisfies the configured gradient test."""
    from pystormtracker.preprocessing.spherical_geometry import sphere_point_and_basis
    from pystormtracker.refinement.bspline import (
        _eval_bspline_val,
        _eval_spherical_intrinsic,
    )

    latitudes = np.linspace(-60.0, 60.0, 31, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 5.0, dtype=np.float64)
    theta = np.deg2rad(90.0 - latitudes)[:, np.newaxis]
    phi = np.deg2rad(longitudes)[np.newaxis, :]
    frame = _spherical_peak(
        theta,
        phi,
        np.deg2rad(75.3),
        np.deg2rad(132.4),
        25.0,
    )
    built = build_spherical_bspline_surface(
        frame, latitudes, longitudes, periodic_x=True
    )
    assert built.surface is not None
    surface = built.surface
    initial_latitude = 13.0
    initial_longitude = 135.0
    tolerance = 1.0e-5
    refined = refine_spherical_bspline_feature_point(
        surface,
        initial_latitude,
        initial_longitude,
        is_minimum=False,
        gradient_tolerance=tolerance,
    )

    assert refined.status == "success"
    point, _e_theta, _e_phi = sphere_point_and_basis(
        np.deg2rad(90.0 - refined.latitude), np.deg2rad(refined.longitude)
    )
    _value, g_theta, g_phi, _gradient, _theta, _phi = _eval_spherical_intrinsic(
        surface.theta_knots,
        surface.phi_knots,
        surface.coeffs,
        -1.0,
        point,
    )
    scale = abs(
        _eval_bspline_val(
            surface.theta_knots,
            surface.phi_knots,
            surface.coeffs,
            np.deg2rad(90.0 - initial_latitude),
            np.deg2rad(initial_longitude),
        )
    )
    assert np.hypot(g_theta, g_phi) / max(scale, 1.0e-12) < tolerance
