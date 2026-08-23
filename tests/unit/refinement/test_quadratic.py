from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.refinement import (
    refine_quadratic_feature_coordinates,
    refine_quadratic_feature_point,
    refine_quadratic_feature_points,
    refine_spherical_quadratic_feature_points,
    spherical_quadratic_status_name,
)


def test_refine_quadratic_feature_point_preserves_negative_longitudes() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([-180.0, -179.0, -178.0])

    refined_lat, refined_lon, refined_value = refine_quadratic_feature_point(
        frame, 1, 1, lat, lon, periodic_x=True
    )

    assert refined_lat == 11.0
    assert refined_lon == pytest.approx(-179.1)
    assert refined_value > frame[1, 1]


def test_refine_quadratic_feature_point_keeps_projected_x_nonperiodic() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    y = np.array([-100.0, 0.0, 100.0])
    x = np.array([-100.0, 0.0, 100.0])

    refined_y, refined_x, refined_value = refine_quadratic_feature_point(
        frame, 1, 1, y, x, periodic_x=False
    )

    assert refined_y == 0.0
    assert refined_x == pytest.approx(-10.0)
    assert refined_value > frame[1, 1]


def test_refine_quadratic_feature_point_does_not_wrap_projected_boundary() -> None:
    frame = np.ones((3, 3), dtype=np.float64)
    y = np.array([-100.0, 0.0, 100.0])
    x = np.array([-100.0, 0.0, 100.0])

    refined = refine_quadratic_feature_point(frame, 1, 0, y, x, periodic_x=False)

    assert refined == (0.0, -100.0, 1.0)


def test_batch_refine_quadratic_feature_points() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([-180.0, -179.0, -178.0])

    q_lats, q_lons, q_vals = refine_quadratic_feature_points(
        frame,
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
        lat,
        lon,
        periodic_x=True,
    )
    assert q_lats[0] == 11.0
    assert q_lons[0] == pytest.approx(-179.1)
    assert q_vals[0] > frame[1, 1]


def test_batch_refine_quadratic_feature_coordinates() -> None:
    frame = np.array([[0.0, 0.5, 0.0], [0.6, 1.0, 0.4], [0.0, 0.5, 0.0]])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([-180.0, -179.0, -178.0])

    q_lats, q_lons, q_vals = refine_quadratic_feature_coordinates(
        frame,
        np.array([11.0], dtype=np.float64),
        np.array([-179.0], dtype=np.float64),
        lat,
        lon,
        periodic_x=True,
    )
    assert q_lats[0] == 11.0
    assert q_lons[0] == pytest.approx(-179.1)
    assert q_vals[0] > frame[1, 1]


def _spherical_quadratic_stencil(
    *,
    center_latitude: float = 45.0,
    target: NDArray[np.float64] | None = None,
    hessian: NDArray[np.float64] | None = None,
    basis_rotation_degrees: float = 0.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build an exact local tangent-quadratic eight-neighbour stencil."""
    latitudes = center_latitude + np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    longitudes = np.arange(0.0, 360.0, 5.0)
    row = 2
    column = 0
    theta = np.deg2rad(90.0 - latitudes[row])
    phi = np.deg2rad(longitudes[column])
    center = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    e_theta = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    a = np.deg2rad(5.0)
    b = np.sin(theta) * np.deg2rad(5.0)
    target = np.array([0.25, -0.40], dtype=np.float64) if target is None else target
    hessian = (
        np.array([[-3.0, -0.4], [-0.4, -2.0]], dtype=np.float64)
        if hessian is None
        else hessian
    )
    angle = np.deg2rad(basis_rotation_degrees)
    basis_rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    gradient = -hessian @ target
    physical_target = basis_rotation.T @ target
    frame = np.zeros((latitudes.size, longitudes.size), dtype=np.float64)

    for row_offset, column_offset in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        neighbor_row = row + row_offset
        neighbor_column = (column + column_offset) % longitudes.size
        neighbor_theta = np.deg2rad(90.0 - latitudes[neighbor_row])
        neighbor_phi = np.deg2rad(longitudes[neighbor_column])
        neighbor = np.array(
            [
                np.sin(neighbor_theta) * np.cos(neighbor_phi),
                np.sin(neighbor_theta) * np.sin(neighbor_phi),
                np.cos(neighbor_theta),
            ]
        )
        dot = float(np.clip(np.dot(neighbor, center), -1.0, 1.0))
        alpha = np.arccos(dot)
        tangent = alpha / np.sin(alpha) * (neighbor - dot * center)
        coordinates = np.array(
            [np.dot(tangent, e_theta) / a, np.dot(tangent, e_phi) / b]
        )
        rotated_coordinates = basis_rotation @ coordinates
        frame[neighbor_row, neighbor_column] = (
            gradient @ rotated_coordinates
            + 0.5 * rotated_coordinates @ hessian @ rotated_coordinates
        )

    eta = a * physical_target[0] * e_theta + b * physical_target[1] * e_phi
    eta_norm = np.linalg.norm(eta)
    target_point = np.cos(eta_norm) * center + np.sin(eta_norm) / eta_norm * eta
    expected = np.array(
        [
            np.rad2deg(np.arcsin(target_point[2])),
            np.rad2deg(np.arctan2(target_point[1], target_point[0])) % 360.0,
        ]
    )
    return frame, latitudes, longitudes, physical_target, expected


def test_spherical_quadratic_refines_in_exact_tangent_coordinates() -> None:
    frame, latitudes, longitudes, target, expected = _spherical_quadratic_stencil()

    refined = refine_spherical_quadratic_feature_points(
        frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=False,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == "success"
    assert refined.latitudes[0] == pytest.approx(expected[0], abs=1.0e-10)
    assert refined.longitudes[0] == pytest.approx(expected[1], abs=1.0e-10)
    np.testing.assert_allclose(refined.normalized_displacements[0], target)
    np.testing.assert_allclose(
        refined.hessian_eigenvalues[0], np.linalg.eigvalsh([[-3.0, -0.4], [-0.4, -2.0]])
    )


def test_spherical_quadratic_reports_rejection_and_preserves_grid_fallback() -> None:
    frame, latitudes, longitudes, _target, _expected = _spherical_quadratic_stencil()
    frame[1, -1] = np.nan
    refined = refine_spherical_quadratic_feature_points(
        frame,
        np.array([0, 2, 2], dtype=np.int64),
        np.array([0, 0, 0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=False,
    )

    statuses = [
        spherical_quadratic_status_name(int(code)) for code in refined.status_codes
    ]
    assert statuses == [
        "invalid_neighborhood",
        "nonfinite_failure",
        "nonfinite_failure",
    ]
    assert refined.latitudes.tolist() == pytest.approx([35.0, 45.0, 45.0])
    assert refined.longitudes.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert refined.values[0] == frame[0, 0]


def test_spherical_quadratic_rejects_the_wrong_curvature() -> None:
    frame, latitudes, longitudes, _target, _expected = _spherical_quadratic_stencil()
    refined = refine_spherical_quadratic_feature_points(
        frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=True,
    )

    assert (
        spherical_quadratic_status_name(int(refined.status_codes[0]))
        == "wrong_curvature"
    )
    assert refined.latitudes[0] == 45.0
    assert refined.longitudes[0] == 0.0


def test_spherical_quadratic_refines_a_high_latitude_minimum() -> None:
    frame, latitudes, longitudes, target, expected = _spherical_quadratic_stencil(
        center_latitude=75.0,
        hessian=np.array([[2.5, 0.6], [0.6, 1.7]], dtype=np.float64),
    )

    refined = refine_spherical_quadratic_feature_points(
        frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=True,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == "success"
    assert refined.latitudes[0] == pytest.approx(expected[0], abs=1.0e-10)
    assert refined.longitudes[0] == pytest.approx(expected[1], abs=1.0e-10)
    np.testing.assert_allclose(refined.normalized_displacements[0], target)


def test_spherical_quadratic_recovers_a_rotated_tangent_model() -> None:
    frame, latitudes, longitudes, target, expected = _spherical_quadratic_stencil(
        basis_rotation_degrees=37.0
    )

    refined = refine_spherical_quadratic_feature_points(
        frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=False,
    )

    assert spherical_quadratic_status_name(int(refined.status_codes[0])) == "success"
    assert refined.latitudes[0] == pytest.approx(expected[0], abs=1.0e-10)
    assert refined.longitudes[0] == pytest.approx(expected[1], abs=1.0e-10)
    np.testing.assert_allclose(refined.normalized_displacements[0], target)


def test_spherical_quadratic_is_invariant_to_signed_longitude_representation() -> None:
    frame, latitudes, longitudes, _target, _expected = _spherical_quadratic_stencil(
        center_latitude=75.0
    )
    canonical = refine_spherical_quadratic_feature_points(
        frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=False,
    )
    signed = np.where(longitudes >= 180.0, longitudes - 360.0, longitudes)
    order = np.argsort(signed)
    signed_longitudes = signed[order]
    signed_frame = frame[:, order]
    signed_column = int(np.flatnonzero(order == 0)[0])
    signed_result = refine_spherical_quadratic_feature_points(
        signed_frame,
        np.array([2], dtype=np.int64),
        np.array([signed_column], dtype=np.int64),
        latitudes,
        signed_longitudes,
        is_minimum=False,
    )

    assert canonical.status_codes.tolist() == signed_result.status_codes.tolist()
    assert canonical.latitudes[0] == pytest.approx(
        signed_result.latitudes[0], abs=1e-10
    )
    longitude_difference = (
        canonical.longitudes[0] - signed_result.longitudes[0] + 180.0
    ) % 360.0 - 180.0
    assert longitude_difference == pytest.approx(0.0, abs=1e-10)


def test_spherical_quadratic_rejects_saddle_and_nonlocal_stationary_points() -> None:
    saddle_frame, latitudes, longitudes, _target, _expected = (
        _spherical_quadratic_stencil(
            hessian=np.array([[2.0, 0.3], [0.3, -1.0]], dtype=np.float64)
        )
    )
    outside_frame, _latitudes, _longitudes, _target, _expected = (
        _spherical_quadratic_stencil(
            target=np.array([1.2, -0.4], dtype=np.float64),
            hessian=np.array([[2.0, 0.2], [0.2, 1.5]], dtype=np.float64),
        )
    )
    saddle = refine_spherical_quadratic_feature_points(
        saddle_frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=True,
    )
    outside = refine_spherical_quadratic_feature_points(
        outside_frame,
        np.array([2], dtype=np.int64),
        np.array([0], dtype=np.int64),
        latitudes,
        longitudes,
        is_minimum=True,
    )

    assert (
        spherical_quadratic_status_name(int(saddle.status_codes[0]))
        == "wrong_curvature"
    )
    assert (
        spherical_quadratic_status_name(int(outside.status_codes[0]))
        == "outside_locality"
    )
