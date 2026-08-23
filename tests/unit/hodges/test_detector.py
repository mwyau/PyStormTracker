from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.hodges.detector import (
    DUFF_FEATURE_CUTOFF,
    HodgesDetector,
    _compute_object_properties,
    _detect_track_rectangular_candidates,
    _detect_track_rectangular_candidates_reference,
    _find_object_extrema,
    _find_object_first_indices,
    _group_object_extrema,
    _label_connected_components,
    detect_hodges_frame,
)
from pystormtracker.hodges.tracker import HodgesFeatureRefinement, HodgesTracker
from pystormtracker.io.data_loader import DataLoader
from pystormtracker.refinement.bspline import BsplineRefinementResult


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Clear the DataLoader cache before each test."""
    DataLoader._ds_cache.clear()


def test_hodges_default_feature_refinement_is_bspline() -> None:
    """The documented scientific Hodges default remains explicit in code."""
    assert HodgesTracker().feature_refinement == "bspline"


def test_spherical_quadratic_uses_a_fixed_three_by_three_neighborhood() -> None:
    with pytest.raises(ValueError, match="fixed 3 by 3 grid neighborhood"):
        HodgesDetector.from_xarray(
            xr.DataArray(np.zeros((1, 5, 5)), dims=("time", "latitude", "longitude")),
            variable_name="msl",
        ).detect(
            feature_refinement="spherical_quadratic",
            search_window_size=5,
        )


@patch("xarray.open_dataset")
def test_hodges_detector_init(mock_open: MagicMock) -> None:
    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), np.ones((1, 3, 3)))},
        coords={"time": [0], "latitude": [0, 1, 2], "longitude": [0, 1, 2]},
    )
    mock_open.return_value = ds

    detector = HodgesDetector(pathname="test.nc", variable_name="msl")
    detector._ensure_open()

    mock_open.assert_called_once_with(
        Path("test.nc"), engine="h5netcdf", decode_times=False
    )


@patch("xarray.open_dataset")
def test_hodges_detector_detect_mock(mock_open: MagicMock) -> None:
    # Create real xarray data for reliable behavior
    data: NDArray[np.float64] = np.ones((1, 7, 7)) * 1000
    # Create a nice quadratic peak for feature-point interpolation
    # f(y,x) = 1000 - (y-3)^2 - (x-3.2)^2
    # Grid peak will be at (3,3), refined should be at (3, 3.2)
    for i in range(7):
        for j in range(7):
            data[0, i, j] = 1000 - (i - 3) ** 2 - (j - 3.2) ** 2

    times: NDArray[np.datetime64] = np.array(["2025-12-01"], dtype="datetime64[ns]")
    lats: NDArray[np.float64] = np.arange(7, dtype=float)
    lons: NDArray[np.float64] = np.arange(7, dtype=float)

    ds = xr.Dataset(
        data_vars={"msl": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    mock_open.return_value = ds

    detector = HodgesDetector(pathname="test2.nc", variable_name="msl")
    # Regional data rejects spherical_bspline
    with pytest.raises(
        ValueError, match="spherical_bspline requires a global periodic longitude grid"
    ):
        detector.detect(
            search_window_size=5,
            intensity_threshold=0.0,
            detection_mode="max",
            feature_refinement="spherical_bspline",
        )

    # bspline supports regional frames
    raw_results = detector.detect(
        search_window_size=5,
        intensity_threshold=0.0,
        detection_mode="max",
        feature_refinement="bspline",
    )
    assert len(raw_results) == 1
    result = raw_results[0]
    assert result.latitudes.size == 1
    assert result.latitudes[0] == pytest.approx(3.0, abs=1e-2)
    assert result.longitudes[0] == pytest.approx(3.2, abs=1e-2)

    unrefined = detector.detect(
        search_window_size=5,
        intensity_threshold=0.0,
        detection_mode="max",
        feature_refinement="grid",
    )[0]
    assert unrefined.longitudes[0] == 3.0
    assert unrefined.values[0] == pytest.approx(data[0, 3, 3])


def test_hodges_detector_from_xarray_keeps_generic_detection_values() -> None:
    values = np.ones((1, 5, 5), dtype=np.float64)
    values[0, 2, 2] = -1.0
    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01"], dtype="datetime64[ns]"),
            "latitude": np.arange(5, dtype=np.float64),
            "longitude": np.arange(5, dtype=np.float64),
        },
        name="msl_spectral_filtered",
    )

    detector = HodgesDetector.from_xarray(data, variable_name="msl")
    result = detector.detect(
        search_window_size=3,
        intensity_threshold=0.0,
        detection_mode="min",
        feature_refinement="grid",
    )[0]

    assert result.values.tolist() == [-1.0]
    assert set(result.diagnostics) == {
        "raw_value",
        "object_gridcell_area_km2",
        "object_moment_fitted_area_km2",
        "object_moment_major_axis_km",
        "object_moment_minor_axis_km",
        "object_moment_orientation_degrees",
    }
    np.testing.assert_array_equal(result.diagnostics["raw_value"], np.array([-1.0]))
    assert result.diagnostic_units["raw_value"] is None
    assert result.diagnostic_units["object_gridcell_area_km2"] == "km2"


def test_spherical_spline_refines_synthetic_gaussian_depression() -> None:
    latitudes = np.linspace(-89.0, 89.0, 90, dtype=np.float64)
    longitudes = np.arange(360.0, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(longitudes, latitudes)

    target_latitude = 12.34
    target_longitude = 123.45
    dx = (lon_mesh - target_longitude) * np.cos(np.radians(lat_mesh))
    dy = lat_mesh - target_latitude
    gaussian = 1000.0 - 50.0 * np.exp(-(dx**2 + dy**2) / (2.0 * 2.0**2))
    values = gaussian[np.newaxis, :, :]

    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01"], dtype="datetime64[ns]"),
            "latitude": latitudes,
            "longitude": longitudes,
        },
        name="msl",
    )

    detector = HodgesDetector.from_xarray(data, variable_name="msl")
    result = detector.detect(
        intensity_threshold=990.0,
        detection_mode="min",
        feature_refinement="spherical_bspline",
        bspline_max_iterations=100,
    )[0]

    assert result.latitudes.tolist() == pytest.approx([target_latitude], abs=0.1)
    assert result.longitudes.tolist() == pytest.approx([target_longitude], abs=0.1)
    assert detector.last_refinement_diagnostics[0].status == "success"


def test_track_smoopy_uses_source_surviving_periodic_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.ones((1, 5, 4), dtype=np.float64)
    values[0, 2, 0] = -3.0
    values[0, 2, 2] = -1.0
    values[0, 2, 3] = -2.0
    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01"], dtype="datetime64[ns]"),
            "latitude": np.arange(5.0),
            "longitude": np.arange(4.0) * 90.0,
        },
        name="msl",
    )
    evaluated_longitudes: list[float] = []

    def refine(
        _surface: object,
        latitude: float,
        longitude: float,
        **_kwargs: object,
    ) -> BsplineRefinementResult:
        evaluated_longitudes.append(longitude)
        return BsplineRefinementResult(latitude, longitude, -3.0, "success")

    monkeypatch.setattr(
        "pystormtracker.hodges.detector.refine_bspline_feature_point", refine
    )
    detector = HodgesDetector.from_xarray(data, variable_name="msl")
    result = detector.detect(
        intensity_threshold=0.0,
        detection_mode="min",
        min_grid_points=2,
        feature_refinement="bspline",
    )[0]

    assert evaluated_longitudes == [360.0]
    np.testing.assert_array_equal(result.longitudes, np.array([360.0]))


@pytest.mark.parametrize("feature_refinement", ["bspline", "spherical_bspline"])
def test_spline_detection_is_invariant_to_signed_longitudes(
    feature_refinement: HodgesFeatureRefinement,
) -> None:
    latitudes = np.linspace(-40.0, 40.0, 9, dtype=np.float64)
    unsigned_longitudes = np.arange(0.0, 360.0, 22.5, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(unsigned_longitudes, latitudes)
    # The physical extremum is -90 degrees in the signed representation.
    # This exercises the former rectangular candidate/refinement mismatch.
    longitude_distance = (lon_grid - 270.0 + 180.0) % 360.0 - 180.0
    frame = 0.02 * (lat_grid - 5.0) ** 2 + 0.01 * longitude_distance**2

    signed_order = np.argsort((unsigned_longitudes + 180.0) % 360.0 - 180.0)
    signed_longitudes = (unsigned_longitudes[signed_order] + 180.0) % 360.0 - 180.0
    signed_frame = frame[:, signed_order]

    unsigned_step, _ = detect_hodges_frame(
        frame,
        np.datetime64("2024-01-01"),
        latitudes,
        unsigned_longitudes,
        intensity_threshold=80.0,
        mode="min",
        min_grid_points=3,
        feature_refinement=feature_refinement,
        track_smoopy_optimization_scale=1.0,
        bspline_max_iterations=100,
    )
    signed_step, _ = detect_hodges_frame(
        signed_frame,
        np.datetime64("2024-01-01"),
        latitudes,
        signed_longitudes,
        intensity_threshold=80.0,
        mode="min",
        min_grid_points=3,
        feature_refinement=feature_refinement,
        track_smoopy_optimization_scale=1.0,
        bspline_max_iterations=100,
    )

    np.testing.assert_allclose(signed_step.latitudes, unsigned_step.latitudes)
    np.testing.assert_allclose(
        np.mod(signed_step.longitudes, 360.0),
        np.mod(unsigned_step.longitudes, 360.0),
    )
    np.testing.assert_allclose(signed_step.values, unsigned_step.values)
    np.testing.assert_allclose(
        signed_step.diagnostics["raw_value"],
        unsigned_step.diagnostics["raw_value"],
    )


def test_rectangular_spline_preserves_a_physical_endpoint_extremum() -> None:
    """A minimum across 0/360 has identical signed and unsigned detection."""
    latitudes = np.linspace(-40.0, 40.0, 17, dtype=np.float64)
    unsigned_longitudes = np.arange(0.0, 360.0, 10.0, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(unsigned_longitudes, latitudes)
    longitude_distance = (lon_grid - 359.5 + 180.0) % 360.0 - 180.0
    frame = 100.0 - 30.0 * np.exp(
        -(((lat_grid - 5.0) / 8.0) ** 2) - (longitude_distance / 12.0) ** 2
    )
    signed_order = np.argsort((unsigned_longitudes + 180.0) % 360.0 - 180.0)
    signed_longitudes = (unsigned_longitudes[signed_order] + 180.0) % 360.0 - 180.0
    signed_frame = frame[:, signed_order]

    unsigned_step, unsigned_diagnostics = detect_hodges_frame(
        frame,
        np.datetime64("2024-01-01"),
        latitudes,
        unsigned_longitudes,
        intensity_threshold=90.0,
        mode="min",
        min_grid_points=3,
        feature_refinement="bspline",
        track_smoopy_optimization_scale=1.0,
        bspline_max_iterations=100,
    )
    signed_step, signed_diagnostics = detect_hodges_frame(
        signed_frame,
        np.datetime64("2024-01-01"),
        latitudes,
        signed_longitudes,
        intensity_threshold=90.0,
        mode="min",
        min_grid_points=3,
        feature_refinement="bspline",
        track_smoopy_optimization_scale=1.0,
        bspline_max_iterations=100,
    )

    assert unsigned_step.latitudes.size == 1
    np.testing.assert_allclose(signed_step.latitudes, unsigned_step.latitudes)
    np.testing.assert_allclose(signed_step.longitudes, unsigned_step.longitudes)
    np.testing.assert_allclose(signed_step.values, unsigned_step.values)
    assert signed_diagnostics[0].status == unsigned_diagnostics[0].status


@pytest.mark.parametrize("is_min", [True, False])
def test_packed_rectangular_candidates_match_reference_scan(is_min: bool) -> None:
    """The packed detector keeps TRACK candidate arrays exactly unchanged."""
    latitudes = np.linspace(-80.0, 80.0, 17, dtype=np.float64)
    unsigned_longitudes = np.arange(0.0, 360.0, 10.0, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(unsigned_longitudes, latitudes)
    frame = (
        100.0
        - 30.0 * np.exp(-(((lat_grid - 5.0) / 8.0) ** 2))
        + 0.01 * np.cos(np.radians(lon_grid * 3.0))
    )
    if not is_min:
        frame = -frame
    order = np.argsort((unsigned_longitudes + 180.0) % 360.0 - 180.0)
    signed_longitudes = (unsigned_longitudes[order] + 180.0) % 360.0 - 180.0
    signed_frame = frame[:, order]

    reference = _detect_track_rectangular_candidates_reference(
        signed_frame,
        latitudes,
        signed_longitudes,
        intensity_threshold=95.0 if is_min else -95.0,
        is_min=is_min,
        min_grid_points=3,
    )
    packed = _detect_track_rectangular_candidates(
        signed_frame,
        latitudes,
        signed_longitudes,
        intensity_threshold=95.0 if is_min else -95.0,
        is_min=is_min,
        min_grid_points=3,
    )
    for expected, actual in zip(reference, packed, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_spherical_duplicate_ranking_uses_immutable_initial_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latitudes = np.linspace(-60.0, 60.0, 5, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, 45.0, dtype=np.float64)
    frame = np.full((latitudes.size, longitudes.size), 2.0, dtype=np.float64)
    frame[2, 1:6] = 0.0
    frame[2, 1] = -1.0
    frame[2, 5] = -1.0

    def collapse_to_second_minimum(
        _surface: object,
        latitude: float,
        longitude: float,
        **_kwargs: object,
    ) -> BsplineRefinementResult:
        return BsplineRefinementResult(latitude, 225.0, -2.0, "success")

    monkeypatch.setattr(
        "pystormtracker.hodges.detector.refine_spherical_bspline_feature_point",
        collapse_to_second_minimum,
    )

    step, _ = detect_hodges_frame(
        frame,
        np.datetime64("2024-01-01"),
        latitudes,
        longitudes,
        intensity_threshold=0.0,
        mode="min",
        min_grid_points=3,
        feature_refinement="spherical_bspline",
    )

    assert step.values.size == 3
    assert step.values[0] < DUFF_FEATURE_CUTOFF
    assert step.values[1] < DUFF_FEATURE_CUTOFF
    assert step.values[2] > DUFF_FEATURE_CUTOFF


def test_threshold_selection_is_inclusive_and_rejects_nan() -> None:
    frame = np.array([[-np.inf, -1.0, 0.0, np.nan, np.inf]], dtype=np.float64)

    labels_max, _ = _label_connected_components(
        frame, threshold=0.0, is_min=False, periodic_x=False
    )
    np.testing.assert_array_equal(
        (labels_max > 0).astype(np.float64),
        np.array([[0.0, 0.0, 1.0, 0.0, 1.0]]),
    )

    labels_min, _ = _label_connected_components(
        frame, threshold=0.0, is_min=True, periodic_x=False
    )
    np.testing.assert_array_equal(
        (labels_min > 0).astype(np.float64),
        np.array([[1.0, 1.0, 1.0, 0.0, 0.0]]),
    )


def test_hodges_detector_groups_tied_adjacent_extrema() -> None:
    values = np.zeros((1, 4, 4), dtype=np.float64)
    values[0, :3, :3] = 1.0
    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01"], dtype="datetime64[ns]"),
            "latitude": np.arange(1.0, 5.0),
            "longitude": np.arange(1.0, 5.0),
        },
        name="msl",
    )

    detector = HodgesDetector.from_xarray(data, variable_name="msl")
    result = detector.detect(
        intensity_threshold=1.0,
        detection_mode="max",
        feature_refinement="grid",
        group_adjacent_extrema=True,
    )[0]

    np.testing.assert_array_equal(result[1], np.array([2.0]))
    np.testing.assert_array_equal(result[2], np.array([2.0]))
    np.testing.assert_array_equal(result[3], np.array([1.0]))


def test_hodges_detector_grouping_requires_grid_feature_points() -> None:
    data = xr.DataArray(
        np.ones((1, 3, 3), dtype=np.float64),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2025-12-01"], dtype="datetime64[ns]"),
            "latitude": np.arange(3.0),
            "longitude": np.arange(3.0),
        },
        name="msl",
    )

    detector = HodgesDetector.from_xarray(data, variable_name="msl")
    with pytest.raises(ValueError, match="requires feature_refinement='grid'"):
        detector.detect(group_adjacent_extrema=True)


def test_ccl_does_not_join_projected_x_boundaries() -> None:
    frame = np.zeros((3, 5), dtype=np.float64)
    frame[1, 0] = 1.0
    frame[1, -1] = 1.0

    _, global_objects = _label_connected_components(
        frame, threshold=0.5, is_min=False, periodic_x=True
    )
    _, projected_objects = _label_connected_components(
        frame, threshold=0.5, is_min=False, periodic_x=False
    )

    assert global_objects == 1
    assert projected_objects == 2


def test_ccl_matches_track_vertex_and_edge_connectivity() -> None:
    frame = np.zeros((4, 4), dtype=np.float64)
    frame[0, 0] = 1.0
    frame[1, 1] = 1.0

    vertex_labels, vertex_count = _label_connected_components(
        frame, threshold=0.5, is_min=False, periodic_x=False
    )
    edge_labels, edge_count = _label_connected_components(
        frame,
        threshold=0.5,
        is_min=False,
        periodic_x=False,
        vertex_connectivity=False,
    )

    assert vertex_count == 1
    np.testing.assert_array_equal(
        vertex_labels,
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )
    assert edge_count == 2
    np.testing.assert_array_equal(
        edge_labels,
        np.array(
            [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
    )


def test_object_first_indices_use_track_west_to_east_order() -> None:
    labels = np.array([[0, 2, 2, 0], [1, 1, 0, 3], [1, 0, 0, 3]], dtype=np.int32)

    rows, columns = _find_object_first_indices(labels, num_objects=3)

    np.testing.assert_array_equal(rows, np.array([3, 1, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(columns, np.array([4, 0, 1, 3], dtype=np.int64))


def test_object_extrema_do_not_compete_across_object_labels() -> None:
    frame = np.zeros((3, 3), dtype=np.float64)
    frame[0, 0] = 10.0
    frame[1, 1] = 9.0
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[0, 0] = 1
    labels[1, 1] = 2

    extrema = _find_object_extrema(
        frame,
        labels,
        num_objects=2,
        size=3,
        is_min=False,
        min_points=1,
        periodic_x=False,
    )

    np.testing.assert_array_equal(
        extrema,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_object_extrema_boundary_flag_matches_track() -> None:
    frame = np.zeros((3, 4), dtype=np.float64)
    frame[0, :3] = np.array([1.0, 9.0, 1.0])
    labels = np.zeros((3, 4), dtype=np.int32)
    labels[0, :3] = 1

    unexcluded = _find_object_extrema(
        frame,
        labels,
        num_objects=1,
        size=3,
        is_min=False,
        min_points=1,
        periodic_x=False,
    )
    excluded = _find_object_extrema(
        frame,
        labels,
        num_objects=1,
        size=3,
        is_min=False,
        min_points=1,
        periodic_x=False,
        exclude_boundary_extrema=True,
    )

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[0, 1] = 1.0
    np.testing.assert_array_equal(unexcluded, expected)
    np.testing.assert_array_equal(excluded, np.zeros((3, 4), dtype=np.float64))


def test_min_grid_points_retains_objects_at_the_public_boundary() -> None:
    frame = np.zeros((3, 4), dtype=np.float64)
    frame[0, 0] = 1.0
    frame[1, 2:] = 1.0
    labels = np.zeros((3, 4), dtype=np.int32)
    labels[0, 0] = 1
    labels[1, 2:] = 2

    extrema = _find_object_extrema(
        frame,
        labels,
        num_objects=2,
        size=3,
        is_min=False,
        min_points=2,
        periodic_x=False,
    )

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[1, 2:] = 1.0
    np.testing.assert_array_equal(extrema, expected)


def test_group_object_extrema_matches_track_plateau_representative() -> None:
    frame = np.zeros((4, 4), dtype=np.float64)
    frame[:3, :3] = 1.0
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[:3, :3] = 1
    extrema = _find_object_extrema(
        frame,
        labels,
        num_objects=1,
        size=3,
        is_min=False,
        min_points=1,
        periodic_x=False,
    )

    lats, lons, values, object_ids = _group_object_extrema(
        extrema,
        labels,
        frame,
        np.arange(1.0, 5.0),
        np.arange(1.0, 5.0),
        periodic_x=False,
    )

    np.testing.assert_array_equal(lats, np.array([2.0]))
    np.testing.assert_array_equal(lons, np.array([2.0]))
    np.testing.assert_array_equal(values, np.array([1.0]))
    np.testing.assert_array_equal(object_ids, np.array([1], dtype=np.int32))


def test_projected_object_area_uses_kilometer_coordinates() -> None:
    frame = np.ones((3, 3), dtype=np.float64)
    labels = np.ones((3, 3), dtype=np.int32)
    y = np.array([-100.0, 0.0, 100.0])
    x = np.array([-100.0, 0.0, 100.0])

    raw_area, fitted_area, major, minor, _ = _compute_object_properties(
        frame,
        labels,
        1,
        y,
        x,
        threshold=0.0,
        is_min=False,
        spherical_coords=False,
    )

    assert raw_area[1] == 90_000.0
    assert fitted_area[1] > 0.0
    assert major[1] > 0.0
    assert minor[1] > 0.0


def test_spherical_object_moments_unwrap_longitude_seam() -> None:
    lon = np.linspace(0.0, 360.0, 144, endpoint=False)
    lat = np.array([-2.5, 0.0, 2.5])
    frame = np.zeros((3, 144), dtype=np.float64)
    labels = np.zeros((3, 144), dtype=np.int32)
    frame[1, [0, -1]] = 1.0
    labels[1, [0, -1]] = 1

    _, _, major, _, _ = _compute_object_properties(
        frame,
        labels,
        1,
        lat,
        lon,
        threshold=0.0,
        is_min=False,
        spherical_coords=True,
    )

    assert 200.0 < major[1] < 400.0
