from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.hodges.constants import (
    MGE_MAX_ITERATIONS_DEFAULT,
)
from pystormtracker.hodges.detections import HodgesCenterFrame
from pystormtracker.hodges.detector import DUFF_FEATURE_VALUE
from pystormtracker.hodges.linker import HodgesLinker
from pystormtracker.hodges.mge import _mge_iteration
from pystormtracker.models.time import encode_time_values
from pystormtracker.models.tracker import CenterFrame


def _hodges_detection_step(
    time: np.datetime64,
    longitudes: NDArray[np.float64],
    primary_values: NDArray[np.float64],
    raw_values: NDArray[np.float64],
) -> HodgesCenterFrame:
    """Create an aligned Hodges frame with identifiable diagnostics."""
    return HodgesCenterFrame(
        time,
        np.zeros(primary_values.size, dtype=np.float64),
        longitudes,
        primary_values,
        {
            "raw_value": raw_values,
            "object_gridcell_area_km2": primary_values + 1000.0,
        },
        {
            "raw_value": None,
            "object_gridcell_area_km2": "km2",
        },
    )


def test_hodges_linker_init() -> None:
    linker = HodgesLinker(w1=0.5, w2=0.5, dmax=10.0)
    assert linker.w1 == 0.5
    assert linker.w2 == 0.5
    assert linker.dmax == 6.5


def test_hodges_linker_removes_isolated_one_frame_feature() -> None:
    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    empty = linker.link([], primary_variable="msl", mode="min")
    assert len(empty) == 0

    one_frame_detections: list[CenterFrame] = [
        CenterFrame(
            time=np.datetime64("2025-12-01T00:00:00"),
            latitudes=np.array([10.0]),
            longitudes=np.array([20.0]),
            values=np.array([1000.0]),
        )
    ]
    one_frame = linker.link(
        one_frame_detections,
        primary_variable="msl",
        mode="min",
    )
    assert len(one_frame) == 0


def test_hodges_linker_feature_filter_and_workspace_match_source_order() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    detections = [
        CenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([0.0, 0.0]),
            np.array([-1.0, 1.0]),
            np.array([1.0, 1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T12:00:00"),
            np.array([0.0]),
            np.array([20.0]),
            np.array([1.0]),
        ),
    ]

    filtered = linker._filter_feature_points(detections)
    assert [step.values.size for step in filtered] == [1, 2, 0]
    workspace = linker._initialize_mge_workspace(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 1.0]),
        np.array([0, 1, 3, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        workspace.assignments,
        np.array([[0, 2, -1], [-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
    )

    native_filtered = linker._filter_feature_points_numba(detections)
    for source_step, native_step in zip(filtered, native_filtered, strict=True):
        np.testing.assert_array_equal(source_step.latitudes, native_step.latitudes)
        np.testing.assert_array_equal(source_step.longitudes, native_step.longitudes)
        np.testing.assert_array_equal(source_step.values, native_step.values)

    native_offsets = np.array([0, 1, 3, 3], dtype=np.int64)
    native_workspace = linker._initialize_mge_workspace_numba(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 1.0]),
        native_offsets,
    )
    np.testing.assert_array_equal(native_workspace.assignments, workspace.assignments)


def test_hodges_linker_filters_every_aligned_diagnostic_column() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    detections = [
        _hodges_detection_step(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0, 20.0]),
            np.array([1.0, 2.0]),
            np.array([101.0, 102.0]),
        ),
        _hodges_detection_step(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([1.0]),
            np.array([3.0]),
            np.array([103.0]),
        ),
    ]

    filtered = linker._filter_feature_points(detections)

    np.testing.assert_array_equal(filtered[0].values, np.array([1.0]))
    np.testing.assert_array_equal(
        filtered[0].diagnostics["raw_value"],
        np.array([101.0]),
    )
    np.testing.assert_array_equal(
        filtered[0].diagnostics["object_gridcell_area_km2"],
        np.array([1001.0]),
    )


def test_hodges_linker_removes_source_duff_feature_after_connectivity() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    detections = [
        CenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([0.0]),
            np.array([DUFF_FEATURE_VALUE]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        ),
    ]

    filtered = linker._filter_feature_points(detections)

    # TRACK checks adjacent feature points before it removes their DUFF_PT
    # counterparts, so the valid second-frame point remains connected.
    assert [step.values.size for step in filtered] == [0, 1]


def test_hodges_linker_exports_aligned_diagnostics_with_primary_variables() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    tracks = linker.link(
        [
            _hodges_detection_step(
                np.datetime64("2025-12-01T00:00:00"),
                np.array([0.0]),
                np.array([10.0]),
                np.array([110.0]),
            ),
            _hodges_detection_step(
                np.datetime64("2025-12-01T06:00:00"),
                np.array([1.0]),
                np.array([20.0]),
                np.array([120.0]),
            ),
        ],
        primary_variable="msl",
        mode="min",
        unit="Pa",
    )

    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0].variables["msl"], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(
        tracks[0].variables["raw_value"],
        np.array([110.0, 120.0]),
    )
    np.testing.assert_array_equal(
        tracks[0].variables["object_gridcell_area_km2"],
        np.array([1010.0, 1020.0]),
    )
    assert tracks.metadata.units["raw_value"] == "Pa"
    assert tracks.metadata.units["object_gridcell_area_km2"] == "km2"


def test_hodges_diagnostic_schema_requires_aligned_columns_and_units() -> None:
    with pytest.raises(ValueError, match="match feature count"):
        HodgesCenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            {"raw_value": np.array([1.0, 2.0])},
            {"raw_value": None},
        )

    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    conflict = _hodges_detection_step(
        np.datetime64("2025-12-01T00:00:00"),
        np.array([0.0]),
        np.array([1.0]),
        np.array([101.0]),
    )
    with pytest.raises(ValueError, match="conflicts with primary variable"):
        linker.link([conflict], primary_variable="raw_value", mode="min")


def test_hodges_detection_steps_own_read_only_aligned_columns() -> None:
    latitudes = np.array([10.0], dtype=np.float64)
    longitudes = np.array([20.0], dtype=np.float64)
    values = np.array([30.0], dtype=np.float64)
    raw_values = np.array([40.0], dtype=np.float64)
    step = HodgesCenterFrame(
        np.datetime64("2025-12-01T00:00:00"),
        latitudes,
        longitudes,
        values,
        {"raw_value": raw_values},
        {"raw_value": None},
    )

    latitudes[0] = -10.0
    longitudes[0] = -20.0
    values[0] = -30.0
    raw_values[0] = -40.0
    np.testing.assert_array_equal(step.latitudes, np.array([10.0]))
    np.testing.assert_array_equal(step.longitudes, np.array([20.0]))
    np.testing.assert_array_equal(step.values, np.array([30.0]))
    np.testing.assert_array_equal(step.diagnostics["raw_value"], np.array([40.0]))
    with pytest.raises(ValueError, match="read-only"):
        step.values[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        step.diagnostics["raw_value"][0] = 0.0


def test_hodges_diagnostics_follow_feature_indices_through_topology_changes() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    tracks = np.array(
        [
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    tracks[0, 2], tracks[1, 2] = tracks[1, 2], tracks[0, 2]
    tracks[2, 3], tracks[1, 3] = tracks[1, 3], tracks[2, 3]

    feature_values = np.arange(10.0, 20.0, dtype=np.float64)
    feature_lats = np.zeros(10, dtype=np.float64)
    feature_lons = np.arange(10, dtype=np.float64) * 10.0
    failed = linker._apply_track_fail(
        tracks,
        0,
        1,
        feature_lats,
        feature_lons,
        direction="forward",
    )
    finalized = linker._split_track_sections(failed)
    output = linker._build_tracks(
        finalized,
        np.arange(5, dtype=np.int64),
        feature_lats,
        feature_lons,
        feature_values,
        {
            "raw_value": feature_values + 100.0,
            "object_gridcell_area_km2": feature_values + 1000.0,
        },
        primary_variable="msl",
        mode="min",
        bounds=None,
        units={
            "msl": "Pa",
            "raw_value": "Pa",
            "object_gridcell_area_km2": "km2",
        },
        processing=(),
    )

    observed_primary = np.concatenate([track.variables["msl"] for track in output])
    np.testing.assert_array_equal(np.sort(observed_primary), feature_values)
    for track in output:
        primary = track.variables["msl"]
        np.testing.assert_array_equal(track.variables["raw_value"], primary + 100.0)
        np.testing.assert_array_equal(
            track.variables["object_gridcell_area_km2"],
            primary + 1000.0,
        )


def test_hodges_linker_initialization_averages_endpoint_zone_dmax() -> None:
    linker = HodgesLinker(
        dmax=4.0,
        dmax_zones=np.array(
            [[0.0, 0.0, -1.0, 1.0, 2.0], [0.0, 360.0, -1.0, 1.0, 4.0]],
            dtype=np.float64,
        ),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )

    workspace = linker._initialize_mge_workspace(
        np.array([0.0, 0.0]),
        np.array([0.0, 2.5]),
        np.array([0, 1, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        workspace.assignments,
        np.array([[0, 1], [-1, -1]], dtype=np.int64),
    )


def test_hodges_linker_rejects_uncovered_configured_zone() -> None:
    linker = HodgesLinker(
        dmax=4.0,
        dmax_zones=np.array([[0.0, 10.0, -1.0, 1.0, 4.0]], dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    detections = [
        CenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([20.0]),
            np.array([1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([0.0]),
            np.array([21.0]),
            np.array([1.0]),
        ),
    ]

    with pytest.raises(ValueError, match="do not cover feature"):
        linker.link(detections, primary_variable="msl", mode="min")


def test_hodges_linker_uses_zone_maximum_as_source_default_dmax() -> None:
    linker = HodgesLinker(
        dmax=4.0,
        dmax_zones=np.array(
            [[0.0, 360.0, -90.0, 0.0, 6.0], [0.0, 360.0, 0.0, 90.0, 8.0]],
            dtype=np.float64,
        ),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )

    assert linker.dmax == 8.0


def test_hodges_linker_maps_time_gaps_to_preceding_nmiss_counts() -> None:
    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        time_step_ms=6 * 60 * 60 * 1000,
    )

    missing_counts = linker._infer_missing_input_counts(
        np.array([0, 6, 18, 36], dtype=np.int64) * 60 * 60 * 1000
    )

    np.testing.assert_array_equal(missing_counts, np.array([0, 1, 2, 0]))


def test_hodges_linker_rejects_nonintegral_explicit_cadence_gaps() -> None:
    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        time_step_ms=6 * 60 * 60 * 1000,
    )

    with pytest.raises(ValueError, match="integral multiple"):
        linker._infer_missing_input_counts(
            np.array([0, 7 * 60 * 60 * 1000], dtype=np.int64)
        )


def test_hodges_linker_validates_zone_coverage_before_feature_prefilter() -> None:
    linker = HodgesLinker(
        dmax=1.0,
        dmax_zones=np.array([[0.0, 1.0, -90.0, 90.0, 1.0]], dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    detections = [
        CenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([5.0]),
            np.array([1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([0.0]),
            np.array([20.0]),
            np.array([1.0]),
        ),
    ]

    with pytest.raises(ValueError, match="do not cover feature"):
        linker.link(detections, primary_variable="msl", mode="min")


def test_hodges_linker_selects_missing_frame_parameter_for_time_jump() -> None:
    linker = HodgesLinker(
        dmax=2.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        missing_frame_parameters=np.array(
            [[2.0, 0.5], [10.0, 0.5]],
            dtype=np.float64,
        ),
        time_step_ms=6 * 60 * 60 * 1000,
    )
    detections = [
        CenterFrame(
            np.datetime64("2025-12-01T00:00:00"),
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T06:00:00"),
            np.array([0.0]),
            np.array([1.0]),
            np.array([1.0]),
        ),
        CenterFrame(
            np.datetime64("2025-12-01T18:00:00"),
            np.array([0.0]),
            np.array([8.0]),
            np.array([1.0]),
        ),
    ]

    tracks = linker.link(detections, primary_variable="msl", mode="min")

    assert len(tracks) == 1
    assert [point.lon for point in tracks[0]] == [0.0, 1.0, 8.0]
    np.testing.assert_array_equal(
        tracks[0].times,
        encode_time_values(
            [
                np.datetime64("2025-12-01T00:00:00"),
                np.datetime64("2025-12-01T06:00:00"),
                np.datetime64("2025-12-01T18:00:00"),
            ]
        ),
    )


def test_hodges_linker_preserves_observed_times_around_missing_input_frames() -> None:
    linker = HodgesLinker(
        dmax=2.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        missing_frame_parameters=np.array(
            [[2.0, 0.5], [6.0, 0.5], [10.0, 0.5]],
            dtype=np.float64,
        ),
        time_step_ms=6 * 60 * 60 * 1000,
    )
    times = [
        np.datetime64("2025-12-01T00:00:00"),
        np.datetime64("2025-12-01T06:00:00"),
        np.datetime64("2025-12-02T00:00:00"),
        np.datetime64("2025-12-02T06:00:00"),
    ]
    detections = [
        CenterFrame(
            time,
            np.array([0.0]),
            np.array([longitude]),
            np.array([1.0]),
        )
        for time, longitude in zip(times, (0.0, 1.0, 20.0, 21.0), strict=True)
    ]

    tracks = linker.link(detections, primary_variable="msl", mode="min")

    assert len(tracks) == 2
    assert [point.lon for point in tracks[0]] == [0.0, 1.0]
    assert [point.lon for point in tracks[1]] == [20.0, 21.0]
    np.testing.assert_array_equal(tracks[0].times, encode_time_values(times[:2]))
    np.testing.assert_array_equal(tracks[1].times, encode_time_values(times[2:]))


def test_mge_uses_nmiss_selected_phimax_for_phantom_cost() -> None:
    tracks = np.array([[0, 1, -1], [-1, -1, 2]], dtype=np.int64)
    latitudes = np.zeros(3, dtype=np.float64)
    longitudes = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    missing_counts = np.array([0, 1, 0], dtype=np.int64)
    dmax_parameters = np.array([10.0, 10.0], dtype=np.float64)
    zones = np.zeros((0, 5), dtype=np.float64)
    adaptive = np.zeros((2, 0), dtype=np.float64)

    accepted = _mge_iteration(
        tracks.copy(),
        latitudes,
        longitudes,
        1,
        True,
        0.5,
        0.5,
        dmax_parameters,
        np.array([0.5, 0.5], dtype=np.float64),
        missing_counts,
        zones,
        adaptive,
    )
    rejected = _mge_iteration(
        tracks.copy(),
        latitudes,
        longitudes,
        1,
        True,
        0.5,
        0.5,
        dmax_parameters,
        np.array([0.5, 0.0], dtype=np.float64),
        missing_counts,
        zones,
        adaptive,
    )

    assert accepted == (0, 1)
    assert rejected == (-1, -1)


def test_hodges_linker_rejects_multi_parameter_zones_or_adaptive_tables() -> None:
    parameters = np.array([[2.0, 0.5], [10.0, 0.5]], dtype=np.float64)

    with pytest.raises(ValueError, match="multiple missing-frame parameter sets"):
        HodgesLinker(missing_frame_parameters=parameters)

    with pytest.raises(ValueError, match="time_step_ms is required"):
        HodgesLinker(
            dmax_zones=np.zeros((0, 5), dtype=np.float64),
            adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
            missing_frame_parameters=parameters,
        )


def test_hodges_linker_track_fail_moves_source_contiguous_sections() -> None:
    linker = HodgesLinker(
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    features_lat = np.zeros(6, dtype=np.float64)
    features_lon = np.array([0.0, 1.0, 2.0, 20.0, 21.0, 40.0], dtype=np.float64)

    forward = np.array([[0, 1, 2, 3, 4, -1], [-1, -1, -1, -1, -1, -1]], dtype=np.int64)
    forward = linker._apply_track_fail(
        forward,
        0,
        1,
        features_lat,
        features_lon,
        direction="forward",
    )
    np.testing.assert_array_equal(
        forward,
        np.array([[0, 1, 2, -1, -1, -1], [-1, -1, -1, 3, 4, -1]], dtype=np.int64),
    )

    backward = np.array([[0, 3, 4, 5], [-1, -1, -1, -1]], dtype=np.int64)
    backward = linker._apply_track_fail(
        backward,
        0,
        2,
        features_lat,
        features_lon,
        direction="backward",
    )
    np.testing.assert_array_equal(
        backward,
        np.array([[-1, 3, 4, 5], [0, -1, -1, -1]], dtype=np.int64),
    )


def test_hodges_linker_final_split_preserves_paired_workspace_order() -> None:
    track_matrix = np.array(
        [[0, 1, -1, 2, 3, -1, 4], [-1, -1, -1, -1, -1, -1, -1]],
        dtype=np.int64,
    )
    split = HodgesLinker._split_track_sections(track_matrix)

    np.testing.assert_array_equal(
        split,
        np.array(
            [
                [-1, -1, -1, -1, -1, -1, 4],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 2, 3, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        ),
    )


def test_hodges_linker_adaptive_preprocessing_filter_uses_directional_split() -> None:
    linker = HodgesLinker(
        w1=1.0,
        w2=0.0,
        dmax=5.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.array(
            [[1.0, 2.0, 3.0, 4.0], [0.1, 0.1, 0.1, 0.1]],
            dtype=np.float64,
        ),
    )
    features_lat = np.zeros(4, dtype=np.float64)
    features_lon = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    track_matrix = np.array(
        [[0, 1, 2, 3], [-1, -1, -1, -1]],
        dtype=np.int64,
    )

    forward = linker._apply_track_constraint_filter(
        track_matrix.copy(),
        features_lat,
        features_lon,
        direction="forward",
    )
    np.testing.assert_array_equal(
        forward,
        np.array(
            [[0, 1, -1, -1], [-1, -1, -1, -1], [-1, -1, 2, 3], [-1, -1, -1, -1]],
            dtype=np.int64,
        ),
    )

    backward = linker._apply_track_constraint_filter(
        track_matrix.copy(),
        features_lat,
        features_lon,
        direction="backward",
    )
    np.testing.assert_array_equal(
        backward,
        np.array(
            [[-1, -1, 2, 3], [-1, -1, -1, -1], [0, 1, -1, -1], [-1, -1, -1, -1]],
            dtype=np.int64,
        ),
    )


def test_hodges_linker_link_straight() -> None:
    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )

    # Create two detections moving in a straight line
    # T0: (0,0) and (10,10)
    # T1: (0,1) and (10,11)
    # T2: (0,2) and (10,12)

    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")
    t2 = np.datetime64("2025-12-01T12:00:00")

    detections: list[CenterFrame] = [
        CenterFrame(
            t0, np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.array([1000.0, 1000.0])
        ),
        CenterFrame(
            t1, np.array([0.0, 10.0]), np.array([1.0, 11.0]), np.array([990.0, 990.0])
        ),
        CenterFrame(
            t2, np.array([0.0, 10.0]), np.array([2.0, 12.0]), np.array([980.0, 980.0])
        ),
    ]

    tracks = linker.link(detections, primary_variable="msl", mode="min")

    assert len(tracks) == 2
    # Verify track 1
    tr1 = tracks[0]
    assert len(tr1) == 3
    assert tr1[0].lat == 0.0
    assert tr1[1].lat == 0.0
    assert tr1[2].lat == 0.0

    # Verify track 2
    tr2 = tracks[1]
    assert len(tr2) == 3
    assert tr2[0].lat == 10.0
    assert tr2[1].lat == 10.0
    assert tr2[2].lat == 10.0


def test_hodges_linker_link_crossing() -> None:
    """
    Test that MGE correctly resolves track crossing which
    nearest-neighbor might fail.
    """
    linker = HodgesLinker(
        dmax=15.0,
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )

    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")
    t2 = np.datetime64("2025-12-01T12:00:00")

    # Two tracks crossing at T1
    # Track A: (0,0) -> (5,5) -> (10,10)
    # Track B: (0,10) -> (5,5) -> (10,0)

    # Detections (sorted by lat for ambiguity)
    detections: list[CenterFrame] = [
        CenterFrame(
            t0, np.array([0.0, 0.0]), np.array([0.0, 10.0]), np.array([1000.0, 1000.0])
        ),
        CenterFrame(
            t1,
            np.array([5.0, 5.0001]),
            np.array([5.0, 5.0001]),
            np.array([990.0, 990.0]),
        ),
        CenterFrame(
            t2, np.array([10.0, 10.0]), np.array([10.0, 0.0]), np.array([980.0, 980.0])
        ),
    ]

    tracks = linker.link(detections, primary_variable="msl", mode="min")

    assert len(tracks) == 2
    # One track should go from (0,0) to (10,10)
    found_a = False
    for tr in tracks:
        if tr[0].lat == 0.0 and tr[0].lon == 0.0 and tr[2].lat == 10.0:
            found_a = True
            break
    assert found_a


class TestMgeMaxIterations:
    """Tests for the mge_max_iterations parameter."""

    def test_default_mge_max_iterations_is_three(self) -> None:
        linker = HodgesLinker()
        assert linker.mge_max_iterations == 3
        assert MGE_MAX_ITERATIONS_DEFAULT == 3

    def test_nonpositive_mge_max_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="mge_max_iterations must be positive"):
            HodgesLinker(mge_max_iterations=0)
        with pytest.raises(ValueError, match="mge_max_iterations must be positive"):
            HodgesLinker(mge_max_iterations=-1)

    def test_positive_mge_max_iterations_accepted(self) -> None:
        for n in (1, 2, 3, 10, 100):
            linker = HodgesLinker(mge_max_iterations=n)
            assert linker.mge_max_iterations == n


class TestDirectionalMGE:
    """Tests for TRACK-style directional MGE scheduling."""

    def test_three_or_fewer_frames_skip_mge(self) -> None:
        """TRACK only enters its MGE outer loop when frame_num is greater than 3."""
        linker = HodgesLinker(
            dmax_zones=np.zeros((0, 5), dtype=np.float64),
            adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        )
        initial = np.array([[0, 1, 2], [-1, -1, -1]], dtype=np.int64)

        result = linker._run_directional_mge(
            initial,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            3,
        )

        np.testing.assert_array_equal(result, initial)

    def test_each_direction_converges_before_switching(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direction repeats until a complete sweep makes no exchange."""
        linker = HodgesLinker(
            mge_max_iterations=3,
            dmax_zones=np.zeros((0, 5), dtype=np.float64),
            adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        )

        forward_results = [True, True, False, False]
        backward_results = [True, False]
        calls: list[str] = []

        def fake_forward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts

            calls.append("forward")
            updated = tracks.copy()
            updated[0, 0] += 1

            assert forward_results, "unexpected additional forward sweep"
            return updated, forward_results.pop(0)

        def fake_backward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts

            calls.append("backward")
            updated = tracks.copy()
            updated[0, 0] += 1

            assert backward_results, "unexpected additional backward sweep"
            return updated, backward_results.pop(0)

        monkeypatch.setattr(
            linker,
            "_run_forward_mge_iteration",
            fake_forward,
        )
        monkeypatch.setattr(
            linker,
            "_run_backward_mge_iteration",
            fake_backward,
        )

        result = linker._run_directional_mge(
            np.zeros((1, 4), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            4,
        )

        assert calls == [
            "forward",
            "forward",
            "forward",
            "backward",
            "backward",
            "forward",
        ]
        assert forward_results == []
        assert backward_results == []
        assert result[0, 0] == len(calls)

    def test_final_outer_round_is_forward_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The third default outer round must not run backward MGE."""
        linker = HodgesLinker(
            mge_max_iterations=3,
            dmax_zones=np.zeros((0, 5), dtype=np.float64),
            adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        )

        # Every active directional stage performs one exchange followed by one
        # no-exchange sweep. This forces all three outer rounds to execute.
        forward_results = [
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        backward_results = [
            True,
            False,
            True,
            False,
        ]
        calls: list[str] = []

        def fake_forward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts

            calls.append("forward")
            updated = tracks.copy()
            updated[0, 0] += 1

            assert forward_results, "unexpected additional forward sweep"
            return updated, forward_results.pop(0)

        def fake_backward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts

            calls.append("backward")
            updated = tracks.copy()
            updated[0, 0] += 1

            assert backward_results, "unexpected additional backward sweep"
            return updated, backward_results.pop(0)

        monkeypatch.setattr(
            linker,
            "_run_forward_mge_iteration",
            fake_forward,
        )
        monkeypatch.setattr(
            linker,
            "_run_backward_mge_iteration",
            fake_backward,
        )

        result = linker._run_directional_mge(
            np.zeros((1, 4), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            4,
        )

        assert calls == [
            # Outer round 1
            "forward",
            "forward",
            "backward",
            "backward",
            # Outer round 2
            "forward",
            "forward",
            "backward",
            "backward",
            # Outer round 3: forward only
            "forward",
            "forward",
        ]
        assert forward_results == []
        assert backward_results == []
        assert result[0, 0] == len(calls)

    def test_inactive_directions_stop_before_iteration_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MGE stops naturally once both directional stages are inactive."""
        linker = HodgesLinker(
            mge_max_iterations=3,
            dmax_zones=np.zeros((0, 5), dtype=np.float64),
            adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        )
        calls: list[str] = []

        def fake_forward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts
            calls.append("forward")
            return tracks, False

        def fake_backward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
            missing_input_counts: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames, missing_input_counts
            calls.append("backward")
            return tracks, False

        monkeypatch.setattr(
            linker,
            "_run_forward_mge_iteration",
            fake_forward,
        )
        monkeypatch.setattr(
            linker,
            "_run_backward_mge_iteration",
            fake_backward,
        )

        linker._run_directional_mge(
            np.zeros((1, 4), dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            4,
        )

        assert calls == ["forward", "backward"]
