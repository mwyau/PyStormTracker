from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.models.tracker import RawDetectionStep
from pystormtracker.models.tracks import TracksMetadata, _TracksBuilder
from pystormtracker.simple.linker import SimpleLinker, great_circle_distance_matrix


def test_simple_linker_init() -> None:
    linker = SimpleLinker(threshold=1000.0)
    assert linker.threshold == 1000.0


def test_simple_linker_append() -> None:
    linker = SimpleLinker()
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))

    t0 = np.datetime64("2025-12-01T00:00:00")
    lats_1: NDArray[np.float64] = np.array([0.0])
    lons_1: NDArray[np.float64] = np.array([0.0])
    vars_1 = np.array([1000.0], dtype=np.float64)
    step_data_1 = RawDetectionStep(t0, lats_1, lons_1, vars_1)
    linker.append(builder, step_data_1)

    t6 = np.datetime64("2025-12-01T06:00:00")
    lats_2: NDArray[np.float64] = np.array([1.0])
    lons_2: NDArray[np.float64] = np.array([1.0])
    vars_2 = np.array([990.0], dtype=np.float64)
    step_data_2 = RawDetectionStep(t6, lats_2, lons_2, vars_2)
    linker.append(builder, step_data_2)
    tracks = builder.finish()

    assert len(tracks) == 1
    assert len(tracks[0]) == 2


def _step(
    time: str,
    lats: list[float],
    lons: list[float],
    values: list[float],
) -> RawDetectionStep:
    return RawDetectionStep(
        np.datetime64(time),
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


def _builder() -> _TracksBuilder:
    return _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))


def _packed_time(value: str) -> int:
    return int(np.datetime64(value, "ms").astype(np.int64))


def test_first_frame_creates_all_active_tails() -> None:
    linker = SimpleLinker()
    builder = _builder()

    linker.append(
        builder, _step("2025-12-01T00:00", [0.0, 10.0], [0.0, 10.0], [1.0, 2.0])
    )

    assert linker._tail_ids == {1, 2}
    tracks = builder.finish()
    np.testing.assert_array_equal(tracks.ids, [1, 2])
    np.testing.assert_array_equal(tracks.offsets, [0, 1, 2])


def test_empty_frame_clears_all_active_tails() -> None:
    linker = SimpleLinker()
    builder = _builder()
    linker.append(
        builder, _step("2025-12-01T00:00", [0.0, 10.0], [0.0, 10.0], [1.0, 2.0])
    )

    linker.append(builder, _step("2025-12-01T06:00", [], [], []))

    assert linker._tail_ids == set()
    tracks = builder.finish()
    np.testing.assert_array_equal(tracks.ids, [1, 2])
    np.testing.assert_array_equal(tracks.offsets, [0, 1, 2])


def test_detection_after_empty_frame_starts_and_keeps_only_new_track() -> None:
    linker = SimpleLinker()
    builder = _builder()
    linker.append(
        builder, _step("2025-12-01T00:00", [0.0, 10.0], [0.0, 10.0], [1.0, 2.0])
    )
    linker.append(builder, _step("2025-12-01T06:00", [], [], []))
    linker.append(builder, _step("2025-12-01T12:00", [0.0], [0.0], [3.0]))

    assert linker._tail_ids == {3}

    # This point is close to the old frame-0 endpoint. It must extend track 3,
    # proving that historical tracks are not reactivated as active tails.
    linker.append(builder, _step("2025-12-01T18:00", [0.0], [0.0], [4.0]))
    tracks = builder.finish()

    np.testing.assert_array_equal(tracks.ids, [1, 2, 3])
    np.testing.assert_array_equal(tracks.offsets, [0, 1, 2, 4])
    np.testing.assert_array_equal(
        tracks.times,
        [
            _packed_time("2025-12-01T00:00"),
            _packed_time("2025-12-01T00:00"),
            _packed_time("2025-12-01T12:00"),
            _packed_time("2025-12-01T18:00"),
        ],
    )
    np.testing.assert_array_equal(tracks.lats, [0.0, 10.0, 0.0, 0.0])
    np.testing.assert_array_equal(tracks.lons, [0.0, 10.0, 0.0, 0.0])
    np.testing.assert_array_equal(tracks.variables["msl"], [1.0, 2.0, 3.0, 4.0])


def test_temporal_gap_starts_only_new_active_tracks() -> None:
    linker = SimpleLinker()
    builder = _builder()
    linker.append(
        builder, _step("2025-12-01T00:00", [0.0, 10.0], [0.0, 10.0], [1.0, 2.0])
    )
    linker.append(
        builder, _step("2025-12-01T06:00", [0.0, 10.0], [0.0, 10.0], [3.0, 4.0])
    )

    # The 12-hour jump is larger than the learned six-hour step.
    linker.append(builder, _step("2025-12-01T18:00", [0.0], [0.0], [5.0]))
    assert linker._tail_ids == {3}
    linker.append(builder, _step("2025-12-02T00:00", [0.0], [0.0], [6.0]))

    tracks = builder.finish()
    np.testing.assert_array_equal(tracks.ids, [1, 2, 3])
    np.testing.assert_array_equal(tracks.offsets, [0, 2, 4, 6])
    np.testing.assert_array_equal(
        tracks.times,
        [
            _packed_time("2025-12-01T00:00"),
            _packed_time("2025-12-01T06:00"),
            _packed_time("2025-12-01T00:00"),
            _packed_time("2025-12-01T06:00"),
            _packed_time("2025-12-01T18:00"),
            _packed_time("2025-12-02T00:00"),
        ],
    )
    np.testing.assert_array_equal(
        tracks.variables["msl"], [1.0, 3.0, 2.0, 4.0, 5.0, 6.0]
    )


def test_consecutive_frames_keep_matching_and_deterministic_ids() -> None:
    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        linker = SimpleLinker()
        builder = _builder()
        linker.append(
            builder,
            _step("2025-12-01T00:00", [10.0, 0.0], [10.0, 0.0], [2.0, 1.0]),
        )
        linker.append(
            builder,
            _step("2025-12-01T06:00", [10.2, 0.2], [10.2, 0.2], [4.0, 3.0]),
        )
        tracks = builder.finish()
        return tracks.ids, tracks.offsets, tracks.variables["msl"]

    first = run()
    second = run()
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first[0], [1, 2])
    np.testing.assert_array_equal(first[1], [0, 2, 4])
    np.testing.assert_array_equal(first[2], [1.0, 3.0, 2.0, 4.0])


def test_great_circle_distance_crosses_dateline() -> None:
    distances = great_circle_distance_matrix(
        np.array([0.0]),
        np.array([179.0]),
        np.array([0.0]),
        np.array([-179.0]),
    )

    assert distances.shape == (1, 1)
    assert distances[0, 0] == pytest.approx(222.39, rel=1e-3)


def test_great_circle_distance_clamps_identical_points() -> None:
    distances = great_circle_distance_matrix(
        np.array([90.0]),
        np.array([0.0]),
        np.array([90.0]),
        np.array([180.0]),
    )

    assert np.isfinite(distances).all()
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
