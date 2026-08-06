from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.hodges.constants import MAX_ITERATIONS_DEFAULT
from pystormtracker.hodges.linker import HodgesLinker
from pystormtracker.models.tracker import RawDetectionStep


def test_hodges_linker_init() -> None:
    linker = HodgesLinker(w1=0.5, w2=0.5, dmax=10.0)
    assert linker.w1 == 0.5
    assert linker.w2 == 0.5
    assert linker.dmax == 10.0


def test_hodges_linker_preserves_zero_and_one_frame_inputs() -> None:
    linker = HodgesLinker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
    )
    empty = linker.link([], primary_var="msl", mode="min")
    assert len(empty) == 0

    one_frame_detections: list[RawDetectionStep] = [
        RawDetectionStep(
            time=np.datetime64("2025-12-01T00:00:00"),
            latitudes=np.array([10.0]),
            longitudes=np.array([20.0]),
            values=np.array([1000.0]),
        )
    ]
    one_frame = linker.link(
        one_frame_detections,
        primary_var="msl",
        mode="min",
    )
    assert len(one_frame) == 1
    assert len(one_frame[0]) == 1
    assert one_frame[0][0].lat == 10.0


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

    detections: list[RawDetectionStep] = [
        RawDetectionStep(
            t0, np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.array([1000.0, 1000.0])
        ),
        RawDetectionStep(
            t1, np.array([0.0, 10.0]), np.array([1.0, 11.0]), np.array([990.0, 990.0])
        ),
        RawDetectionStep(
            t2, np.array([0.0, 10.0]), np.array([2.0, 12.0]), np.array([980.0, 980.0])
        ),
    ]

    tracks = linker.link(detections, primary_var="msl", mode="min")

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
    detections: list[RawDetectionStep] = [
        RawDetectionStep(
            t0, np.array([0.0, 0.0]), np.array([0.0, 10.0]), np.array([1000.0, 1000.0])
        ),
        RawDetectionStep(
            t1,
            np.array([5.0, 5.0001]),
            np.array([5.0, 5.0001]),
            np.array([990.0, 990.0]),
        ),
        RawDetectionStep(
            t2, np.array([10.0, 10.0]), np.array([10.0, 0.0]), np.array([980.0, 980.0])
        ),
    ]

    tracks = linker.link(detections, primary_var="msl", mode="min")

    assert len(tracks) == 2
    # One track should go from (0,0) to (10,10)
    found_a = False
    for tr in tracks:
        if tr[0].lat == 0.0 and tr[0].lon == 0.0 and tr[2].lat == 10.0:
            found_a = True
            break
    assert found_a


class TestMaxIterations:
    """Tests for the max_iterations parameter."""

    def test_default_max_iterations_is_three(self) -> None:
        linker = HodgesLinker()
        assert linker.max_iterations == 3
        assert MAX_ITERATIONS_DEFAULT == 3

    def test_nonpositive_max_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            HodgesLinker(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            HodgesLinker(max_iterations=-1)

    def test_positive_max_iterations_accepted(self) -> None:
        for n in (1, 2, 3, 10, 100):
            linker = HodgesLinker(max_iterations=n)
            assert linker.max_iterations == n


class TestDirectionalMGE:
    """Tests for TRACK-style directional MGE scheduling."""

    def test_each_direction_converges_before_switching(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A direction repeats until a complete sweep makes no exchange."""
        linker = HodgesLinker(max_iterations=3)

        forward_results = [True, True, False, False]
        backward_results = [True, False]
        calls: list[str] = []

        def fake_forward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames

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
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames

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
        linker = HodgesLinker(max_iterations=3)

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
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames

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
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames

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
        linker = HodgesLinker(max_iterations=3)
        calls: list[str] = []

        def fake_forward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames
            calls.append("forward")
            return tracks, False

        def fake_backward(
            tracks: NDArray[np.int64],
            features_lat: NDArray[np.float64],
            features_lon: NDArray[np.float64],
            n_frames: int,
        ) -> tuple[NDArray[np.int64], bool]:
            del features_lat, features_lon, n_frames
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

        initial = np.zeros((1, 4), dtype=np.int64)
        result = linker._run_directional_mge(
            initial,
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            4,
        )

        assert calls == ["forward", "backward"]
        np.testing.assert_array_equal(result, initial)
