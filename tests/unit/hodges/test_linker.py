from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.hodges.constants import MAX_ITERATIONS_DEFAULT
from pystormtracker.hodges.linker import HodgesLinker
from pystormtracker.models.tracker import RawDetectionStep


def _make_detections(
    n_frames: int,
    *,
    points_per_frame: int = 1,
    offset_per_step: float = 1.0,
) -> list[RawDetectionStep]:
    """Create a list of RawDetectionStep objects for testing."""
    detections: list[RawDetectionStep] = []
    for i in range(n_frames):
        t = np.datetime64("2000-01-01T00:00:00") + np.timedelta64(i, "h")
        offsets = [float(j) + offset_per_step * i for j in range(points_per_frame)]
        lats = np.array(offsets)
        lons = np.array(offsets)
        vals = np.array([1000.0 + offset_per_step * i for _ in range(points_per_frame)])
        detections.append(RawDetectionStep(t, lats, lons, vals))
    return detections


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

    def test_reaching_max_iterations_does_not_raise(self) -> None:
        """Reaching max_iterations is a normal termination path."""
        detections = _make_detections(10)
        linker = HodgesLinker(max_iterations=3)
        # Should not raise even with limited iterations
        tracks = linker.link(
            detections,
            primary_var="msl",
            mode="min",
        )
        assert tracks is not None


class TestDirectionalMGE:
    """Tests for the directional MGE control flow."""

    def test_single_frame_produces_single_point_track(self) -> None:
        """A single frame with one detection produces a track with one point."""
        detections = _make_detections(1, points_per_frame=1)
        linker = HodgesLinker()
        tracks = linker.link(
            detections,
            primary_var="msl",
            mode="min",
        )
        assert len(tracks) == 1
        assert len(tracks[0]) == 1

    def test_two_frames_produces_tracks(self) -> None:
        detections = _make_detections(2, points_per_frame=2, offset_per_step=1.0)
        linker = HodgesLinker()
        tracks = linker.link(
            detections,
            primary_var="msl",
            mode="min",
        )
        assert len(tracks) >= 1

    def test_deterministic_output(self) -> None:
        """Same input produces same output."""
        detections = _make_detections(5, points_per_frame=1, offset_per_step=1.0)
        linker = HodgesLinker(max_iterations=3)
        tracks1 = linker.link(
            detections,
            primary_var="msl",
            mode="min",
        )
        tracks2 = linker.link(
            detections,
            primary_var="msl",
            mode="min",
        )
        assert len(tracks1) == len(tracks2)
        for t1, t2 in zip(tracks1, tracks2, strict=True):
            np.testing.assert_array_equal(t1.lats, t2.lats)
            np.testing.assert_array_equal(t1.lons, t2.lons)
