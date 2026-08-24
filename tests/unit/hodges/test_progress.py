from __future__ import annotations

from io import StringIO

import dask
import numpy as np

from pystormtracker.hodges.detections import HodgesCenterFrame
from pystormtracker.hodges.progress import HodgesDaskProgress


class _Clock:
    """Controllable monotonic clock for deterministic progress accounting tests."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _step(feature_count: int) -> HodgesCenterFrame:
    """Build one minimal completed frame result with a known feature count."""
    return HodgesCenterFrame(
        np.datetime64("2024-01-01T00:00:00"),
        np.zeros(feature_count, dtype=np.float64),
        np.zeros(feature_count, dtype=np.float64),
        np.zeros(feature_count, dtype=np.float64),
    )


def _complete_frame(
    progress: HodgesDaskProgress, index: int, feature_count: int
) -> None:
    """Deliver one Dask posttask callback without running worker tasks."""
    progress._posttask(
        f"hodges-frame-{index:06d}",
        _step(feature_count),
        None,
        None,
        None,
    )


def _pass_through(step: HodgesCenterFrame) -> HodgesCenterFrame:
    """Provide a minimal named segment task for a real local Dask callback."""
    return step


def test_hodges_dask_progress_tracks_out_of_order_frame_frontier() -> None:
    clock = _Clock()
    output = StringIO()
    progress = HodgesDaskProgress(
        total_frames=4,
        total_segments=2,
        frame_workers=2,
        mge_workers=3,
        stream=output,
        clock=clock,
        refresh_seconds=1000.0,
    )
    progress._start(None)

    _complete_frame(progress, 2, 5)
    first = progress.snapshot()
    assert first.completed_frames == 1
    assert first.highest_contiguous_frame == -1
    assert first.detected_features == 5

    clock.advance(1.0)
    _complete_frame(progress, 0, 3)
    _complete_frame(progress, 1, 4)
    middle = progress.snapshot()
    assert middle.completed_frames == 3
    assert middle.highest_contiguous_frame == 2
    assert middle.detected_features == 12
    assert middle.recent_frames_per_second == 2.0
    assert middle.eta_seconds == 0.5

    _complete_frame(progress, 3, 2)
    final = progress.snapshot()
    assert final.completed_frames == 4
    assert final.highest_contiguous_frame == 3
    assert final.detected_features == 14
    assert "Detection/refinement complete" in output.getvalue()
    progress.interrupt()


def test_hodges_dask_progress_counts_only_named_unique_tasks() -> None:
    clock = _Clock()
    progress = HodgesDaskProgress(
        total_frames=3,
        total_segments=2,
        frame_workers=2,
        mge_workers=3,
        stream=StringIO(),
        clock=clock,
    )
    progress._start(None)

    progress._posttask("array-source-0", _step(9), None, None, None)
    progress._posttask("hodges-frame-not-an-index", _step(9), None, None, None)
    progress._posttask("hodges-frame-000003", _step(9), None, None, None)
    progress._posttask("hodges-mge-segment-000000", None, None, None, None)
    progress._posttask("hodges-mge-segment-000000", None, None, None, None)
    progress._posttask("hodges-mge-segment-000001", None, None, None, None)

    snapshot = progress.snapshot()
    assert snapshot.completed_frames == 0
    assert snapshot.detected_features == 0
    assert snapshot.completed_segments == 2
    progress.interrupt()


def test_hodges_dask_progress_reports_interruption_separately() -> None:
    output = StringIO()
    progress = HodgesDaskProgress(
        total_frames=1,
        total_segments=1,
        frame_workers=1,
        mge_workers=1,
        stream=output,
    )
    progress._start(None)

    progress.interrupted()

    text = output.getvalue()
    assert "Hodges tracking interrupted" in text
    assert "failed" not in text


def test_hodges_dask_progress_recognizes_stable_delayed_keys() -> None:
    clock = _Clock()
    progress = HodgesDaskProgress(
        total_frames=1,
        total_segments=1,
        frame_workers=1,
        mge_workers=1,
        stream=StringIO(),
        clock=clock,
    )
    source = dask.delayed(_step)(
        3,
        dask_key_name="source-frame-000000",
    )
    prepared = dask.delayed(_pass_through)(
        source,
        dask_key_name="hodges-prepared-frame-000000",
    )
    frame = dask.delayed(_pass_through)(
        prepared,
        dask_key_name="hodges-frame-000000",
    )
    segment = dask.delayed(_pass_through)(
        frame,
        dask_key_name="hodges-mge-segment-000000",
    )

    with progress:
        dask.compute(  # type: ignore[no-untyped-call]
            segment, scheduler="single-threaded"
        )

    snapshot = progress.snapshot()
    assert snapshot.prepared_frames == 1
    assert snapshot.completed_frames == 1
    assert snapshot.completed_segments == 1
    assert snapshot.detected_features == 3
