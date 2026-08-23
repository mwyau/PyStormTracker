from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.hodges.segments import (
    TrackingSegment,
    merge_segments,
    plan_tracking_segments,
)
from pystormtracker.models.tracks import TracksMetadata, _TracksBuilder


def test_plan_tracking_segments_monolithic() -> None:
    chunks = plan_tracking_segments(62, None)
    assert len(chunks) == 1
    assert chunks[0] == TrackingSegment(0, 0, 62, 0)

    chunks_large = plan_tracking_segments(62, 100)
    assert len(chunks_large) == 1
    assert chunks_large[0] == TrackingSegment(0, 0, 62, 0)


def test_plan_tracking_segments_non_overlapping() -> None:
    chunks = plan_tracking_segments(140, 60, overlap=0)
    assert len(chunks) == 3
    assert chunks[0] == TrackingSegment(0, 0, 60, 0)
    assert chunks[1] == TrackingSegment(1, 60, 120, 0)
    assert chunks[2] == TrackingSegment(2, 120, 140, 0)


def test_plan_tracking_segments_hodges_track_24_chunks() -> None:
    chunks = plan_tracking_segments(1464, 62, overlap=2)
    assert len(chunks) == 24
    assert chunks[0] == TrackingSegment(0, 0, 62, 0)
    assert chunks[1] == TrackingSegment(1, 59, 123, 2)
    assert chunks[2] == TrackingSegment(2, 120, 184, 2)
    assert chunks[-1] == TrackingSegment(23, 1401, 1464, 2)


def test_plan_tracking_segments_invalid_inputs() -> None:
    assert plan_tracking_segments(0, 62) == []
    assert plan_tracking_segments(-5, 62) == []
    with pytest.raises(ValueError, match="segment_frames must be positive"):
        plan_tracking_segments(100, 0)
    with pytest.raises(ValueError, match="segment_frames must be positive"):
        plan_tracking_segments(100, -10)
    with pytest.raises(ValueError, match="overlap must be nonnegative"):
        plan_tracking_segments(100, 62, overlap=-1)


def test_merge_segments_basic_continuation() -> None:
    meta = TracksMetadata(
        primary_variable="msl",
        mode="min",
        units={"msl": "Pa", "raw_value": "Pa"},
        bounds=None,
        processing=(),
    )
    # Chunk 0: 3 time steps: t=100, 200, 300
    t0 = np.array([100, 200, 300], dtype=np.int64)
    lat0 = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    lon0 = np.array([50.0, 51.0, 52.0], dtype=np.float64)
    val0 = np.array([990.0, 985.0, 980.0], dtype=np.float64)
    raw0 = np.array([992.0, 987.0, 982.0], dtype=np.float64)

    builder0 = _TracksBuilder(meta)
    builder0.new_track(1)
    builder0.extend(1, t0, lat0, lon0, {"msl": val0, "raw_value": raw0})
    chunk0_tracks = builder0.finish()

    # Chunk 1: overlap at t=100, 200, 300 (splice_index=2 -> t=300), then t=400, 500
    t1 = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    lat1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64)
    lon1 = np.array([50.0, 51.0, 52.0, 53.0, 54.0], dtype=np.float64)
    val1 = np.array([990.0, 985.0, 980.0, 975.0, 970.0], dtype=np.float64)
    raw1 = np.array([992.0, 987.0, 982.0, 977.0, 972.0], dtype=np.float64)

    builder1 = _TracksBuilder(meta)
    builder1.new_track(1)
    builder1.extend(1, t1, lat1, lon1, {"msl": val1, "raw_value": raw1})
    chunk1_tracks = builder1.finish()

    plan = [
        TrackingSegment(0, 0, 3, splice_index=0),
        TrackingSegment(1, 0, 5, splice_index=2),
    ]

    spliced = merge_segments([chunk0_tracks, chunk1_tracks], plan)
    assert len(spliced) == 1
    s_tr = spliced[0]
    assert np.array_equal(s_tr.times, np.array([100, 200, 300, 400, 500]))
    assert np.allclose(s_tr.lats, np.array([10.0, 11.0, 12.0, 13.0, 14.0]))
    assert np.allclose(s_tr.lons, np.array([50.0, 51.0, 52.0, 53.0, 54.0]))
    assert np.allclose(
        s_tr.variables["msl"], np.array([990.0, 985.0, 980.0, 975.0, 970.0])
    )
    assert "raw_value" in s_tr.variables
    assert np.allclose(
        s_tr.variables["raw_value"],
        np.array([992.0, 987.0, 982.0, 977.0, 972.0]),
    )
