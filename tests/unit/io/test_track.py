from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pystormtracker.io.track import read_track, write_track
from pystormtracker.metrics.compare import compare_tracks
from pystormtracker.models.tracks import TracksMetadata, _TracksBuilder


def _write_two_point_track(tmp_path: Path) -> Path:
    path = tmp_path / "frames.track"
    path.write_text(
        "TRACK_NUM 1\nTRACK_ID 1\nPOINT_NUM 2\n1 0.0 0.0 1000.0\n2 1.0 0.0 999.0\n",
        encoding="utf-8",
    )
    return path


def test_read_track_rejects_ambiguous_numeric_time(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Ambiguous numeric TRACK point time"):
        read_track(_write_two_point_track(tmp_path))


def test_read_track_maps_track_frame_indices_to_exact_source_times(
    tmp_path: Path,
) -> None:
    source_times = np.array(
        [np.datetime64("2024-01-01T00:00"), np.datetime64("2024-01-01T06:00")]
    )
    loaded = read_track(
        _write_two_point_track(tmp_path),
        track_numeric_time="frame_index",
        track_frame_times=source_times,
    )

    np.testing.assert_array_equal(
        loaded.times,
        np.array([1704067200000, 1704088800000], dtype=np.int64),
    )
    match = compare_tracks(loaded, loaded).matches[0]
    assert match.reference.duration_hours == 6.0
    assert match.reference.mean_speed_kmh == pytest.approx(18.53, rel=1.0e-3)


def test_read_track_accepts_unix_seconds_only_when_explicit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seconds.track"
    path.write_text(
        "TRACK_NUM 1\nTRACK_ID 1\nPOINT_NUM 1\n1704067200 0.0 0.0 1.0\n",
        encoding="utf-8",
    )
    loaded = read_track(path, track_numeric_time="unix_seconds")
    assert loaded.times.tolist() == [1704067200000]


def test_write_track_roundtrip(tmp_path: Path) -> None:
    source_times = np.array(
        [np.datetime64("2024-01-01T00:00"), np.datetime64("2024-01-01T06:00")]
    )
    original = read_track(
        _write_two_point_track(tmp_path),
        track_numeric_time="frame_index",
        track_frame_times=source_times,
    )
    out_path = tmp_path / "roundtrip.track"
    write_track(original, out_path)
    reloaded = read_track(out_path)
    assert len(reloaded) == len(original)
    np.testing.assert_array_equal(reloaded.times, original.times)
    np.testing.assert_allclose(reloaded.lats, original.lats)
    np.testing.assert_allclose(reloaded.lons, original.lons)


def test_write_track_rejects_subhour_timestamps(tmp_path: Path) -> None:
    source_times = np.array([np.datetime64("2024-01-01T00:30")])
    builder = _TracksBuilder(TracksMetadata("Intensity1", "max", {"Intensity1": "1"}))
    builder.add_track(1, source_times, [0.0], [0.0], {"Intensity1": [1.0]})
    with pytest.raises(ValueError, match="whole-hour timestamps.*unsupported cadence"):
        write_track(builder.finish(), tmp_path / "rejected.track")
