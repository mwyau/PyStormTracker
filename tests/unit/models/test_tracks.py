from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pystormtracker.io.imilast import read_imilast, write_imilast
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


def test_tracks_init() -> None:
    t = Tracks()
    assert len(t) == 0
    assert t.head == []
    assert t.tail == []
    assert t.time_range is None


def test_tracks_append_and_access() -> None:
    t = Tracks()
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})
    c2 = Center(t0, 1, 1, {"msl": 1})

    tr1 = t.add_track([c1])
    assert len(t) == 1
    assert t[0] == tr1

    tr2 = t.add_track([c2])
    assert len(t) == 2
    assert t[1] == tr2

    # Test __setitem__ with a track from the same Tracks object
    tr_new_same = t.add_track([c1, c2])
    t[0] = tr_new_same
    assert t[0].track_id == tr_new_same.track_id

    # Test __setitem__ with a track from a different Tracks object
    # Note: we need another Tracks object to get a properly baked Track view
    # to replace into the first Tracks object.
    t2 = Tracks()
    tr_new = t2.add_track([c1, c2])

    # Replaces the first track. The new track is appended at the end.
    t[0] = tr_new
    assert len(t[-1]) == 2
    assert t[-1][0].lat == c1.lat
    assert t[-1][1].lat == c2.lat


def test_tracks_head_tail() -> None:
    t = Tracks()
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})
    tr = t.add_track([c1])

    t.head = [tr]
    assert len(t.head) == 1
    assert t.head[0].track_id == tr.track_id

    t.tail = [tr]
    assert len(t.tail) == 1
    assert t.tail[0].track_id == tr.track_id


def test_tracks_init_with_data() -> None:
    t_id: NDArray[np.int64] = np.array([1])
    t_time: NDArray[np.datetime64] = np.array(
        ["2025-12-01T00:00:00"], dtype="datetime64[s]"
    )
    t_lat: NDArray[np.float64] = np.array([0.0])
    t_lon: NDArray[np.float64] = np.array([0.0])
    t_vars: dict[str, NDArray[np.float64]] = {"msl": np.array([0.0])}

    t = Tracks(track_ids=t_id, times=t_time, lats=t_lat, lons=t_lon, vars_dict=t_vars)
    assert len(t) == 1
    assert t[0][0].vars["msl"] == 0.0


def test_tracks_iterator() -> None:
    t = Tracks()
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})
    c2 = Center(t0, 1, 1, {"msl": 1})
    tr1 = t.add_track([c1])
    tr2 = t.add_track([c2])

    collected = list(t)
    assert collected == [tr1, tr2]


def test_tracks_imilast_io(tmp_path: Path) -> None:
    """Test round-trip serialization to IMILAST format."""
    t = Tracks()
    # Create two tracks with standardized datetime64 times
    t1_time = np.datetime64("2025-12-01T00:00:00")
    t1_time2 = np.datetime64("2025-12-01T06:00:00")

    c1 = Center(t1_time, 10.0, 20.0, {"Intensity1": 1000.0})
    c2 = Center(t1_time2, 11.0, 21.0, {"Intensity1": 990.0})
    t.add_track([c1, c2])

    # Track 2: 1 point
    c3 = Center(t1_time, -10.0, -20.0, {"Intensity1": 1010.0})
    t.add_track([c3])

    out_file = tmp_path / "test_io.txt"
    write_imilast(t, out_file)

    # Read back
    t2 = read_imilast(out_file)

    assert len(t2) == 2

    # Check that we can compare them (compare handles sorting internally)
    t.compare(t2)


def test_track_methods() -> None:
    t = Tracks()
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})
    c2 = Center(t0 + np.timedelta64(6, "h"), 1, 1, {"msl": 1})

    track = t.add_track([c1])
    assert len(track) == 1
    assert track[0].lat == c1.lat

    # iter
    assert [c.lat for c in track] == [c1.lat]

    # append
    track.append(c2)
    assert len(track) == 2
    assert track[1].lat == c2.lat

    # extend
    t2 = Tracks()
    track2 = t2.add_track([Center(t0 + np.timedelta64(12, "h"), 2, 2, {"msl": 2})])
    track.extend(track2)
    assert len(track) == 3
    assert track[2].vars["msl"] == 2

    # abs_dist (distance between last point of track and another center)
    c3 = Center(t0 + np.timedelta64(18, "h"), 0, 1, {"msl": 0})
    # Haversine distance from track last point (2,2) to target center (0,1)
    dist = track.abs_dist(c3)
    assert dist > 240
    assert dist < 260


def test_tracks_sort() -> None:
    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = t0 + np.timedelta64(6, "h")

    tracks = Tracks()
    tr1 = tracks.add_track([Center(t1, 0, 0, {"msl": 0})])
    tr2 = tracks.add_track([Center(t0, 10, 10, {"msl": 0})])
    tr3 = tracks.add_track([Center(t0, 5, 5, {"msl": 0})])

    tracks.sort()

    # Should be sorted by time, then lat, then lon
    assert tracks[0] == tr3  # t0, lat 5
    assert tracks[1] == tr2  # t0, lat 10
    assert tracks[2] == tr1  # t1, lat 0


def test_tracks_equality() -> None:
    t1 = Tracks()
    t2 = Tracks()
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})

    t1.add_track([c1])
    t2.add_track([c1])

    # __eq__ testing is only defined on Track, not Tracks
    assert t1[0] == t2[0]

    # Not eq type
    assert t1[0] != "not a track"

    # Not eq len
    t2[0].append(c1)
    assert t1[0] != t2[0]

    # Not eq elements
    t3 = Tracks()
    t3.add_track([Center(t0, 1, 1, {"msl": 1})])
    assert t1[0] != t3[0]


def test_tracks_empty_track() -> None:
    t = Tracks()
    tr = t.add_track([])
    assert len(tr) == 0
    assert tr.track_id == 1  # First ID is 1

    # Test extending from self vs other
    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 0})
    tr.append(c1)

    t2 = Tracks()
    t2.add_track([c1])

    # Test extend onto a different track in the same Tracks object
    tr2_same = t.add_track([c1])
    tr.extend(tr2_same)
    assert len(tr) == 2


def test_tracks_init_no_vars() -> None:
    t_id: NDArray[np.int64] = np.array([1])
    t_time: NDArray[np.datetime64] = np.array(
        ["2025-12-01T00:00:00"], dtype="datetime64[s]"
    )
    t_lat: NDArray[np.float64] = np.array([0.0])
    t_lon: NDArray[np.float64] = np.array([0.0])

    t = Tracks(track_ids=t_id, times=t_time, lats=t_lat, lons=t_lon)
    assert len(t) == 1
    assert len(t.vars) == 0


def test_tracks_bulk_append() -> None:
    t = Tracks()
    t_id: NDArray[np.int64] = np.array([1, 2])
    t_time: NDArray[np.datetime64] = np.array(
        ["2025-12-01T00:00:00", "2025-12-01T00:00:00"], dtype="datetime64[s]"
    )
    t_lat: NDArray[np.float64] = np.array([0.0, 1.0])
    t_lon: NDArray[np.float64] = np.array([0.0, 1.0])
    t_vars: dict[str, NDArray[np.float64]] = {"msl": np.array([0.0, 1.0])}

    t.bulk_append(t_id, t_time, t_lat, t_lon, t_vars)
    assert len(t) == 2
    assert t[1][0].vars["msl"] == 1.0

    # Bulk append with missing var to test NaN filling
    t_id2: NDArray[np.int64] = np.array([3])
    t_time2: NDArray[np.datetime64] = np.array(
        ["2025-12-01T06:00:00"], dtype="datetime64[s]"
    )
    t_lat2: NDArray[np.float64] = np.array([2.0])
    t_lon2: NDArray[np.float64] = np.array([2.0])
    t_vars2: dict[str, NDArray[np.float64]] = {"new_var": np.array([2.0])}

    t.bulk_append(t_id2, t_time2, t_lat2, t_lon2, t_vars2)
    assert len(t) == 3
    assert np.isnan(t[2][0].vars["msl"])
    assert t[2][0].vars["new_var"] == 2.0
    assert np.isnan(t[0][0].vars["new_var"])
