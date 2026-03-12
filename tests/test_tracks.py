from pathlib import Path

import numpy as np

from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Track, Tracks
from pystormtracker.io.imilast import read_imilast, write_imilast


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

    # Test __setitem__
    # Note: we need another Tracks object to get a properly baked Track view 
    # to replace into the first Tracks object.
    t2 = Tracks()
    tr_new = t2.add_track([c1, c2])
    
    # Replaces the first track. The new track is appended at the end.
    t[0] = tr_new
    assert len(t[-1]) == 2
    assert t[-1][0].lat == c1.lat
    assert t[-1][1].lat == c2.lat


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
