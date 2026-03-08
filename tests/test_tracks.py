from pathlib import Path

import pytest

from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Track, Tracks


def test_tracks_init() -> None:
    t = Tracks()
    assert len(t) == 0
    assert t.head == []
    assert t.tail == []
    assert t.time_range is None


def test_tracks_append_and_access() -> None:
    t = Tracks()
    c1 = Center(0, 0, 0, 0)
    c2 = Center(1, 1, 1, 1)

    t.append(Track([c1]))
    assert len(t) == 1
    assert t[0] == Track([c1])

    t.append(Track([c2]))
    assert len(t) == 2
    assert t[1] == Track([c2])

    # Test __setitem__
    t[0] = Track([c1, c2])
    assert t[0] == Track([c1, c2])


def test_tracks_iterator() -> None:
    t = Tracks()
    c1 = Center(0, 0, 0, 0)
    c2 = Center(1, 1, 1, 1)
    t.append(Track([c1]))
    t.append(Track([c2]))

    collected = list(t)
    assert collected == [Track([c1]), Track([c2])]


def test_tracks_imilast_io(tmp_path: Path) -> None:
    """Test round-trip serialization to IMILAST format."""
    t = Tracks()
    # Create two tracks
    # Track 1: 2 points
    c1 = Center(2025120100, 10.0, 20.0, 1000.0)
    c2 = Center(2025120106, 11.0, 21.0, 990.0)
    t.append(Track([c1, c2]))
    # Track 2: 1 point
    c3 = Center(2025120100, -10.0, -20.0, 1010.0)
    t.append(Track([c3]))

    out_file = tmp_path / "test_io.txt"
    t.to_imilast(str(out_file))

    # Read back
    t2 = Tracks.from_imilast(out_file)

    assert len(t2) == 2

    # Sort original tracks to match from_imilast sorting
    t._tracks.sort(key=lambda tr: (tr[0].time, tr[0].lat, tr[0].lon))

    # Check that we can compare them
    t.compare(t2)


def test_tracks_compare_mismatch() -> None:
    """Test that compare raises AssertionError on mismatch."""
    t1 = Tracks()
    t1.append(Track([Center(0, 0, 0, 0)]))

    t2 = Tracks()
    t2.append(Track([Center(0, 0, 0, 10)]))  # different var

    with pytest.raises(AssertionError):
        t1.compare(t2)

    t3 = Tracks()
    t3.append(Track([Center(0, 0, 0, 0), Center(1, 1, 1, 1)]))  # longer track
    with pytest.raises(AssertionError):
        t1.compare(t3)
