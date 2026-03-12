import numpy as np

from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.linker import SimpleLinker


def test_simple_linker_init() -> None:
    linker = SimpleLinker(threshold=1000.0)
    assert linker.threshold == 1000.0


def test_simple_linker_append_center() -> None:
    linker = SimpleLinker()
    tracks = Tracks()

    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 1000})
    linker.append_center(tracks, [c1])

    assert len(tracks) == 1
    assert tracks.time_range is not None
    assert tracks.time_range.start == t0
    assert tracks.time_range.end == t0

    t6 = np.datetime64("2025-12-01T06:00:00")
    c2 = Center(t6, 1, 1, {"msl": 990})
    linker.append_center(tracks, [c2])

    assert len(tracks) == 1
    assert tracks.time_range.end == t6
    assert tracks.time_range.step == np.timedelta64(6, "h")
    assert len(tracks[0]) == 2


def test_simple_linker_extend_track() -> None:
    linker = SimpleLinker()
    t1 = Tracks()
    t2 = Tracks()

    t0 = np.datetime64("2025-12-01T00:00:00")
    c1 = Center(t0, 0, 0, {"msl": 1000})
    linker.append_center(t1, [c1])

    t6 = np.datetime64("2025-12-01T06:00:00")
    c2 = Center(t6, 1, 1, {"msl": 990})
    linker.append_center(t2, [c2])

    linker.extend_track(t1, t2)

    assert len(t1) == 1
    assert len(t1[0]) == 2
    assert t1.time_range is not None
    assert t1.time_range.end == t6
