from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.linker import SimpleLinker


def test_simple_linker_init() -> None:
    linker = SimpleLinker(threshold=1000.0)
    assert linker.threshold == 1000.0


def test_simple_linker_append_center() -> None:
    linker = SimpleLinker()
    tracks = Tracks()

    c1 = Center(0, 0, 0, 1000)
    linker.append_center(tracks, [c1])

    assert len(tracks) == 1
    assert tracks.time_range is not None
    assert tracks.time_range.start == 0
    assert tracks.time_range.end == 0

    c2 = Center(6, 1, 1, 990)
    linker.append_center(tracks, [c2])

    assert len(tracks) == 1
    assert tracks.time_range.end == 6
    assert tracks.time_range.step == 6
    assert len(tracks[0]) == 2


def test_simple_linker_extend_track() -> None:
    linker = SimpleLinker()
    t1 = Tracks()
    t2 = Tracks()

    c1 = Center(0, 0, 0, 1000)
    linker.append_center(t1, [c1])

    c2 = Center(6, 1, 1, 990)
    linker.append_center(t2, [c2])

    linker.extend_track(t1, t2)

    assert len(t1) == 1
    assert len(t1[0]) == 2
    assert t1.time_range is not None
    assert t1.time_range.end == 6
