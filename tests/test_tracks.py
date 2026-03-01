import pytest
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks

def test_tracks_init():
    t = Tracks()
    assert len(t) == 0
    assert t.head == []
    assert t.tail == []
    assert t.tstart is None
    assert t.tend is None
    assert t.dt is None

def test_tracks_append_and_access():
    t = Tracks()
    c1 = Center(0, 0, 0, 0)
    c2 = Center(1, 1, 1, 1)
    
    t.append([c1])
    assert len(t) == 1
    assert t[0] == [c1]
    
    t.append([c2])
    assert len(t) == 2
    assert t[1] == [c2]
    
    # Test __setitem__
    t[0] = [c1, c2]
    assert t[0] == [c1, c2]

def test_tracks_iterator():
    t = Tracks()
    c1 = [Center(0, 0, 0, 0)]
    c2 = [Center(1, 1, 1, 1)]
    t.append(c1)
    t.append(c2)
    
    collected = list(t)
    assert collected == [c1, c2]
