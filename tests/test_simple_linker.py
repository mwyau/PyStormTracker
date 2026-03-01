import pytest
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.linker import SimpleLinker

def test_simple_linker_append_center():
    linker = SimpleLinker(threshold=1000.0)
    tracks = Tracks()
    
    # First time step
    c1 = Center(time=0, lat=0, lon=0, var=1000)
    c2 = Center(time=0, lat=10, lon=10, var=1000)
    linker.append_center(tracks, [c1, c2])
    
    assert len(tracks) == 2
    assert tracks.head == [0, 1]
    assert tracks.tail == [0, 1]
    assert tracks.tstart == 0
    assert tracks.tend == 0
    
    # Second time step - c3 close to c1, c4 far from everything
    c3 = Center(time=1, lat=0.1, lon=0.1, var=990)
    c4 = Center(time=1, lat=20, lon=20, var=990)
    linker.append_center(tracks, [c3, c4])
    
    assert len(tracks) == 3 # track 0 extended, track 1 ended, track 2 new
    assert tracks[0] == [c1, c3]
    assert tracks[1] == [c2]
    assert tracks[2] == [c4]
    assert tracks.tail == [0, 2]
    assert tracks.tend == 1
    assert tracks.dt == 1

def test_simple_linker_extend_track():
    linker = SimpleLinker(threshold=1000.0)
    
    tracks1 = Tracks()
    c1 = Center(time=0, lat=0, lon=0, var=1000)
    linker.append_center(tracks1, [c1])
    
    tracks2 = Tracks()
    c2 = Center(time=1, lat=0.1, lon=0.1, var=990)
    # Need to set tstart/tend/dt for tracks2 manually if not using linker.append_center
    linker.append_center(tracks2, [c2])
    
    linker.extend_track(tracks1, tracks2)
    
    assert len(tracks1) == 1
    assert tracks1[0] == [c1, c2]
    assert tracks1.tail == [0]
    assert tracks1.tend == 1

def test_simple_linker_nearest_neighbor():
    linker = SimpleLinker(threshold=1000.0)
    tracks = Tracks()
    
    # Time 0
    c1 = Center(time=0, lat=0, lon=0, var=1000)
    c2 = Center(time=0, lat=10, lon=10, var=1000)
    linker.append_center(tracks, [c1, c2])
    
    # Time 1: c3 is close to c1, c4 is close to c2
    # But c3 is EXTREMELY close to c1
    c3 = Center(time=1, lat=0.01, lon=0.01, var=990)
    c4 = Center(time=1, lat=10.01, lon=10.01, var=990)
    
    linker.append_center(tracks, [c3, c4])
    
    assert tracks[0] == [c1, c3]
    assert tracks[1] == [c2, c4]

def test_simple_linker_max_distance():
    linker = SimpleLinker(threshold=10.0) # Very small threshold
    tracks = Tracks()
    
    c1 = Center(time=0, lat=0, lon=0, var=1000)
    linker.append_center(tracks, [c1])
    
    # c2 is far away (more than 10km)
    c2 = Center(time=1, lat=1, lon=1, var=990)
    linker.append_center(tracks, [c2])
    
    assert len(tracks) == 2
    assert tracks[0] == [c1]
    assert tracks[1] == [c2]
