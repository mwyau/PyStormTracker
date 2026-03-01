import math
import pytest
from pystormtracker.models.center import Center

def test_center_init():
    c = Center(time=100.0, lat=10.0, lon=20.0, var=1013.25)
    assert c.time == 100.0
    assert c.lat == 10.0
    assert c.lon == 20.0
    assert c.var == 1013.25

def test_center_repr_str():
    c = Center(time=100.0, lat=10.0, lon=20.0, var=1013.25)
    assert repr(c) == "1013.25"
    assert str(c) == "[time=100.0, lat=10.0, lon=20.0, var=1013.25]"

def test_abs_dist():
    # R = 6367.0
    c1 = Center(time=0, lat=0, lon=0, var=0)
    c2 = Center(time=0, lat=0, lon=1, var=0)
    
    # Distance for 1 degree longitude at equator: R * (1 * pi/180)
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.abs_dist(c2) == pytest.approx(expected, rel=1e-5)

def test_abs_dist_type_error():
    c1 = Center(time=0, lat=0, lon=0, var=0)
    with pytest.raises(TypeError, match="must be compared with a Center object"):
        c1.abs_dist("not a center")

def test_lat_dist():
    c1 = Center(time=0, lat=0, lon=0, var=0)
    c2 = Center(time=0, lat=1, lon=0, var=0)
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.lat_dist(c2) == pytest.approx(expected, rel=1e-5)

def test_lon_dist():
    # At equator
    c1 = Center(time=0, lat=0, lon=0, var=0)
    c2 = Center(time=0, lat=0, lon=1, var=0)
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.lon_dist(c2) == pytest.approx(expected, rel=1e-5)
    
    # At 60 degrees latitude, cos(60) = 0.5
    c3 = Center(time=0, lat=60, lon=0, var=0)
    c4 = Center(time=0, lat=60, lon=1, var=0)
    expected_60 = 6367.0 * (math.pi / 180.0) * 0.5
    assert c3.lon_dist(c4) == pytest.approx(expected_60, rel=1e-5)
