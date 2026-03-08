import math

import numpy as np
import pytest

from pystormtracker.models.center import Center


def test_center_init() -> None:
    time = np.datetime64("2025-12-01T00:00:00")
    c = Center(time=time, lat=10.0, lon=20.0, var=1013.25)
    assert c.time == time
    assert c.lat == 10.0
    assert c.lon == 20.0
    assert c.var == 1013.25


def test_center_repr_str() -> None:
    time = np.datetime64("2025-12-01T00:00:00")
    c = Center(time=time, lat=10.0, lon=20.0, var=1013.25)
    assert repr(c) == "1013.25"
    assert "lat=10.0" in str(c)


def test_abs_dist() -> None:
    c1 = Center(time=np.datetime64(0, "s"), lat=0, lon=0, var=0)
    c2 = Center(time=np.datetime64(0, "s"), lat=0, lon=1, var=0)
    # 1 degree at equator is approx 111.12 km
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.abs_dist(c2) == pytest.approx(expected, rel=1e-5)


def test_lat_dist() -> None:
    c1 = Center(time=np.datetime64(0, "s"), lat=0, lon=0, var=0)
    c2 = Center(time=np.datetime64(0, "s"), lat=1, lon=0, var=0)
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.lat_dist(c2) == pytest.approx(expected, rel=1e-5)


def test_lon_dist() -> None:
    c1 = Center(time=np.datetime64(0, "s"), lat=0, lon=0, var=0)
    c2 = Center(time=np.datetime64(0, "s"), lat=0, lon=1, var=0)
    expected = 6367.0 * (math.pi / 180.0)
    assert c1.lon_dist(c2) == pytest.approx(expected, rel=1e-5)

    # At 60 degrees latitude, cos(60) = 0.5
    c3 = Center(time=np.datetime64(0, "s"), lat=60, lon=0, var=0)
    c4 = Center(time=np.datetime64(0, "s"), lat=60, lon=1, var=0)
    expected_60 = 6367.0 * (math.pi / 180.0) * 0.5
    assert c3.lon_dist(c4) == pytest.approx(expected_60, rel=1e-5)
