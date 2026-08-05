from __future__ import annotations

import numpy as np

import pystormtracker as pst
from pystormtracker.models.time import TimePoint
from pystormtracker.models.tracker import RawDetectionStep


def accepts_tracker(tracker: pst.Tracker) -> pst.Tracker:
    return tracker


def test_public_api_exports() -> None:
    assert sorted(pst.__all__) == [
        "HealpixTracker",
        "HodgesTracker",
        "SimpleTracker",
        "Track",
        "Tracker",
        "Tracks",
    ]
    assert pst.SimpleTracker is not None
    assert pst.HodgesTracker is not None
    assert pst.HealpixTracker is not None
    assert pst.Tracker is not None
    assert pst.Track is not None
    assert pst.Tracks is not None


def test_protocol_conformance() -> None:
    simple = pst.SimpleTracker()
    hodges = pst.HodgesTracker()
    healpix = pst.HealpixTracker()

    assert accepts_tracker(simple) is simple
    assert accepts_tracker(hodges) is hodges
    assert accepts_tracker(healpix) is healpix


def test_raw_detection_step_named_tuple() -> None:
    time_val: TimePoint = 1735689600000
    lats = np.array([10.0, 20.0], dtype=np.float64)
    lons = np.array([30.0, 40.0], dtype=np.float64)
    vals = np.array([100.0, 200.0], dtype=np.float64)

    step = RawDetectionStep(time_val, lats, lons, vals)

    # Test tuple unpacking
    t, la, lo, v = step
    assert t == time_val
    np.testing.assert_array_equal(la, lats)
    np.testing.assert_array_equal(lo, lons)
    np.testing.assert_array_equal(v, vals)

    # Test named access
    assert step.time == time_val
    np.testing.assert_array_equal(step.latitudes, lats)
    np.testing.assert_array_equal(step.longitudes, lons)
    np.testing.assert_array_equal(step.values, vals)
