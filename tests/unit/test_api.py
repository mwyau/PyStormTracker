from __future__ import annotations

import numpy as np

import pystormtracker as pst
from pystormtracker.models.time import TimePoint
from pystormtracker.models.tracker import CenterFrame


def accepts_tracker(tracker: pst.Tracker) -> pst.Tracker:
    return tracker


def test_public_api_exports() -> None:
    assert pst.__all__ == [
        "Center",
        "HealpixTracker",
        "HodgesTracker",
        "SimpleTracker",
        "Track",
        "Tracker",
        "Tracks",
        "load_tracks",
        "save_tracks",
    ]
    assert pst.SimpleTracker is not None
    assert pst.HodgesTracker is not None
    assert pst.HealpixTracker is not None
    assert pst.Tracker is not None
    assert pst.Track is not None
    assert pst.Tracks is not None
    assert pst.Center is not None
    assert pst.load_tracks is not None
    assert pst.save_tracks is not None


def test_supported_subpackage_entry_points() -> None:
    from pystormtracker.healpix import HealpixDetector, HealpixTracker
    from pystormtracker.hodges import HodgesTracker
    from pystormtracker.io import DataLoader, infer_format
    from pystormtracker.metrics import compute_track_metrics
    from pystormtracker.models import Center, Tracks
    from pystormtracker.preprocessing import SHTFilter
    from pystormtracker.refinement import (
        build_bspline_surface,
        build_spherical_bspline_surface,
        refine_bspline_feature_point,
        refine_quadratic_feature_point,
        refine_quadratic_feature_points,
        refine_spherical_bspline_feature_point,
        refine_spherical_quadratic_feature_points,
    )
    from pystormtracker.simple import SimpleDetector, SimpleLinker, SimpleTracker

    assert SimpleTracker is pst.SimpleTracker
    assert HodgesTracker is pst.HodgesTracker
    assert HealpixTracker is pst.HealpixTracker
    assert all(
        callable(entry_point)
        for entry_point in (
            SimpleDetector,
            SimpleLinker,
            HealpixDetector,
            DataLoader,
            infer_format,
            compute_track_metrics,
            SHTFilter,
            build_bspline_surface,
            build_spherical_bspline_surface,
            refine_bspline_feature_point,
            refine_quadratic_feature_point,
            refine_quadratic_feature_points,
            refine_spherical_bspline_feature_point,
            refine_spherical_quadratic_feature_points,
        )
    )
    assert Center is pst.Center
    assert Tracks is pst.Tracks

    import pystormtracker.models as model_namespace

    assert not hasattr(model_namespace, "CenterFrame")


def test_protocol_conformance() -> None:
    simple = pst.SimpleTracker()
    hodges = pst.HodgesTracker()
    healpix = pst.HealpixTracker()

    assert accepts_tracker(simple) is simple
    assert accepts_tracker(hodges) is hodges
    assert accepts_tracker(healpix) is healpix


def test_center_frame_named_tuple() -> None:
    time_val: TimePoint = 1735689600000
    lats = np.array([10.0, 20.0], dtype=np.float64)
    lons = np.array([30.0, 40.0], dtype=np.float64)
    vals = np.array([100.0, 200.0], dtype=np.float64)

    step = CenterFrame(time_val, lats, lons, vals)

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
