from __future__ import annotations

import numpy as np

import pystormtracker as pst
from pystormtracker import (
    healpix,
    hodges,
    io,
    metrics,
    models,
    preprocessing,
    refinement,
    simple,
)
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


def test_supported_package_namespaces_are_curated() -> None:
    assert simple.__all__ == ["SimpleTracker"]
    assert hodges.__all__ == ["HodgesTracker"]
    assert healpix.__all__ == ["HealpixTracker"]

    assert models.__all__ == [
        "Center",
        "DetectionMode",
        "ProcessingStep",
        "Projection",
        "ResolvedDetectionMode",
        "SpatialBounds",
        "Track",
        "Tracker",
        "Tracks",
        "TracksMetadata",
    ]

    assert io.__all__ == [
        "SUPPORTED_FORMATS",
        "DataLoader",
        "SupportedFormat",
        "infer_format",
        "load_tracks",
        "save_tracks",
    ]
    assert preprocessing.__all__ == [
        "BoundaryTaper",
        "DCTFilter",
        "SHTFilter",
        "SpectralRegridder",
        "compute_vorticity_divergence",
    ]
    assert refinement.__all__ == []

    assert metrics.__all__ == [
        "compute_cormax",
        "compute_eke",
        "compute_high_wind_index",
        "compute_track_metrics",
        "compute_variance_metric",
        "find_best_cca_truncation",
        "train_cca_model",
    ]


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
