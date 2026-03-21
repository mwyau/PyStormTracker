from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from pystormtracker.hodges.tracker import HodgesTracker


def test_hodges_tracker_init() -> None:
    tracker = HodgesTracker(w1=0.3, min_lifetime=5)
    assert tracker.w1 == 0.3
    assert tracker.min_lifetime == 5


def test_hodges_tracker_standard_defaults() -> None:
    tracker = HodgesTracker()
    assert tracker.dmax == 6.5
    assert tracker.zones is not None
    assert len(tracker.zones) == 3
    assert tracker.adapt_thresholds is not None
    assert len(tracker.adapt_thresholds) == 4


def test_hodges_tracker_override_constraints() -> None:
    custom_zones = np.array([[0.0, 360.0, -90.0, 90.0, 10.0]], dtype=np.float64)
    custom_thresholds = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    custom_values = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float64)

    tracker = HodgesTracker(
        zones=custom_zones,
        adapt_thresholds=custom_thresholds,
        adapt_values=custom_values,
        use_standard_constraints=False,
    )

    assert tracker.zones is not None
    assert tracker.zones[0, 4] == 10.0
    assert np.array_equal(tracker.adapt_thresholds, custom_thresholds)
    assert np.array_equal(tracker.adapt_values, custom_values)


@patch("pystormtracker.hodges.detector.HodgesDetector.detect")
def test_hodges_tracker_track_single_chunk(mock_detect: MagicMock) -> None:
    # Mock detection results
    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")

    mock_detect.return_value = [
        (t0, np.array([0.0]), np.array([0.0]), {"msl": np.array([1000.0])}),
        (t1, np.array([1.0]), np.array([1.0]), {"msl": np.array([990.0])}),
    ]

    tracker = HodgesTracker(min_lifetime=2)
    # Patch get_time and get_xarray to avoid file opening
    with patch(
        "pystormtracker.hodges.detector.HodgesDetector.get_time", return_value=[t0, t1]
    ), patch(
        "pystormtracker.hodges.detector.HodgesDetector.get_xarray",
        return_value=MagicMock(),
    ):
        tracks = tracker.track("dummy.nc", "msl", filter=False)

    assert len(tracks) == 1
    assert len(tracks[0]) == 2
    assert tracks[0][0].lat == 0.0
    assert tracks[0][1].lat == 1.0
