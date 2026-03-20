from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pathlib import Path

from pystormtracker.hodges.tracker import HodgesTracker


def test_hodges_tracker_init() -> None:
    tracker = HodgesTracker(w1=0.3, min_lifetime=5)
    assert tracker.w1 == 0.3
    assert tracker.min_lifetime == 5


def test_hodges_tracker_from_config(tmp_path: Path) -> None:
    # Create dummy zone.dat
    zone_file = tmp_path / "zone.dat"
    zone_file.write_text("1\n0.0 360.0 -90.0 90.0 5.5\n")

    # Create dummy adapt.dat
    adapt_file = tmp_path / "adapt.dat"
    adapt_file.write_text("1.0 2.0 3.0 4.0\n1.0 0.5 0.2 0.1\n")

    tracker = HodgesTracker.from_config(
        zone_file=str(zone_file), adapt_file=str(adapt_file), w1=0.4
    )

    assert tracker.w1 == 0.4
    assert tracker.zones is not None
    assert tracker.zones[0, 4] == 5.5
    assert tracker.adapt_thresholds is not None
    assert tracker.adapt_thresholds[0] == 1.0
    assert tracker.adapt_values is not None
    assert tracker.adapt_values[1] == 0.5


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
    # Patch get_time to avoid file opening
    with patch(
        "pystormtracker.hodges.detector.HodgesDetector.get_time", return_value=[t0, t1]
    ):
        tracks = tracker.track("dummy.nc", "msl")

    assert len(tracks) == 1
    assert len(tracks[0]) == 2
    assert tracks[0][0].lat == 0.0
    assert tracks[0][1].lat == 1.0
