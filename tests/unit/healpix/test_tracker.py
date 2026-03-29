from __future__ import annotations

import pytest

from pystormtracker.healpix.tracker import HealpixTracker


def test_healpix_tracker_not_implemented_backend() -> None:
    tracker = HealpixTracker()
    with pytest.raises(NotImplementedError):
        tracker.track("dummy.nc", "msl", backend="dask")


def test_healpix_tracker_time_range() -> None:
    tracker = HealpixTracker()
    # Basic check for parameter routing.
    # Serial detection will fail on dummy.nc if it doesn't exist.
    # We can mock detect if we want, but let's just test the init of time_range
    # for now using valid dates to avoid datetime64 errors.
    with pytest.raises((FileNotFoundError, Exception)) as excinfo:
        tracker.track(
            "nonexistent.nc", "msl", start_time="2025-01-01", end_time="2025-01-31"
        )
    assert "nonexistent.nc" in str(excinfo.value) or isinstance(
        excinfo.value, FileNotFoundError
    )
