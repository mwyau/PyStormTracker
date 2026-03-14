from unittest.mock import MagicMock, patch

from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.tracker import SimpleTracker


def test_tracker_time_range() -> None:
    tracker = SimpleTracker()

    # Mock _detect_serial to avoid actually running detection
    with patch.object(tracker, "_detect_serial", return_value=Tracks()):
        # Only start_time provided (end_time should be NaT)
        tracker.track("dummy.nc", "msl", start_time="2025-01-01")

        # Only end_time provided (start_time should be NaT)
        tracker.track("dummy.nc", "msl", end_time="2025-01-31")


def test_tracker_mpi_backend() -> None:
    tracker = SimpleTracker()

    # Mock run_simple_mpi and mpi4py
    with (
        patch(
            "pystormtracker.simple.concurrent.run_simple_mpi", return_value=Tracks()
        ) as mock_run_mpi,
        patch.dict("sys.modules", {"mpi4py": MagicMock()}),
    ):
        tracker.track("dummy.nc", "msl", backend="mpi")
        mock_run_mpi.assert_called_once()


def test_tracker_dask_backend() -> None:
    tracker = SimpleTracker()

    with patch(
        "pystormtracker.simple.concurrent.run_simple_dask", return_value=Tracks()
    ) as mock_run_dask:
        tracker.track("dummy.nc", "msl", backend="dask")
        mock_run_dask.assert_called_once()
