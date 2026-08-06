from unittest.mock import MagicMock, patch

from pystormtracker.models.tracks import Tracks, TracksMetadata
from pystormtracker.simple.tracker import SimpleTracker


def _empty_tracks() -> Tracks:
    return Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))


def test_tracker_time_range() -> None:
    tracker = SimpleTracker()

    with patch.object(tracker, "_detect_serial", return_value=_empty_tracks()):
        tracker.track("dummy.nc", "msl", start_time="2025-01-01")
        tracker.track("dummy.nc", "msl", end_time="2025-01-31")


def test_tracker_defaults_disable_optional_filter_and_refinement() -> None:
    tracker = SimpleTracker()

    with patch.object(tracker, "_detect_serial", return_value=_empty_tracks()):
        tracker.track("dummy.nc", "msl")

    assert tracker.filter_lmin is None
    assert tracker.filter_lmax is None
    assert tracker.taper_points == 0
    assert tracker.feature_point_method == "grid"


def test_tracker_mpi_backend() -> None:
    tracker = SimpleTracker(
        backend="mpi",
        projection="sh_stereo",
        stereo_grid_spacing_km=200.0,
        extent=(-1000.0, 1000.0, -900.0, 900.0),
        filter_lmin=0,
        filter_lmax=21,
        feature_point_method="quadratic",
    )

    with (
        patch(
            "pystormtracker.simple.concurrent.run_simple_mpi",
            return_value=_empty_tracks(),
        ) as mock_run_mpi,
        patch.dict("sys.modules", {"mpi4py": MagicMock()}),
    ):
        tracker.track("dummy.nc", "msl")
        mock_run_mpi.assert_called_once()
        assert mock_run_mpi.call_args.kwargs["projection"] == "sh_stereo"
        assert mock_run_mpi.call_args.kwargs["stereo_grid_spacing_km"] == 200.0
        assert mock_run_mpi.call_args.kwargs["filter_lmax"] == 21
        assert mock_run_mpi.call_args.kwargs["filter_lmin"] == 0
        assert mock_run_mpi.call_args.kwargs["feature_point_method"] == "quadratic"


def test_tracker_dask_backend() -> None:
    tracker = SimpleTracker(
        backend="dask",
        projection="nh_stereo",
        stereo_grid_spacing_km=250.0,
        extent=(-1000.0, 1000.0, -800.0, 800.0),
        filter_lmin=0,
        filter_lmax=17,
        feature_point_method="quadratic",
    )

    with patch(
        "pystormtracker.simple.concurrent.run_simple_dask", return_value=_empty_tracks()
    ) as mock_run_dask:
        tracker.track("dummy.nc", "msl")
        mock_run_dask.assert_called_once()
        assert mock_run_dask.call_args.kwargs["projection"] == "nh_stereo"
        assert mock_run_dask.call_args.kwargs["stereo_grid_spacing_km"] == 250.0
        assert mock_run_dask.call_args.kwargs["filter_lmax"] == 17
        assert mock_run_dask.call_args.kwargs["filter_lmin"] == 0
        assert mock_run_dask.call_args.kwargs["feature_point_method"] == "quadratic"
