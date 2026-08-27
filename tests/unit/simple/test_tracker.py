from __future__ import annotations

from inspect import signature
from unittest.mock import MagicMock, patch

import pytest

from pystormtracker.models.tracks import Tracks, TracksMetadata
from pystormtracker.simple.tracker import SimpleTracker


def _empty_tracks() -> Tracks:
    return Tracks.empty(
        TracksMetadata(
            primary_variable="msl",
            mode="min",
            units={"msl": "Pa"},
        )
    )


def test_tracker_init_and_kwargs() -> None:
    tracker = SimpleTracker(
        projection="nh_stereo",
        stereo_grid_spacing_km=150.0,
        extent=(-1000.0, 1000.0, -1000.0, 1000.0),
        lmin=0,
        lmax=42,
        taper_points=5,
        search_window_size=7,
        feature_refinement="grid",
    )
    assert tracker.projection == "nh_stereo"
    assert tracker.stereo_grid_spacing_km == 150.0
    assert tracker.search_window_size == 7
    assert tracker.feature_refinement == "grid"

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        signature(SimpleTracker).bind(invalid_param=123)


def test_tracker_mpi_backend() -> None:
    tracker = SimpleTracker(
        backend="mpi",
        projection="sh_stereo",
        stereo_grid_spacing_km=200.0,
        extent=(-1000.0, 1000.0, -900.0, 900.0),
        lmin=0,
        lmax=21,
        feature_refinement="quadratic",
    )

    with (
        patch.object(
            tracker,
            "_track_mpi",
            return_value=_empty_tracks(),
        ) as mock_track_mpi,
        patch.dict("sys.modules", {"mpi4py": MagicMock()}),
    ):
        tracker.track("dummy.nc", "msl")
        mock_track_mpi.assert_called_once()
        assert mock_track_mpi.call_args.kwargs["variable"] == "msl"
        assert mock_track_mpi.call_args.kwargs["detection_mode"] == "min"


def test_tracker_dask_backend() -> None:
    tracker = SimpleTracker(
        backend="dask",
        projection="nh_stereo",
        stereo_grid_spacing_km=250.0,
        extent=(-1000.0, 1000.0, -800.0, 800.0),
        lmin=0,
        lmax=17,
        feature_refinement="quadratic",
    )

    with patch.object(
        tracker,
        "_track_dask",
        return_value=_empty_tracks(),
    ) as mock_track_dask:
        tracker.track("dummy.nc", "msl")
        mock_track_dask.assert_called_once()
        assert mock_track_dask.call_args.kwargs["variable"] == "msl"
        assert mock_track_dask.call_args.kwargs["detection_mode"] == "min"
