from __future__ import annotations

from inspect import signature
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from pystormtracker.backends import local_dask_executor
from pystormtracker.hodges.detections import HodgesCenterFrame
from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.models.tracks import Tracks, TracksMetadata


def test_hodges_tracker_init() -> None:
    tracker = HodgesTracker(w1=0.3, min_track_points=5)
    assert tracker.w1 == 0.3
    assert tracker.min_track_points == 5


def test_hodges_tracker_standard_defaults() -> None:
    tracker = HodgesTracker()
    assert tracker.dmax == 6.5
    assert tracker.dmax_zones is not None
    assert len(tracker.dmax_zones) == 3
    assert tracker.adaptive_smoothness is not None
    assert tracker.adaptive_smoothness.shape == (2, 4)
    assert tracker.phimax == 1.0
    assert tracker.spectral_taper == 1.0
    assert tracker.feature_refinement == "bspline"
    assert tracker.bspline_smoothing == 0.0
    assert tracker.track_smoopy_optimization_scale == 1.0


def test_hodges_tracker_override_constraints() -> None:
    custom_zones = np.array([[0.0, 360.0, -90.0, 90.0, 10.0]], dtype=np.float64)
    custom_params = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 0.5, 0.2, 0.1]], dtype=np.float64
    )

    tracker = HodgesTracker(
        dmax_zones=custom_zones,
        adaptive_smoothness=custom_params,
    )

    assert tracker.dmax_zones is not None
    assert tracker.dmax_zones[0, 4] == 10.0
    assert tracker.dmax == 10.0
    assert np.array_equal(tracker.adaptive_smoothness, custom_params)
    assert tracker.phimax == 1.0


def test_hodges_tracker_constructor_validation() -> None:
    with pytest.raises(ValueError, match="w1 and w2 must be nonnegative"):
        HodgesTracker(w1=-0.1)
    with pytest.raises(ValueError, match="dmax must be positive"):
        HodgesTracker(dmax=0.0)
    with pytest.raises(ValueError, match="requires feature_refinement='grid'"):
        HodgesTracker(group_adjacent_extrema=True)
    with pytest.raises(ValueError, match="spectral_taper"):
        HodgesTracker(spectral_taper=0.0)
    with pytest.raises(ValueError, match=r"shape \(2, 0\) or \(2, 4\)"):
        HodgesTracker(adaptive_smoothness=np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="strictly increasing"):
        HodgesTracker(
            adaptive_smoothness=np.array(
                [[1.0, 1.0, 2.0, 3.0], [1.0, 0.5, 0.2, 0.1]],
                dtype=np.float64,
            )
        )
    with pytest.raises(ValueError, match="multiple missing-frame parameter sets"):
        HodgesTracker(
            missing_frame_parameters=np.array(
                [[2.0, 0.5], [10.0, 0.5]],
                dtype=np.float64,
            )
        )


def test_hodges_tracker_exposes_independent_execution_controls() -> None:
    tracker = HodgesTracker(
        backend="dask",
        frame_workers=1,
        sht_threads=16,
        mge_workers=16,
    )

    assert tracker.frame_workers == 1
    assert tracker.sht_threads == 16
    assert tracker.mge_workers == 16

    with pytest.raises(ValueError, match="frame_workers"):
        HodgesTracker(backend="serial", frame_workers=1)
    with pytest.raises(ValueError, match="mge_workers"):
        HodgesTracker(backend="serial", mge_workers=1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        signature(HodgesTracker).bind(workers=4)


@patch("pystormtracker.hodges.tracker.merge_segments")
@patch("pystormtracker.hodges.tracker._link_hodges_segment_task")
@patch("pystormtracker.hodges.tracker._detect_hodges_frame_task")
@patch("pystormtracker.hodges.tracker.local_dask_executor", wraps=local_dask_executor)
def test_hodges_dask_uses_separate_frame_and_mge_executors(
    mock_executor: MagicMock,
    mock_detect: MagicMock,
    mock_link: MagicMock,
    mock_merge: MagicMock,
) -> None:
    times = np.arange(5).astype("timedelta64[h]") + np.datetime64("2025-12-01")
    data = xr.DataArray(
        np.zeros((5, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": times,
            "lat": [-90.0, 0.0, 90.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
        name="msl",
        attrs={"units": "Pa"},
    )
    frame = HodgesCenterFrame(
        times[0],
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    empty_tracks = Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))
    mock_detect.return_value = frame
    mock_link.return_value = empty_tracks
    mock_merge.return_value = empty_tracks

    tracker = HodgesTracker(
        backend="dask",
        frame_workers=1,
        sht_threads=1,
        mge_workers=2,
        segment_frames=2,
        feature_refinement="grid",
    )
    result = tracker.track(data, "msl")

    assert result == empty_tracks
    assert [call.args[0] for call in mock_executor.call_args_list] == [1, 2]
    assert mock_link.call_count > 1
    assert all(
        isinstance(detection, HodgesCenterFrame)
        for call in mock_link.call_args_list
        for detection in call.args[0]
    )
    mock_merge.assert_called_once()


def test_hodges_tracker_requires_time_step_for_multi_parameter_tracking() -> None:
    tracker = HodgesTracker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        missing_frame_parameters=np.array(
            [[2.0, 0.5], [10.0, 0.5]],
            dtype=np.float64,
        ),
    )
    with pytest.raises(ValueError, match="time_step is required"):
        tracker.track(
            xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "lat", "lon")),
            "msl",
        )


def test_hodges_tracker_accepts_bspline() -> None:
    tracker = HodgesTracker(
        feature_refinement="bspline",
    )

    assert tracker.feature_refinement == "bspline"


def test_hodges_tracker_accepts_missing_frame_parameter_sets() -> None:
    parameters = np.array([[2.0, 0.5], [10.0, 0.8]], dtype=np.float64)

    tracker = HodgesTracker(
        dmax_zones=np.zeros((0, 5), dtype=np.float64),
        adaptive_smoothness=np.zeros((2, 0), dtype=np.float64),
        missing_frame_parameters=parameters,
    )

    assert tracker.dmax == 2.0
    assert tracker.phimax == 0.5
    assert np.array_equal(tracker.missing_frame_parameters, parameters)


def test_hodges_tracker_rejects_nonpositive_chunks() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        HodgesTracker(segment_frames=0)


@patch("pystormtracker.hodges.tracker.merge_segments")
@patch("pystormtracker.hodges.tracker._link_hodges_segment_task")
def test_hodges_tracker_processes_chunks_and_splices(
    mock_link_task: MagicMock, mock_merge: MagicMock
) -> None:
    times = np.arange(5).astype("timedelta64[h]") + np.datetime64("2025-12-01")
    data = xr.DataArray(
        np.zeros((5, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-1.0, 1.0], "lon": [0.0, 1.0]},
        name="msl",
    )
    empty_tracks = Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))
    mock_link_task.return_value = empty_tracks
    mock_merge.return_value = empty_tracks

    HodgesTracker(segment_frames=2, feature_refinement="grid").track(data, "msl")

    assert mock_link_task.call_count > 1
    assert mock_merge.call_count == 1


@patch("pystormtracker.hodges.detector.HodgesDetector.detect")
def test_hodges_tracker_track_single_chunk(mock_detect: MagicMock) -> None:
    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")

    mock_detect.return_value = [
        (t0, np.array([0.0]), np.array([0.0]), np.array([1000.0])),
        (t1, np.array([1.0]), np.array([1.0]), np.array([990.0])),
    ]

    data = xr.DataArray(
        np.zeros((2, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [t0, t1], "lat": [0.0], "lon": [0.0]},
        name="msl",
    )
    tracker = HodgesTracker(min_track_points=2, backend="serial")
    with patch(
        "pystormtracker.hodges.tracker.normalize_tracking_data", return_value=data
    ):
        tracks = tracker.track("dummy.nc", "msl")

    assert len(tracks) == 1
    assert len(tracks[0]) == 2
    assert tracks[0][0].lat == 0.0
    assert tracks[0][1].lat == 1.0


@patch("pystormtracker.hodges.detector.HodgesDetector.detect")
def test_hodges_tracker_propagates_detector_diagnostics(
    mock_detect: MagicMock,
) -> None:
    t0 = np.datetime64("2025-12-01T00:00:00")
    t1 = np.datetime64("2025-12-01T06:00:00")
    diagnostic_units = {
        "raw_value": None,
        "object_gridcell_area_km2": "km2",
    }
    mock_detect.return_value = [
        HodgesCenterFrame(
            t0,
            np.array([0.0]),
            np.array([0.0]),
            np.array([1000.0]),
            {
                "raw_value": np.array([1001.0]),
                "object_gridcell_area_km2": np.array([10.0]),
            },
            diagnostic_units,
        ),
        HodgesCenterFrame(
            t1,
            np.array([0.0]),
            np.array([1.0]),
            np.array([990.0]),
            {
                "raw_value": np.array([991.0]),
                "object_gridcell_area_km2": np.array([20.0]),
            },
            diagnostic_units,
        ),
    ]
    data = xr.DataArray(
        np.zeros((2, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [t0, t1], "lat": [0.0], "lon": [0.0]},
        name="msl",
    )
    tracker = HodgesTracker(min_track_points=2, backend="serial")
    with patch(
        "pystormtracker.hodges.tracker.normalize_tracking_data", return_value=data
    ):
        tracks = tracker.track("dummy.nc", "msl")

    assert len(tracks) == 1
    np.testing.assert_array_equal(
        tracks[0].variables["raw_value"],
        np.array([1001.0, 991.0]),
    )
    np.testing.assert_array_equal(
        tracks[0].variables["object_gridcell_area_km2"],
        np.array([10.0, 20.0]),
    )
    assert tracks.metadata.units["raw_value"] == "Pa"


def test_hodges_tracker_preprocess_projection() -> None:
    ny, nx = 73, 144
    time = np.array([np.datetime64("2025-12-01T00:00:00")], dtype="datetime64[ns]")
    data = np.random.default_rng().random((1, ny, nx))
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": time,
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    tracker = HodgesTracker()
    processed, _steps = tracker._preprocess_standard_track(da, projection="nh_stereo")
    assert processed.dims == ("time", "y", "x")
    assert processed.attrs["projection"] == "nh_stereo"

    processed_sh, _steps = tracker._preprocess_standard_track(
        da, projection="sh_stereo"
    )
    assert processed_sh.dims == ("time", "y", "x")
    assert processed_sh.attrs["projection"] == "sh_stereo"


def test_hodges_tracker_mpi_error_propagation() -> None:
    """Root raises RuntimeError when any rank reports a local error via gather."""
    import sys

    times = np.arange(4).astype("timedelta64[h]") + np.datetime64("2025-12-01")
    data = xr.DataArray(
        np.zeros((4, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [-1.0, 1.0], "lon": [0.0, 1.0]},
        name="msl",
    )

    # Simulate rank-0 (root) view: rank=0, world_size=2.
    # gather receives [None, "rank 1 segment 1 frames [2:4] source '<array>': boom"].
    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 0
    mock_comm.Get_size.return_value = 2
    mock_comm.gather.side_effect = [
        [None, "rank 1 segment 1 frames [2:4] source '<array>': boom"],  # errors
        [
            [(0, Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"})))],
            [],
        ],  # results
    ]
    # `from mpi4py import MPI` resolves to sys.modules["mpi4py"].MPI.
    mock_MPI_module = MagicMock()
    mock_MPI_module.COMM_WORLD = mock_comm
    # Intracomm type annotation — just needs to be a MagicMock.
    mock_MPI_module.Intracomm = MagicMock()
    mock_mpi4py = MagicMock()
    mock_mpi4py.MPI = mock_MPI_module

    with (
        patch.dict(sys.modules, {"mpi4py": mock_mpi4py, "mpi4py.MPI": mock_MPI_module}),
        patch(
            "pystormtracker.hodges.tracker.HodgesTracker._run_segment_task",
            return_value=Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"})),
        ),
    ):
        tracker = HodgesTracker(backend="mpi", segment_frames=2)
        with pytest.raises(RuntimeError, match="MPI tracking failed"):
            tracker._track_mpi(
                data,
                primary_variable="msl",
                mode="min",
                bounds=None,
                unit="Pa",
                processing=(),
                threshold=None,
            )
