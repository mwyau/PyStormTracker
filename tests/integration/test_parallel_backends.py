from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from ducc0.healpix import Healpix_Base  # ty: ignore[unresolved-import]

from pystormtracker.healpix.tracker import HealpixTracker
from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.trackjson import read_trackjson
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.tracker import SimpleTracker


@pytest.fixture
def sample_msl_dataset() -> xr.DataArray:
    """Create a multi-frame synthetic global dataset with depression objects."""
    n_times = 12
    times = np.arange(n_times).astype("timedelta64[h]") + np.datetime64("2024-01-01")
    lats = np.linspace(-80.0, 80.0, 81)
    lons = np.linspace(0.0, 360.0, 180, endpoint=False)

    values = np.full((n_times, len(lats), len(lons)), 101325.0, dtype=np.float64)

    # Moving depression (moves 1 grid cell = 2 deg lon per step)
    for t in range(n_times):
        lat_c = 45 + (t % 2)
        lon_c = (30 + t) % len(lons)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                y_i = lat_c + dy
                x_i = (lon_c + dx) % len(lons)
                if 0 <= y_i < len(lats):
                    r2 = dy**2 + dx**2
                    dep_val = 98000.0 + r2 * 200.0
                    values[t, y_i, x_i] = min(values[t, y_i, x_i], dep_val)

    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lats, "lon": lons},
        name="msl",
        attrs={"units": "Pa"},
    )


def _assert_tracks_equal(t1: Tracks, t2: Tracks) -> None:
    assert t1.metadata.primary_variable == t2.metadata.primary_variable
    assert t1.metadata.mode == t2.metadata.mode
    assert t1.metadata.units == t2.metadata.units
    assert t1.metadata.bounds == t2.metadata.bounds
    assert t1.metadata.processing == t2.metadata.processing
    assert len(t1) == len(t2)
    np.testing.assert_array_equal(t1.ids, t2.ids)
    np.testing.assert_array_equal(t1.offsets, t2.offsets)
    np.testing.assert_array_equal(t1.times, t2.times)
    np.testing.assert_allclose(t1.lats, t2.lats, atol=1.0e-8)
    np.testing.assert_allclose(t1.lons, t2.lons, atol=1.0e-8)
    for var_name in t1.variables:
        assert var_name in t2.variables
        np.testing.assert_allclose(
            t1.variables[var_name], t2.variables[var_name], atol=1.0e-8
        )


def _assert_synthetic_moving_depression(tracks: Tracks) -> None:
    """Check the known feature path encoded by ``sample_msl_dataset``."""
    assert len(tracks) == 1
    assert len(tracks[0]) == 12
    np.testing.assert_allclose(
        tracks[0].lats,
        np.where(np.arange(12) % 2 == 0, 10.0, 12.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tracks[0].lons,
        np.arange(60.0, 84.0, 2.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tracks[0].variables["msl"],
        np.full(12, 98000.0),
        rtol=0.0,
        atol=1.0e-8,
    )


@pytest.mark.integration
@pytest.mark.parametrize("workers", [4])
def test_hodges_dask_matches_serial_spherical_bspline(
    sample_msl_dataset: xr.DataArray, workers: int
) -> None:
    tracker_serial = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="spherical_bspline",
        backend="serial",
    )
    res_serial = tracker_serial.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    tracker_dask = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="spherical_bspline",
        backend="dask",
        frame_workers=workers,
        mge_workers=workers,
    )
    res_dask = tracker_dask.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    _assert_synthetic_moving_depression(res_serial)
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
@pytest.mark.parametrize("workers", [4])
def test_hodges_dask_matches_serial_bspline(
    sample_msl_dataset: xr.DataArray, workers: int
) -> None:
    tracker_serial = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="bspline",
        backend="serial",
    )
    res_serial = tracker_serial.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    tracker_dask = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="bspline",
        backend="dask",
        frame_workers=workers,
        mge_workers=workers,
    )
    res_dask = tracker_dask.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    _assert_synthetic_moving_depression(res_serial)
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("frame_workers", "sht_threads", "mge_workers"),
    [(1, 1, 1), (2, 1, 2)],
)
def test_hodges_stage_controls_preserve_serial_result(
    sample_msl_dataset: xr.DataArray,
    frame_workers: int,
    sht_threads: int,
    mge_workers: int,
) -> None:
    """Independent Dask stage controls preserve the canonical result."""
    tracker_serial = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="grid",
        backend="serial",
    )
    res_serial = tracker_serial.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    tracker_dask = HodgesTracker(
        min_track_points=3,
        min_object_grid_points=1,
        segment_frames=4,
        feature_refinement="grid",
        backend="dask",
        frame_workers=frame_workers,
        sht_threads=sht_threads,
        mge_workers=mge_workers,
    )
    res_dask = tracker_dask.track(
        sample_msl_dataset,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    _assert_synthetic_moving_depression(res_serial)
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
@pytest.mark.parametrize("workers", [4])
def test_simple_dask_matches_serial(
    sample_msl_dataset: xr.DataArray, tmp_path: Path, workers: int
) -> None:
    nc_path = tmp_path / "simple_test.nc"
    sample_msl_dataset.to_netcdf(nc_path, engine="h5netcdf")

    tracker_serial = SimpleTracker(
        search_window_size=5,
        backend="serial",
    )
    res_serial = tracker_serial.track(
        str(nc_path),
        "msl",
        detection_mode="min",
        feature_threshold=50.0,
    )

    tracker_dask = SimpleTracker(
        search_window_size=5,
        backend="dask",
        workers=workers,
    )
    res_dask = tracker_dask.track(
        str(nc_path),
        "msl",
        detection_mode="min",
        feature_threshold=50.0,
    )

    _assert_synthetic_moving_depression(res_serial)
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
def test_simple_mpi_matches_serial_on_synthetic_trajectory(tmp_path: Path) -> None:
    """A real four-rank MPI run preserves a known moving feature trajectory."""
    pytest.importorskip("mpi4py")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec not found in PATH")

    times = np.arange(6).astype("timedelta64[h]") + np.datetime64("2024-01-01")
    latitudes = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    longitudes = np.arange(0.0, 360.0, 2.0)
    values = np.full((6, latitudes.size, longitudes.size), 101325.0)
    for time_index in range(6):
        values[time_index, 2, 30 + time_index] = 98000.0
    data = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        name="msl",
        attrs={"units": "Pa"},
    )
    input_path = tmp_path / "synthetic_msl.nc"
    output_path = tmp_path / "synthetic_mpi.trackjson"
    data.to_netcdf(input_path, engine="h5netcdf")

    tracker = SimpleTracker(search_window_size=5, backend="serial")
    serial = tracker.track(
        data,
        "msl",
        detection_mode="min",
        feature_threshold=50.0,
    )
    subprocess.run(
        [
            mpiexec,
            "-n",
            "4",
            sys.executable,
            "-m",
            "pystormtracker.cli",
            "track",
            "-i",
            str(input_path),
            "--variable",
            "msl",
            "-m",
            "min",
            "--feature-threshold",
            "50.0",
            "--search-window-size",
            "5",
            "--backend",
            "mpi",
            "-o",
            str(output_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mpi = read_trackjson(output_path)

    assert len(serial) == 1
    assert len(serial[0]) == 6
    np.testing.assert_allclose(serial[0].lats, np.zeros(6), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        serial[0].lons,
        np.arange(60.0, 72.0, 2.0),
        rtol=0.0,
        atol=0.0,
    )
    _assert_tracks_equal(serial, mpi)


@pytest.mark.integration
@pytest.mark.parametrize("workers", [4])
def test_healpix_dask_matches_serial(workers: int) -> None:
    n_times = 8
    nside = 8
    npix = 12 * nside * nside  # 768 pixels
    times = np.arange(n_times).astype("timedelta64[h]") + np.datetime64("2024-01-01")

    values = np.full((n_times, npix), 101325.0, dtype=np.float64)

    # Feature at fixed pixel 100 for all time steps
    for t in range(n_times):
        values[t, 100] = 98000.0

    healpix_da = xr.DataArray(
        values,
        dims=("time", "cell"),
        coords={"time": times},
        name="msl",
        attrs={"grid_type": "healpix", "nside": nside, "units": "Pa"},
    )

    tracker_serial = HealpixTracker(
        min_track_points=2,
        min_object_grid_points=1,
        dmax_zones=np.empty((0, 5), dtype=np.float64),
        dmax=10.0,
        segment_frames=4,
        nside=nside,
        feature_refinement="grid",
        backend="serial",
    )
    res_serial = tracker_serial.track(
        healpix_da,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    tracker_dask = HealpixTracker(
        min_track_points=2,
        min_object_grid_points=1,
        dmax_zones=np.empty((0, 5), dtype=np.float64),
        dmax=10.0,
        segment_frames=4,
        nside=nside,
        feature_refinement="grid",
        backend="dask",
        workers=workers,
    )
    res_dask = tracker_dask.track(
        healpix_da,
        "msl",
        detection_mode="min",
        object_threshold=100000.0,
    )

    assert len(res_serial) == 1
    assert len(res_serial[0]) == n_times
    healpix_angles = Healpix_Base(nside, "RING").pix2ang(np.array([100]))[0]
    expected_latitude = 90.0 - float(np.rad2deg(healpix_angles[0]))
    expected_longitude = float((np.rad2deg(healpix_angles[1]) + 180.0) % 360.0 - 180.0)
    np.testing.assert_allclose(
        res_serial[0].lats,
        expected_latitude,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        res_serial[0].lons,
        expected_longitude,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        res_serial[0].variables["msl"],
        np.full(n_times, 98000.0),
        rtol=0.0,
        atol=1.0e-8,
    )
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
def test_healpix_quadratic_tracker_matches_serial_and_dask() -> None:
    """A smooth spherical minimum follows the default quadratic path."""
    n_times = 6
    nside = 8
    npix = 12 * nside * nside
    target = (32.3, 179.2)
    times = np.arange(n_times).astype("timedelta64[h]") + np.datetime64("2024-01-01")

    hp_base = Healpix_Base(nside, "RING")
    pixels = np.arange(npix, dtype=np.int64)
    angles = hp_base.pix2ang(pixels)
    latitudes = np.rad2deg(0.5 * np.pi - angles[:, 0])
    longitudes = np.rad2deg(angles[:, 1]) % 360.0
    target_latitude, target_longitude = np.deg2rad(target)
    target_vector = np.array(
        [
            np.cos(target_latitude) * np.cos(target_longitude),
            np.cos(target_latitude) * np.sin(target_longitude),
            np.sin(target_latitude),
        ]
    )
    points = np.stack(
        (
            np.cos(np.deg2rad(latitudes)) * np.cos(np.deg2rad(longitudes)),
            np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(longitudes)),
            np.sin(np.deg2rad(latitudes)),
        ),
        axis=-1,
    )
    values = np.tile(100000.0 + 10000.0 * (1.0 - points @ target_vector), (n_times, 1))
    healpix_data = xr.DataArray(
        values,
        dims=("time", "cell"),
        coords={"time": times},
        name="msl",
        attrs={"grid_type": "healpix", "nside": nside, "units": "Pa"},
    )

    tracker_serial = HealpixTracker(
        nside=nside,
        min_track_points=2,
        dmax=10.0,
        dmax_zones=np.empty((0, 5), dtype=np.float64),
        segment_frames=4,
        backend="serial",
    )
    tracker_dask = HealpixTracker(
        nside=nside,
        min_track_points=2,
        dmax=10.0,
        dmax_zones=np.empty((0, 5), dtype=np.float64),
        segment_frames=4,
        backend="dask",
        workers=4,
    )
    assert tracker_serial.feature_refinement == "quadratic"
    assert tracker_dask.feature_refinement == "quadratic"

    serial = tracker_serial.track(
        healpix_data,
        "msl",
        detection_mode="min",
        object_threshold=102000.0,
    )
    dask = tracker_dask.track(
        healpix_data,
        "msl",
        detection_mode="min",
        object_threshold=102000.0,
    )

    assert len(serial) == 1
    assert len(serial[0]) == n_times
    grid_pixel = int(np.argmin(values[0]))
    grid_latitude = float(latitudes[grid_pixel])
    grid_longitude = float(longitudes[grid_pixel])

    def angular_error(latitude: float, longitude: float) -> float:
        latitude_rad, longitude_rad = np.deg2rad([latitude, longitude])
        vector = np.array(
            [
                np.cos(latitude_rad) * np.cos(longitude_rad),
                np.cos(latitude_rad) * np.sin(longitude_rad),
                np.sin(latitude_rad),
            ]
        )
        return float(np.rad2deg(np.arccos(np.clip(vector @ target_vector, -1.0, 1.0))))

    refined_error = angular_error(float(serial[0].lats[0]), float(serial[0].lons[0]))
    grid_error = angular_error(grid_latitude, grid_longitude)
    assert refined_error < grid_error
    assert refined_error < 0.1
    np.testing.assert_allclose(serial[0].lats, target[0], atol=0.1, rtol=0.0)
    np.testing.assert_allclose(serial[0].lons, target[1], atol=0.1, rtol=0.0)
    _assert_tracks_equal(serial, dask)


@pytest.mark.integration
def test_hodges_unique_frame_detection_count(sample_msl_dataset: xr.DataArray) -> None:
    """Verify that both serial and Dask Hodges track invocations detect each
    frame exactly once.
    """
    from unittest.mock import patch

    import pystormtracker.hodges.detector as hodges_det_mod
    import pystormtracker.hodges.tracker as hodges_trk_mod

    tracker_serial = HodgesTracker(
        min_track_points=2,
        segment_frames=4,
        backend="serial",
    )

    with patch.object(
        hodges_det_mod,
        "detect_hodges_frame",
        wraps=hodges_det_mod.detect_hodges_frame,
    ) as spy_det:
        res_serial = tracker_serial.track(
            sample_msl_dataset,
            "msl",
            detection_mode="min",
            object_threshold=100000.0,
        )
        assert spy_det.call_count == 12

    tracker_dask = HodgesTracker(
        min_track_points=2,
        segment_frames=4,
        backend="dask",
        frame_workers=4,
        mge_workers=4,
    )

    with patch.object(
        hodges_trk_mod,
        "detect_hodges_frame",
        wraps=hodges_det_mod.detect_hodges_frame,
    ) as spy_det_dask:
        res_dask = tracker_dask.track(
            sample_msl_dataset,
            "msl",
            detection_mode="min",
            object_threshold=100000.0,
        )
        assert spy_det_dask.call_count == 12

    _assert_synthetic_moving_depression(res_serial)
    _assert_tracks_equal(res_serial, res_dask)


@pytest.mark.integration
def test_dask_preserves_lazy_graph_without_eager_full_materialization(
    sample_msl_dataset: xr.DataArray,
) -> None:
    """Verify that Dask backend preserves lazy chunks and does not materialize
    full array values.
    """
    from unittest.mock import patch

    import dask.array as da

    # Ensure dataset is Dask-backed
    da_input = sample_msl_dataset.chunk({"time": 1, "lat": -1, "lon": -1})
    assert isinstance(da_input.data, da.Array)

    # Monkeypatch DataArray.values to disallow full materialization during Dask dispatch
    original_values = xr.DataArray.values.fget  # type: ignore[attr-defined]

    def guarded_values(self: xr.DataArray) -> np.ndarray:
        if self.ndim > 1 and "time" in self.dims and self.sizes["time"] > 1:
            raise RuntimeError(
                f"Full data variable eagerly materialized! Shape: {self.shape}"
            )
        return np.asarray(original_values(self))

    with patch.object(xr.DataArray, "values", property(guarded_values)):
        tracker_dask = HodgesTracker(
            min_track_points=2,
            segment_frames=4,
            backend="dask",
            frame_workers=4,
            mge_workers=4,
        )
        res_dask = tracker_dask.track(
            da_input,
            "msl",
            detection_mode="min",
            object_threshold=100000.0,
        )
        assert len(res_dask) > 0
