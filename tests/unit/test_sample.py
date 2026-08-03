from __future__ import annotations

import argparse

import numpy as np
import pytest
import xarray as xr

from pystormtracker.models.tracks import Tracks, TracksBuilder, TracksMetadata
from pystormtracker.sample import main, sample_tracks


@pytest.fixture
def dummy_dataset() -> xr.Dataset:
    """Creates a synthetic dataset for testing sampling."""
    lat = np.arange(-90, 91, 10)
    lon = np.arange(0, 360, 10)
    time = [np.datetime64("2020-01-01T00:00")]

    # Create a simple gradient: value = lat + lon/10
    data = np.zeros((1, len(lat), len(lon)))
    for i, lt in enumerate(lat):
        for j, ln in enumerate(lon):
            data[0, i, j] = lt + ln / 10.0

    ds = xr.Dataset(
        data_vars={"test_var": (("time", "lat", "lon"), data)},
        coords={"lat": lat, "lon": lon, "time": time},
    )
    return ds


@pytest.fixture
def dummy_tracks() -> Tracks:
    """Creates a dummy track for testing."""
    times = []
    lats = []
    lons = []
    # Point exactly on a grid point (-10, 20) -> value = -10 + 20/10 = -8
    times.append(np.datetime64("2020-01-01T00:00"))
    lats.append(-10.0)
    lons.append(20.0)
    # Point between grid points (5, 15)
    # Neighbors: (0, 10), (0, 20), (10, 10), (10, 20)
    # Values: (0+1=1), (0+2=2), (10+1=11), (10+2=12)
    # Nearest to (10, 20) is (10, 20) -> value 12
    # Bilinear: average of 1, 2, 11, 12 = 6.5
    times.append(np.datetime64("2020-01-01T01:00"))
    lats.append(5.0)
    lons.append(15.0)
    builder = TracksBuilder(TracksMetadata("intensity", "max", {"intensity": "1"}))
    builder.add_track(1, times, lats, lons, {"intensity": [0.0, 0.0]})
    return builder.finish()


def test_sample_nearest(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    tracks = sample_tracks(dummy_tracks, dummy_dataset, "test_var", method="nearest")

    # First point: exact
    assert tracks[0][0].vars["test_var"] == -8.0

    # Second point: nearest to (10, 20) or (0, 10)?
    # (5, 15) is equidistant. xarray's nearest might pick one.
    val = tracks[0][1].vars["test_var"]
    assert val in [1.0, 2.0, 11.0, 12.0]


def test_sample_bilinear(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    tracks = sample_tracks(dummy_tracks, dummy_dataset, "test_var", method="bilinear")

    # First point: exact
    assert tracks[0][0].vars["test_var"] == -8.0

    # Second point: (5, 15) should be 6.5
    assert tracks[0][1].vars["test_var"] == pytest.approx(6.5)


def test_sample_max_radius(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    # Radius of 1200 km around (5, 15)
    # Should include several points.
    tracks = sample_tracks(
        dummy_tracks, dummy_dataset, "test_var", method="max", radius_km=1200.0
    )

    val = tracks[0][1].vars["test_var"]
    # At least the nearest neighbor values should be considered
    assert val >= 12.0


def test_sample_mean_radius(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    tracks = sample_tracks(
        dummy_tracks, dummy_dataset, "test_var", method="mean", radius_km=1000.0
    )
    val = tracks[0][1].vars["test_var"]
    assert not np.isnan(val)


def test_sample_invalid_var(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="Variable 'invalid' not found"):
        sample_tracks(dummy_tracks, dummy_dataset, "invalid")


def test_sample_output_name(dummy_tracks: Tracks, dummy_dataset: xr.Dataset) -> None:
    tracks = sample_tracks(
        dummy_tracks, dummy_dataset, "test_var", output_varname="new_name"
    )
    assert "new_name" in tracks[0][0].vars
    assert tracks[0][0].vars["new_name"] == -8.0


def test_sample_rejects_negative_radius(
    dummy_tracks: Tracks, dummy_dataset: xr.Dataset
) -> None:
    with pytest.raises(ValueError, match="radius must be nonnegative"):
        sample_tracks(
            dummy_tracks,
            dummy_dataset,
            "test_var",
            radius_km=-1.0,
        )


def test_spatial_aggregation_requires_positive_radius() -> None:
    args = argparse.Namespace(
        input="unused.json",
        data="unused.nc",
        var="msl",
        output="unused.json",
        method="mean",
        radius=0.0,
        name=None,
        engine=None,
    )
    with pytest.raises(ValueError, match="requires a positive radius"):
        main(args)
