from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.metrics.eulerian import (
    compute_eke,
    compute_high_wind_index,
    compute_variance_metric,
)


@pytest.fixture
def sample_wind_ds() -> xr.Dataset:
    """Creates a sample dataset with u and v wind components."""
    times = np.arange(0, 100) * np.timedelta64(6, "h") + np.datetime64("2020-01-01")
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(0, 350, 20)

    shape = (len(times), len(lats), len(lons))

    # Constant winds for simple validation
    u = xr.DataArray(
        np.full(shape, 10.0),
        coords=[times, lats, lons],
        dims=["time", "lat", "lon"],
    )
    v = xr.DataArray(
        np.full(shape, 0.0),
        coords=[times, lats, lons],
        dims=["time", "lat", "lon"],
    )

    return xr.Dataset({"u10": u, "v10": v})


def test_compute_high_wind_index(sample_wind_ds: xr.Dataset) -> None:
    # 95th percentile of constant 10 m/s wind should be 10.0
    hwi = compute_high_wind_index(sample_wind_ds, "u10", "v10", percentile=0.95)
    assert hwi.name == "high_wind_index"
    assert np.allclose(hwi.values, 10.0)
    assert "time" in hwi.coords

    # Test custom percentile (e.g., 50th percentile)
    hwi_50 = compute_high_wind_index(sample_wind_ds, "u10", "v10", percentile=0.5)
    assert np.allclose(hwi_50.values, 10.0)


def test_compute_variance_metric() -> None:
    # Create a sine wave in time
    times = np.arange(0, 200) * np.timedelta64(6, "h") + np.datetime64("2020-01-01")
    lats = [0.0]
    lons = [0.0]

    # Simple data: 0, 1, 0, -1, 0 ... every 6 hours
    # 24-h difference will be X(t+4) - X(t)
    # If period is exactly 24 hours, diff is 0.
    data = np.sin(np.arange(len(times)) * np.pi / 2)  # Period = 4 steps = 24 hours
    da = xr.DataArray(
        data[:, None, None],
        coords=[times, lats, lons],
        dims=["time", "lat", "lon"],
        name="test_var",
    )

    # With 24-h filter and 24-h period, Var should be 0
    var = compute_variance_metric(da)
    assert np.allclose(var.values, 0.0)

    # Note: test_compute_variance_metric no longer tests 12-h filter
    # as it's hardcoded to 24-h now.


def test_compute_eke(sample_wind_ds: xr.Dataset) -> None:
    # Constant winds mean variance is 0, so EKE should be 0
    eke = compute_eke(sample_wind_ds["u10"], sample_wind_ds["v10"])
    assert eke.name == "eke"
    assert np.allclose(eke.values, 0.0)
