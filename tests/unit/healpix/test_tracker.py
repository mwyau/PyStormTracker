from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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


def test_healpix_preprocessing_regrids_regular_data() -> None:
    data = xr.DataArray(
        np.ones((1, 9, 16), dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": np.linspace(-90.0, 90.0, 9),
            "lon": np.linspace(0.0, 360.0, 16, endpoint=False),
        },
        name="msl",
    )

    processed = HealpixTracker().preprocess_standard_track(data, lmin=0, lmax=3)

    assert processed.dims == ("time", "cell")
    assert processed.shape == (1, 192)
    assert processed.attrs["map_proj"] == "healpix"
    assert processed.attrs["nside"] == 4
