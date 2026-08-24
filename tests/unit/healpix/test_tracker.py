from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.healpix.tracker import HealpixTracker


def test_healpix_tracker_invalid_nside() -> None:
    with pytest.raises(ValueError, match="nside must be a positive power of two"):
        HealpixTracker(nside=3)


def test_healpix_tracker_time_range() -> None:
    tracker = HealpixTracker()
    with pytest.raises((FileNotFoundError, Exception)) as excinfo:
        tracker.track(
            data="nonexistent.nc",
            variable="msl",
            start_time="2025-01-01",
            end_time="2025-01-31",
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

    processed, steps = HealpixTracker()._preprocess_standard_track(data, lmin=0, lmax=3)

    assert processed.dims == ("time", "cell")
    assert processed.shape == (1, 192)
    assert processed.attrs["projection"] == "healpix"
    assert any(step.operation == "regrid" for step in steps)
    assert processed.attrs["nside"] == 4
