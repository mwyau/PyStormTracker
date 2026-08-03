from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.simple.tracker import SimpleTracker


@pytest.fixture
def global_data() -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 73, 144), dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": np.linspace(-90.0, 90.0, 73),
            "lon": np.linspace(0.0, 360.0, 144, endpoint=False),
        },
        name="msl",
    )


@pytest.mark.parametrize("tracker", [SimpleTracker(), HodgesTracker()])
def test_global_auto_filter_uses_longitude_for_sht_selection(
    tracker: SimpleTracker | HodgesTracker, global_data: xr.DataArray
) -> None:
    with (
        patch("pystormtracker.preprocessing.spectral.SHTFilter") as sht_filter,
        patch("pystormtracker.preprocessing.spectral.DCTFilter") as dct_filter,
    ):
        sht_filter.return_value.filter.return_value = global_data
        tracker.preprocess_standard_track(global_data, lmin=0, lmax=7)

    sht_filter.assert_called_once_with(lmin=0, lmax=7)
    dct_filter.assert_not_called()


@pytest.mark.parametrize("tracker", [SimpleTracker(), HodgesTracker()])
def test_coarse_global_auto_filter_uses_sht(
    tracker: SimpleTracker | HodgesTracker,
) -> None:
    coarse_global = xr.DataArray(
        np.zeros((1, 7, 6), dtype=np.float64),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": np.linspace(-90.0, 90.0, 7),
            "lon": np.arange(0.0, 360.0, 60.0),
        },
        name="msl",
    )
    with (
        patch("pystormtracker.preprocessing.spectral.SHTFilter") as sht_filter,
        patch("pystormtracker.preprocessing.spectral.DCTFilter") as dct_filter,
    ):
        sht_filter.return_value.filter.return_value = coarse_global
        tracker.preprocess_standard_track(coarse_global, lmin=0, lmax=2)

    sht_filter.assert_called_once_with(lmin=0, lmax=2)
    dct_filter.assert_not_called()


@pytest.mark.parametrize("tracker", [SimpleTracker(), HodgesTracker()])
def test_regional_auto_filter_uses_dct(
    tracker: SimpleTracker | HodgesTracker, global_data: xr.DataArray
) -> None:
    regional = global_data.sel(lon=slice(0.0, 100.0))
    with (
        patch("pystormtracker.preprocessing.spectral.SHTFilter") as sht_filter,
        patch("pystormtracker.preprocessing.spectral.DCTFilter") as dct_filter,
    ):
        dct_filter.return_value.filter.return_value = regional
        tracker.preprocess_standard_track(regional, lmin=0, lmax=7)

    dct_filter.assert_called_once_with(lmin=0, lmax=7)
    sht_filter.assert_not_called()


@pytest.mark.parametrize("tracker", [SimpleTracker(), HodgesTracker()])
@pytest.mark.parametrize("map_proj", ["nh_stereo", "sh_stereo"])
def test_polar_preprocessing_uses_requested_lmax(
    tracker: SimpleTracker | HodgesTracker,
    map_proj: str,
    global_data: xr.DataArray,
) -> None:
    processed, _steps = tracker.preprocess_standard_track(
        global_data,
        lmin=0,
        lmax=7,
        map_proj=map_proj,  # type: ignore[arg-type]
        extent=(-100.0, 100.0, -100.0, 100.0),
        resolution=100.0,
    )

    assert processed.dims == ("time", "y", "x")
    assert processed.attrs["map_proj"] == map_proj
    assert processed.attrs["lmax"] == 7
