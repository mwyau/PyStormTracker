from __future__ import annotations

import numpy as np
import xarray as xr

from pystormtracker.models.constants import R_EARTH_KM
from pystormtracker.models.geo import spatial_bounds_from_xarray, stereo_to_latlon


def test_stereo_to_latlon_nh() -> None:
    # Test North Pole
    lat, lon = stereo_to_latlon(0.0, 0.0, hemisphere=1)
    assert np.isclose(lat, 90.0)

    # Test a point on the equator (radius = 2 * R * tan(pi/4) = 2 * R)
    r_eq = 2.0 * R_EARTH_KM

    # Point at x=0, y=r_eq => phi = arctan2(0, -r_eq) = arctan2(0, -1) = pi => lon=180
    lat, lon = stereo_to_latlon(0.0, r_eq, hemisphere=1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 180.0, atol=1e-7)

    # Point at x=r_eq, y=0 => phi = arctan2(r_eq, 0) = pi/2 => lon=90
    lat, lon = stereo_to_latlon(r_eq, 0.0, hemisphere=1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 90.0, atol=1e-7)


def test_stereo_to_latlon_sh() -> None:
    # Test South Pole
    lat, lon = stereo_to_latlon(0.0, 0.0, hemisphere=-1)
    assert np.isclose(lat, -90.0)

    # Test a point on the equator
    r_eq = 2.0 * R_EARTH_KM

    # Point at x=0, y=r_eq => phi = arctan2(0, r_eq) = 0 => lon=0
    lat, lon = stereo_to_latlon(0.0, r_eq, hemisphere=-1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 0.0, atol=1e-7)

    # Point at x=r_eq, y=0 => phi = arctan2(r_eq, 0) = pi/2 => lon=90
    lat, lon = stereo_to_latlon(r_eq, 0.0, hemisphere=-1)
    assert np.isclose(lat, 0.0, atol=1e-7)
    assert np.isclose(lon, 90.0, atol=1e-7)


def test_spatial_bounds_are_independent_of_longitude_storage_order() -> None:
    common = {"time": [np.datetime64("2000-01-01")], "lat": [-10.0, 20.0]}
    ascending = xr.DataArray(
        np.zeros((1, 2, 4)),
        dims=("time", "lat", "lon"),
        coords={**common, "lon": [120.0, 140.0, 160.0, 180.0]},
    )
    descending = ascending.assign_coords(lon=[180.0, 160.0, 140.0, 120.0])

    first = spatial_bounds_from_xarray(ascending)
    second = spatial_bounds_from_xarray(descending)

    assert first == second
    assert first is not None
    assert first.west == 120.0
    assert first.east == -180.0


def test_spatial_bounds_normalize_zero_to_360_and_detect_global_grid() -> None:
    regional = xr.DataArray(
        np.zeros((1, 2, 3)),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": [0.0, 70.0],
            "lon": [120.0, 160.0, 200.0],
        },
    )
    global_grid = xr.DataArray(
        np.zeros((1, 2, 4)),
        dims=("time", "lat", "lon"),
        coords={
            "time": [np.datetime64("2000-01-01")],
            "lat": [0.0, 70.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
    )

    bounds = spatial_bounds_from_xarray(regional)
    assert bounds is not None
    assert bounds.west == 120.0
    assert bounds.east == -160.0
    global_bounds = spatial_bounds_from_xarray(global_grid)
    assert global_bounds is not None
    assert (global_bounds.west, global_bounds.east) == (-180.0, 180.0)
