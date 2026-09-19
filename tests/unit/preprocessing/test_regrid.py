# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import spharmgrid as sg
import xarray as xr

from pystormtracker.preprocessing.regrid import SpectralRegridder
from pystormtracker.preprocessing.spectral import SHTFilter


def test_regrid_to_grid() -> None:
    # 2.5 degree grid (73 x 144)
    ny, nx = 73, 144
    data = np.random.default_rng().random((ny, nx))
    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="test_var",
    )

    regridder = SpectralRegridder()
    regridded = regridder.to_grid(da, 37, 72)

    assert regridded.shape == (37, 72)
    assert regridded.dims == ("lat", "lon")
    assert regridded.name == "test_var"


def test_regrid_to_grid_delegates_regular_path_to_spharmgrid() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    data = xr.DataArray(
        np.random.default_rng(5).random((9, 18)),
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
        name="test_var",
    )
    target = sg.gaussian_grid(8, 16, latitude_order="ascending")
    expected = sg.regrid(data, target, truncation="T3", sht_threads=None)

    with patch(
        "pystormtracker.preprocessing.regrid.sg.regrid",
        wraps=sg.regrid,
    ) as regrid_call:
        actual = SpectralRegridder(lmax=3).to_grid(data, 8, 16, out_geometry="GL")

    np.testing.assert_allclose(actual.values, expected.values)
    assert regrid_call.call_args is not None
    assert regrid_call.call_args.args[0] is data
    assert regrid_call.call_args.kwargs["truncation"] == "T3"
    assert regrid_call.call_args.kwargs["sht_threads"] is None


def test_regrid_to_grid_default_bandwidth_matches_spharmgrid() -> None:
    source = sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
    target = sg.gaussian_grid(8, 16, latitude_order="ascending")
    data = xr.DataArray(
        np.random.default_rng(17).normal(
            size=(source.latitude.size, source.longitude.size)
        ),
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
        name="test_var",
    )

    actual = SpectralRegridder().to_grid(
        data,
        nlat=8,
        nlon=16,
        in_geometry="CC",
        out_geometry="GL",
    )
    expected = sg.regrid(data, target, sht_threads=None)

    np.testing.assert_allclose(actual.values, expected.values)
    np.testing.assert_array_equal(actual.lat.values, expected.lat.values)
    np.testing.assert_array_equal(actual.lon.values, expected.lon.values)


@pytest.mark.parametrize("source_order", ["ascending", "descending"])
def test_regrid_regular_field_coordinates_match_values(
    source_order: Literal["ascending", "descending"],
) -> None:
    source = sg.clenshaw_curtis_grid(17, 36, latitude_order=source_order)
    data = xr.DataArray(
        np.sin(np.deg2rad(source.latitude))[:, None] * np.ones(source.longitude.size),
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
        name="test_var",
    )
    target = sg.gaussian_grid(8, 16, latitude_order="ascending")

    actual = SpectralRegridder(lmax=3).to_grid(
        data,
        nlat=8,
        nlon=16,
        in_geometry="CC",
        out_geometry="GL",
        lat_reverse=False,
    )
    expected = sg.regrid(
        data,
        target,
        truncation="T3",
        sht_threads=None,
    )

    np.testing.assert_allclose(actual.values, expected.values)
    np.testing.assert_array_equal(actual.lat.values, expected.lat.values)
    np.testing.assert_array_equal(actual.lon.values, expected.lon.values)
    np.testing.assert_allclose(
        actual.mean("lon").values,
        np.sin(np.deg2rad(actual.lat.values)),
        atol=1.0e-10,
    )


def test_regrid_invalid_rectangular_grid_does_not_fall_back() -> None:
    data = xr.DataArray(
        np.zeros((9, 18)),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(-80.0, 80.0, 9),
            "lon": np.linspace(0.0, 360.0, 18, endpoint=False),
        },
    )

    with pytest.raises(ValueError, match=r".+"):
        SpectralRegridder(lmax=3).to_grid(data, nlat=8, nlon=16)


def test_regrid_mmax_without_lmax_requires_lmax() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    data = xr.DataArray(
        np.sin(np.deg2rad(source.latitude))[:, None]
        * np.cos(np.deg2rad(source.longitude))[None, :],
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
    )

    with (
        patch(
            "pystormtracker.preprocessing.regrid.sg.regrid",
            wraps=sg.regrid,
        ) as regrid_call,
        pytest.raises(
            ValueError,
            match="lmax is required.*mmax.*regular CC/GL regridding",
        ),
    ):
        SpectralRegridder(mmax=3).to_grid(data, nlat=8, nlon=16)

    regrid_call.assert_not_called()


def test_regrid_equal_lmax_mmax_matches_spharmgrid() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    target = sg.gaussian_grid(8, 16, latitude_order="ascending")
    data = xr.DataArray(
        np.sin(np.deg2rad(source.latitude))[:, None]
        * np.cos(np.deg2rad(source.longitude))[None, :],
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
        name="test_var",
    )

    expected = sg.regrid(data, target, truncation="T3", sht_threads=None)
    actual = SpectralRegridder(lmax=3, mmax=3).to_grid(
        data,
        nlat=8,
        nlon=16,
        out_geometry="GL",
    )

    np.testing.assert_allclose(actual.values, expected.values)
    np.testing.assert_array_equal(actual["lat"], expected["lat"])
    np.testing.assert_array_equal(actual["lon"], expected["lon"])


def test_regrid_nontriangular_mmax_is_rejected() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    data = xr.DataArray(
        np.sin(np.deg2rad(source.latitude))[:, None]
        * np.cos(np.deg2rad(source.longitude))[None, :],
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
    )

    with pytest.raises(ValueError, match="non-triangular"):
        SpectralRegridder(lmax=5, mmax=2).to_grid(data, 8, 16)


def test_regrid_to_grid_regular_dask_path_stays_lazy() -> None:
    source = sg.clenshaw_curtis_grid(9, 18, latitude_order="ascending")
    data = xr.DataArray(
        np.random.default_rng(6).random((9, 18)),
        dims=("lat", "lon"),
        coords={"lat": source.latitude, "lon": source.longitude},
        name="test_var",
    ).chunk({"lat": -1, "lon": -1})

    actual = SpectralRegridder(lmax=3).to_grid(
        data,
        8,
        16,
        out_geometry="GL",
    )

    assert hasattr(actual.data, "dask")


def test_regrid_to_healpix() -> None:
    # 2.5 degree grid (73 x 144)
    ny, nx = 73, 144
    data = np.random.default_rng().random((ny, nx))
    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="test_var",
    )

    regridder = SpectralRegridder()
    nside = 16
    regridded = regridder.to_healpix(da, nside=nside)

    npix = 12 * nside**2
    assert regridded.shape == (npix,)
    assert regridded.dims == ("cell",)
    assert regridded.name == "test_var"
    assert len(regridded.cell) == npix


def test_filter_reduced_gaussian_grid(
    reduced_gaussian_data: xr.DataArray,
) -> None:
    filtered = SHTFilter(
        lmin=0,
        lmax=3,
        out_geometry="CC",
        out_ntheta=8,
        out_nphi=16,
    ).filter(reduced_gaussian_data)

    assert filtered.dims == ("time", "latitude", "longitude")
    assert filtered.shape == (1, 8, 16)
    assert np.isfinite(filtered).all()


def test_regrid_reduced_gaussian_to_regular(
    reduced_gaussian_data: xr.DataArray,
) -> None:
    regridder = SpectralRegridder(lmax=3)
    regridded = regridder.to_grid(
        reduced_gaussian_data.isel(time=0),
        nlat=8,
        nlon=16,
        in_geometry="GL",
    )

    assert regridded.dims == ("lat", "lon")
    assert regridded.shape == (8, 16)
    assert np.isfinite(regridded).all()


def test_regrid_identity() -> None:
    # Test that regridding to the same resolution results in small residuals
    # (Though spectral interpolation isn't perfectly identity if lmax is small)
    ny, nx = 73, 144
    lmax = 42
    # Create a band-limited signal
    lon = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    lat = np.linspace(-np.pi / 2, np.pi / 2, ny)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    data = np.sin(2 * lon_grid) * np.cos(lat_grid)  # Simple wave

    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="test_var",
    )

    regridder = SpectralRegridder(lmax=lmax)
    # Use lat_reverse=False for South to North data
    regridded = regridder.to_grid(da, nlat=ny, nlon=nx, lat_reverse=False)

    # We expect some difference because of spectral truncation
    # but it should be small for a simple wave
    np.testing.assert_allclose(da.values, regridded.values, rtol=2e-2, atol=2e-2)


def test_regrid_to_polar_stereo() -> None:
    # 2.5 degree grid (73 x 144)
    ny, nx = 73, 144
    data = np.random.default_rng().random((ny, nx))
    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="test_var",
    )

    regridder = SpectralRegridder()
    regridded = regridder.to_polar_stereo(
        da,
        hemisphere="nh",
        extent=(-1000.0, 1000.0, -1000.0, 1000.0),
        stereo_grid_spacing_km=100.0,
    )

    assert regridded.shape == (21, 21)
    assert regridded.dims == ("y", "x")
    assert regridded.name == "test_var"
    assert regridded.attrs["projection"] == "nh_stereo"
    assert regridded.attrs["stereo_grid_spacing_km"] == 100.0
    assert len(regridded.y) == 21
    assert len(regridded.x) == 21


def test_regrid_to_polar_stereo_lmax_override() -> None:
    da = xr.DataArray(
        np.ones((73, 144), dtype=np.float64),
        dims=("lat", "lon"),
        coords={
            "lat": np.linspace(-90.0, 90.0, 73),
            "lon": np.linspace(0.0, 360.0, 144, endpoint=False),
        },
    )

    regridded = SpectralRegridder().to_polar_stereo(
        da,
        transform_lmax=7,
        extent=(-100.0, 100.0, -100.0, 100.0),
        stereo_grid_spacing_km=100.0,
    )

    assert regridded.attrs["lmax"] == 7


def test_regrid_to_polar_stereo_with_filter() -> None:
    # 2.5 degree grid
    ny, nx = 73, 144
    # Create a simple field with a low frequency component (l=1)
    # and some noise
    lat = np.linspace(-90, 90, ny)
    lon = np.linspace(0, 360, nx, endpoint=False)
    LAT, _ = np.meshgrid(lat, lon, indexing="ij")
    data = np.sin(np.radians(LAT)) + 0.1 * np.random.default_rng().random((ny, nx))

    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="test_var",
    )

    regridder = SpectralRegridder()
    # No filter
    regridded_raw = regridder.to_polar_stereo(
        da,
        hemisphere="nh",
        extent=(-1000.0, 1000.0, -1000.0, 1000.0),
        transform_lmax=42,
    )
    filtered = SHTFilter(lmin=5, lmax=42).filter(da)
    regridded_filtered = regridder.to_polar_stereo(
        filtered,
        hemisphere="nh",
        extent=(-1000.0, 1000.0, -1000.0, 1000.0),
        transform_lmax=42,
    )

    # The filtered field should have significantly lower mean/variance
    # if low wavenumbers dominate
    assert not np.allclose(regridded_raw.values, regridded_filtered.values)
