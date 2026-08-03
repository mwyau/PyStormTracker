from __future__ import annotations

import numpy as np
import xarray as xr

from pystormtracker.preprocessing.regrid import SpectralRegridder
from pystormtracker.preprocessing.spectral import SHTFilter, apply_sht_filter


def test_regrid_to_grid() -> None:
    # 2.5 degree grid (73 x 144)
    ny, nx = 73, 144
    data = np.random.rand(ny, nx)
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
    # Regrid to 1.0 degree (181 x 360)
    out_ny, out_nx = 181, 360
    regridded = regridder.to_grid(da, nlat=out_ny, nlon=out_nx)

    assert regridded.shape == (out_ny, out_nx)
    assert regridded.dims == ("lat", "lon")
    assert regridded.name == "test_var"
    assert len(regridded.lat) == out_ny
    assert len(regridded.lon) == out_nx


def test_regrid_to_healpix() -> None:
    # 2.5 degree grid (73 x 144)
    ny, nx = 73, 144
    data = np.random.rand(ny, nx)
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
    filtered = apply_sht_filter(
        reduced_gaussian_data,
        lmin=0,
        lmax=3,
        out_geometry="CC",
        out_ntheta=8,
        out_nphi=16,
    )

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
    data = np.random.rand(ny, nx)
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
        da, hemisphere="nh", extent=(-1000.0, 1000.0, -1000.0, 1000.0), resolution=100.0
    )

    assert regridded.shape == (21, 21)
    assert regridded.dims == ("y", "x")
    assert regridded.name == "test_var"
    assert regridded.attrs["projection"] == "nh_stereo"
    assert regridded.attrs["resolution_km"] == 100.0
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
        resolution=100.0,
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
    data = np.sin(np.radians(LAT)) + 0.1 * np.random.rand(ny, nx)

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
