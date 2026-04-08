from __future__ import annotations

import numpy as np
import xarray as xr

from pystormtracker.preprocessing.regrid import SpectralRegridder


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
