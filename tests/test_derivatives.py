from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing.derivatives import (
    apply_wind_derivatives,
    compute_relative_vorticity_divergence,
)


def test_compute_vorticity_divergence_shapes() -> None:
    pytest.importorskip("ducc0")

    ntheta, nphi = 73, 144
    u = np.random.rand(ntheta, nphi)
    v = np.random.rand(ntheta, nphi)

    div, vort = compute_relative_vorticity_divergence(u, v)

    assert div.shape == (ntheta, nphi)
    assert vort.shape == (ntheta, nphi)

    # Check that they aren't all zeros
    assert np.any(div != 0.0)
    assert np.any(vort != 0.0)


def test_apply_wind_derivatives() -> None:
    pytest.importorskip("ducc0")

    ntheta, nphi = 37, 72
    u = xr.DataArray(
        np.random.rand(ntheta, nphi),
        coords={
            "lat": np.linspace(90, -90, ntheta),
            "lon": np.linspace(0, 360, nphi, endpoint=False),
        },
        dims=["lat", "lon"],
    )
    v = xr.DataArray(
        np.random.rand(ntheta, nphi),
        coords={
            "lat": np.linspace(90, -90, ntheta),
            "lon": np.linspace(0, 360, nphi, endpoint=False),
        },
        dims=["lat", "lon"],
    )

    div, vort = apply_wind_derivatives(u, v)

    assert div.dims == ("lat", "lon")
    assert vort.dims == ("lat", "lon")
    assert div.name == "divergence"
    assert vort.name == "relative_vorticity"
    assert np.array_equal(div.lat, u.lat)
    assert np.array_equal(vort.lon, u.lon)


def test_solid_body_rotation() -> None:
    pytest.importorskip("ducc0")

    # Solid body rotation: u = U0 * cos(lat)
    # Vorticity is related to the rotation rate.
    # We just ensure it runs without error and gives a physically sound field
    # (vorticity should be symmetric/anti-symmetric depending on wind setup).
    ntheta, nphi = 73, 144
    lat = np.linspace(np.pi / 2, -np.pi / 2, ntheta)
    lon = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    _lon_grid, lat_grid = np.meshgrid(lon, lat)

    u = np.cos(lat_grid) * 10.0
    v = np.zeros_like(u)

    div, vort = compute_relative_vorticity_divergence(u, v, nthreads=1)

    # Divergence of solid body rotation should be very close to zero
    np.testing.assert_allclose(div, 0, atol=1e-10)

    # Vorticity is non-zero
    assert np.max(np.abs(vort)) > 0
