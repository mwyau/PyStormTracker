from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing.kinematics import (
    Kinematics,
    apply_vort_div,
    compute_vort_div,
)


def test_compute_vort_div_shapes() -> None:
    pytest.importorskip("ducc0")

    ntheta, nphi = 73, 144
    u = np.random.rand(ntheta, nphi)
    v = np.random.rand(ntheta, nphi)

    div, vort = compute_vort_div(u, v)

    assert div.shape == (ntheta, nphi)
    assert vort.shape == (ntheta, nphi)

    # Check that they aren't all zeros
    assert np.any(div != 0.0)
    assert np.any(vort != 0.0)


def test_apply_vort_div() -> None:
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

    div, vort = apply_vort_div(u, v)

    assert div.dims == ("lat", "lon")
    assert vort.dims == ("lat", "lon")
    assert div.name == "divergence"
    assert vort.name == "relative_vorticity"
    assert np.array_equal(div.lat, u.lat)
    assert np.array_equal(vort.lon, u.lon)


def test_kinematics_class() -> None:
    pytest.importorskip("ducc0")

    ntheta, nphi = 37, 72
    u_np = np.random.rand(ntheta, nphi)
    v_np = np.random.rand(ntheta, nphi)

    calc = Kinematics()
    div_np, vort_np = calc.compute(u_np, v_np)

    assert div_np.shape == (ntheta, nphi)
    assert vort_np.shape == (ntheta, nphi)

    u_xr = xr.DataArray(
        u_np,
        coords={
            "lat": np.linspace(90, -90, ntheta),
            "lon": np.linspace(0, 360, nphi, endpoint=False),
        },
        dims=["lat", "lon"],
    )
    v_xr = xr.DataArray(
        v_np,
        coords={
            "lat": np.linspace(90, -90, ntheta),
            "lon": np.linspace(0, 360, nphi, endpoint=False),
        },
        dims=["lat", "lon"],
    )

    div_xr, _vort_xr = calc.compute(u_xr, v_xr)
    assert isinstance(div_xr, xr.DataArray)
    np.testing.assert_allclose(div_xr.values, div_np)


def test_solid_body_rotation() -> None:
    pytest.importorskip("ducc0")

    # Solid body rotation: u = U0 * cos(lat)
    ntheta, nphi = 73, 144
    lat = np.linspace(np.pi / 2, -np.pi / 2, ntheta)
    lon = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    _lon_grid, lat_grid = np.meshgrid(lon, lat)

    u = np.cos(lat_grid) * 10.0
    v = np.zeros_like(u)

    div, vort = compute_vort_div(u, v, nthreads=1)

    # Divergence of solid body rotation should be very close to zero
    np.testing.assert_allclose(div, 0, atol=1e-10)

    # Vorticity is non-zero
    assert np.max(np.abs(vort)) > 0
