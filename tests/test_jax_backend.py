from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

# Skip all tests in this module if jax is not installed
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from pystormtracker.preprocessing.spectral import SpectralFilter
from pystormtracker.preprocessing.kinematics import Kinematics

def test_spectral_filter_jax_parity() -> None:
    np.random.seed(42)
    # Test both 1-deg and 2.5-deg
    for ny, nx, tol in [(181, 360, 1e-12), (73, 144, 1e-2)]:
        data: NDArray[np.float64] = np.random.randn(ny, nx)
        
        sf = SpectralFilter(lmin=5, lmax=42)
        
        # Run ducc0 (serial)
        out_ducc0 = sf.filter(data, backend="serial")
        
        # Run jax
        out_jax = sf.filter(data, backend="jax")
        
        # Check for numerical parity
        np.testing.assert_allclose(out_ducc0, out_jax, atol=tol, rtol=tol)

def test_spectral_filter_jax_xarray() -> None:
    ny, nx = 73, 144
    data = np.random.randn(1, ny, nx)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    sf = SpectralFilter(lmin=5, lmax=42)
    filtered = sf.filter(da, backend="jax")

    assert isinstance(filtered, xr.DataArray)
    assert filtered.shape == (1, ny, nx)
    assert filtered.name == "msl_spectral_filtered"

def test_kinematics_jax_parity() -> None:
    np.random.seed(42)
    for ny, nx in [(181, 360), (73, 144)]:
        u = np.random.randn(ny, nx)
        v = np.random.randn(ny, nx)
        
        km = Kinematics(lmax=42)
        
        # Run ducc0
        div_ducc0, vort_ducc0 = km.compute(u, v, backend="serial")
        
        # Run jax
        div_jax, vort_jax = km.compute(u, v, backend="jax")
        
        # Check for scientific parity (relaxed tolerance for coarse grids)
        np.testing.assert_allclose(div_ducc0, div_jax, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(vort_ducc0, vort_jax, atol=2e-5, rtol=2e-5)

def test_kinematics_jax_xarray() -> None:
    ny, nx = 73, 144
    u_data = np.random.randn(ny, nx)
    v_data = np.random.randn(ny, nx)
    
    u = xr.DataArray(
        u_data,
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
    )
    v = xr.DataArray(v_data, dims=["lat", "lon"], coords=u.coords)

    km = Kinematics(lmax=42)
    div, vort = km.compute(u, v, backend="jax")

    assert isinstance(div, xr.DataArray)
    assert isinstance(vort, xr.DataArray)
    assert div.shape == (ny, nx)
    assert vort.shape == (ny, nx)
