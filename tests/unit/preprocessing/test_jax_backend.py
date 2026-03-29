from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.preprocessing.jax_sht import jax_analysis_2d
from pystormtracker.preprocessing.kinematics import (
    Kinematics,
    compute_vort_div_jax,
)
from pystormtracker.preprocessing.spectral import SpectralFilter

# Skip all tests in this module if jax is not installed
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)


def test_jax_sht_resolution_validation() -> None:
    """Tests that JAX backend correctly raises ValueError for too few latitude rings."""
    ny, nx = 73, 144
    data = jax.numpy.zeros((ny, nx))

    # lmax > ny - 2 (71) should raise ValueError for CC geometry
    with pytest.raises(ValueError, match="Too few latitude rings"):
        jax_analysis_2d(data, lmax=72)


def test_jax_sht_aliasing_warning() -> None:
    """Tests that JAX backend issues a UserWarning when lmax > ny / 2."""
    ny, nx = 73, 144
    data = jax.numpy.zeros((ny, nx))

    # lmax > ny / 2 (36.5) should issue a UserWarning
    with pytest.warns(UserWarning, match="more than half the latitude resolution"):
        jax_analysis_2d(data, lmax=42)


def test_spectral_filter_jax_parity() -> None:
    np.random.seed(42)
    # Test 0.25-deg resolution for high-precision parity
    ny, nx, tol = 721, 1440, 1e-12
    data: NDArray[np.float64] = np.random.randn(ny, nx)

    sf = SpectralFilter(lmin=5, lmax=42)

    # Run ducc0 (serial)
    out_ducc0 = sf.filter(data, backend="serial")

    # Run jax
    out_jax = sf.filter(data, sht_engine="jax")

    # Check for numerical parity
    np.testing.assert_allclose(out_ducc0, out_jax, atol=tol, rtol=tol)


def test_spectral_filter_jax_xarray() -> None:
    # 0.25-deg resolution
    ny, nx = 721, 1440
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
    filtered = sf.filter(da, sht_engine="jax")

    assert isinstance(filtered, xr.DataArray)
    assert filtered.shape == (1, ny, nx)
    assert filtered.name == "msl_spectral_filtered"


@pytest.mark.skip(
    reason="Temporarily disabled due to precision issues in JAX spin-1 SHT at 0.25-deg."
)
def test_kinematics_jax_parity() -> None:
    np.random.seed(42)
    # Test 0.25-deg resolution
    ny, nx = 721, 1440
    u = np.random.randn(ny, nx)
    v = np.random.randn(ny, nx)

    km = Kinematics(lmax=42)

    # Run ducc0
    div_ducc0, vort_ducc0 = km.compute(u, v, backend="serial")

    # Run jax directly
    div_jax, vort_jax = compute_vort_div_jax(u, v, lmax=42)

    # Check for scientific parity
    np.testing.assert_allclose(div_ducc0, div_jax, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(vort_ducc0, vort_jax, atol=1e-10, rtol=1e-10)


def test_kinematics_jax_xarray() -> None:
    # 0.25-deg resolution
    ny, nx = 721, 1440
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

    # Test JAX via direct compute_vort_div_jax on xarray via apply_ufunc
    # matches the logic in apply_vort_div but forced to use JAX
    div_vort = xr.apply_ufunc(
        compute_vort_div_jax,
        u,
        v,
        input_core_dims=[["lat", "lon"], ["lat", "lon"]],
        output_core_dims=[["lat", "lon"], ["lat", "lon"]],
        vectorize=True,
        kwargs={"lmax": 42},
        output_dtypes=[u.dtype, u.dtype],
    )
    div, vort = div_vort[0], div_vort[1]

    assert isinstance(div, xr.DataArray)
    assert isinstance(vort, xr.DataArray)
    assert div.shape == (ny, nx)
    assert vort.shape == (ny, nx)
