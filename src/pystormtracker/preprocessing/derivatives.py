from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.constants import R_EARTH_METERS

try:
    import ducc0  # type: ignore[import-not-found]

    DUCC0_AVAILABLE = True
except ImportError:
    DUCC0_AVAILABLE = False


def compute_relative_vorticity_divergence(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes spatial divergence and relative vorticity from u and v wind components
    using ducc0.

    Args:
        u: Zonal wind (ntheta, nphi).
        v: Meridional wind (ntheta, nphi).
        R: Planetary radius in meters. Default is R_EARTH_METERS.
        lmax: Maximum spherical harmonic degree. If None, derived from ntheta.
        geometry: Grid geometry (default 'CC' for regular lat-lon with poles).
        nthreads: Number of threads. 0 means all available hardware threads.

    Returns:
        div: Divergence (ntheta, nphi)
        vort: Relative vorticity (ntheta, nphi)
    """
    if not DUCC0_AVAILABLE:
        raise ImportError("ducc0 is required for computing vorticity and divergence.")

    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u is {u.shape}, v is {v.shape}")

    ntheta, nphi = u.shape

    if lmax is None:
        if geometry == "CC":
            lmax = ntheta - 2
        elif geometry == "DH":
            lmax = (ntheta - 2) // 2
        else:
            lmax = ntheta - 1

    mmax = min(lmax, (nphi - 1) // 2)

    # Coordinate Transformation
    # v_theta points South, so v_theta = -v
    # v_phi points East, so v_phi = u
    vec_map = np.stack((-v, u), axis=0).astype(np.float64)

    # Spin-1 Analysis
    alm_vec = ducc0.sht.experimental.analysis_2d(
        map=vec_map,
        spin=1,
        lmax=lmax,
        mmax=mmax,
        geometry=geometry,
        nthreads=nthreads,
    )

    alm_E = alm_vec[0]
    alm_B = alm_vec[1]

    # Spectral Scaling
    l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])
    eigen_scale = np.sqrt(l_arr * (l_arr + 1.0)) / R

    alm_div = -eigen_scale * alm_E
    alm_vort = eigen_scale * alm_B

    # Spin-0 Synthesis
    div = ducc0.sht.experimental.synthesis_2d(
        alm=np.expand_dims(alm_div, axis=0),
        spin=0,
        lmax=lmax,
        mmax=mmax,
        ntheta=ntheta,
        nphi=nphi,
        geometry=geometry,
        nthreads=nthreads,
    )[0]

    vort = ducc0.sht.experimental.synthesis_2d(
        alm=np.expand_dims(alm_vort, axis=0),
        spin=0,
        lmax=lmax,
        mmax=mmax,
        ntheta=ntheta,
        nphi=nphi,
        geometry=geometry,
        nthreads=nthreads,
    )[0]

    return div, vort


def apply_wind_derivatives(
    u: xr.DataArray,
    v: xr.DataArray,
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Xarray wrapper for computing relative vorticity and divergence.

    Args:
        u: Zonal wind DataArray (lat, lon). Latitude must be North to South.
        v: Meridional wind DataArray (lat, lon).
        R: Planetary radius in meters. Default is R_EARTH_METERS.
        lmax: Maximum spherical harmonic degree.
        geometry: Grid geometry (default 'CC').
        nthreads: Number of threads.

    Returns:
        div, vort: Divergence and relative vorticity DataArrays.
    """
    div_data, vort_data = compute_relative_vorticity_divergence(
        u.values, v.values, R=R, lmax=lmax, geometry=geometry, nthreads=nthreads
    )

    div = xr.DataArray(div_data, coords=u.coords, dims=u.dims, name="divergence")
    vort = xr.DataArray(
        vort_data, coords=u.coords, dims=u.dims, name="relative_vorticity"
    )

    return div, vort
