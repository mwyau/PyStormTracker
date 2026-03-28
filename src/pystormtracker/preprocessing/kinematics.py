from __future__ import annotations

from typing import Literal, TypedDict, cast, overload

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.constants import R_EARTH_METERS


class KinematicsKwargs(TypedDict, total=False):
    R: float
    lmax: int | None
    geometry: str
    nthreads: int
    backend: str


def compute_vort_div_jax(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes spatial divergence and relative vorticity using JAX and ducc0-parity SHT.
    """
    try:
        import jax
        from .jax_sht import jax_analysis_2d, jax_synthesis_2d
        jax.config.update("jax_enable_x64", True)
    except ImportError as e:
        raise ImportError(
            "The 'jax' backend requires 'jax'. "
            "Install via 'pip install pystormtracker[jax]'."
        ) from e

    ny, nx = u.shape
    if lmax is None:
        lmax = ny - 2  # Standard for ducc0 CC
        
    mmax = min(lmax, nx // 2 - 1)

    # ducc0 spin-1 expects (v_theta, v_phi) = (v, u)
    vec_jax = jax.numpy.stack([v, u], axis=0)
    
    # Forward spin-1 transform
    alm_e, alm_b = jax_analysis_2d(vec_jax, lmax, mmax=mmax, geometry=geometry, spin=1)

    # Spectral Scaling:
    # l_arr contains the degree 'l' for each coefficient in the alm array.
    l_list = []
    for m in range(mmax + 1):
        l_start = max(m, 1)
        if l_start <= lmax:
            l_list.append(jax.numpy.arange(l_start, lmax + 1))
    l_arr = jax.numpy.concatenate(l_list)
    
    eigen_scale = jax.numpy.sqrt(l_arr * (l_arr + 1.0)) / R
    alm_div_active = -eigen_scale * alm_e
    alm_vort_active = eigen_scale * alm_b

    # Synthesis (Scalar transforms for divergence and vorticity)
    # Scalar SHT expects coefficients starting from l=0 for all m.
    # Our spin-1 analysis returned l starting from max(m, 1).
    # Since l=0 has no div/vort (eigen_scale is 0 anyway), we just pad.
    
    # Reconstruct full alm for spin-0 synthesis:
    div_list = []
    vort_list = []
    curr_active = 0
    for m in range(mmax + 1):
        l_start_active = max(m, 1)
        n_l_active = lmax - l_start_active + 1
        
        # Pad l < l_start_active (i.e., l=0 if m=0)
        # For m=0, l_start_active=1, so we pad 1 element (l=0).
        # For m>0, l_start_active=m, so we pad m elements (l=0..m-1).
        n_pad = l_start_active - 0 # assuming we want l starting from 0
        
        div_m = jax.numpy.concatenate([
            jax.numpy.zeros(n_pad, dtype=alm_div_active.dtype),
            alm_div_active[curr_active : curr_active + n_l_active]
        ])
        vort_m = jax.numpy.concatenate([
            jax.numpy.zeros(n_pad, dtype=alm_vort_active.dtype),
            alm_vort_active[curr_active : curr_active + n_l_active]
        ])
        div_list.append(div_m)
        vort_list.append(vort_m)
        curr_active += n_l_active
        
    alm_div = jax.numpy.concatenate(div_list)
    alm_vort = jax.numpy.concatenate(vort_list)

    div = jax_synthesis_2d(alm_div, ny, nx, lmax, mmax=mmax, geometry=geometry, spin=0)
    vort = jax_synthesis_2d(alm_vort, ny, nx, lmax, mmax=mmax, geometry=geometry, spin=0)

    return np.asarray(div, dtype=np.float64), np.asarray(vort, dtype=np.float64)


def compute_vort_div(
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
        geometry: Grid geometry (for ducc0). Default 'CC'.
        nthreads: Number of threads (for ducc0).

    Returns:
        div: Divergence (ntheta, nphi)
        vort: Relative vorticity (ntheta, nphi)
    """
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

    # mmax calculation follows the sampling theorem: for a longitude grid of
    # size nphi, the maximum resolvable wavenumber m is (nphi-1)//2 to avoid
    # aliasing.
    mmax = min(lmax, (nphi - 1) // 2)

    # Standard spectral derivation from wind components:
    # For a vector field V = (u, v) on a sphere:
    # div = 1/(R cos lat) * (du/dlon + d(v cos lat)/dlat)
    # vo  = 1/(R cos lat) * (dv/dlon - d(u cos lat)/dlat)
    #
    # In spherical harmonic space, using E (divergence-like) and
    # B (vorticity-like) modes:
    # div_lm = -[sqrt(l(l+1))/R] * E_lm
    # vort_lm = [sqrt(l(l+1))/R] * B_lm
    # where the [sqrt(l(l+1))/R] term is the eigenvalue of the vector Laplacian.
    # Analysis:
    # ducc0.sht.analysis_2d expects (v_theta, v_phi) for spin-1 (vector) fields.
    # v_theta (meridional) is v, v_phi (zonal) is u.
    # This matches the convention used in NCL's Spherepack wrappers.
    vec_map = np.stack((v, u), axis=0).astype(np.float64)
    alm_vec = ducc0.sht.analysis_2d(
        map=vec_map,
        spin=1,
        lmax=lmax,
        mmax=mmax,
        geometry=geometry,
        nthreads=nthreads,
    )
    alm_E = alm_vec[0]
    alm_B = alm_vec[1]

    # Spectral Scaling:
    # We apply the eigenvalue of the gradient/curl operators in spectral space.
    # l_arr contains the degree 'l' for each coefficient in the alm array.
    l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])
    eigen_scale = np.sqrt(l_arr * (l_arr + 1.0)) / R
    alm_div = -eigen_scale * alm_E
    alm_vort = eigen_scale * alm_B

    # Synthesis
    div = ducc0.sht.synthesis_2d(
        alm=np.expand_dims(alm_div, axis=0),
        spin=0,
        lmax=lmax,
        mmax=mmax,
        ntheta=ntheta,
        nphi=nphi,
        geometry=geometry,
        nthreads=nthreads,
    )[0]
    vort = ducc0.sht.synthesis_2d(
        alm=np.expand_dims(alm_vort, axis=0),
        spin=0,
        lmax=lmax,
        mmax=mmax,
        ntheta=ntheta,
        nphi=nphi,
        geometry=geometry,
        nthreads=nthreads,
    )[0]

    return cast(NDArray[np.float64], div), cast(NDArray[np.float64], vort)


def apply_vort_div(
    u: xr.DataArray,
    v: xr.DataArray,
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    backend: Literal["serial", "mpi", "dask", "jax", "jax-mpi", "jax-dask"] = "serial",
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
        backend: Parallelization backend. Options: 'serial', 'mpi', 'dask', 'jax', 'jax-mpi', 'jax-dask'.

    Returns:
        div, vort: Divergence and relative vorticity DataArrays.
    """
    # Logic for handling parallel dimensions if needed (ufunc)
    kwargs: KinematicsKwargs = {
        "R": R,
        "lmax": lmax,
        "geometry": geometry,
        "nthreads": nthreads if backend not in ("mpi", "dask", "jax-mpi", "jax-dask") else 1,
    }

    # Identify spatial dimensions
    lat_dim = u.dims[-2]
    lon_dim = u.dims[-1]

    # Select core function based on backend
    core_func = compute_vort_div_jax if backend.startswith("jax") else compute_vort_div

    # Use apply_ufunc for broad support
    div_vort = xr.apply_ufunc(
        core_func,
        u,
        v,
        input_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        vectorize=True,
        kwargs=kwargs,
        dask="parallelized" if backend in ("dask", "jax-dask") else "forbidden",
        output_dtypes=[u.dtype, u.dtype],
    )

    div = div_vort[0].copy()
    vort = div_vort[1].copy()

    div.name = "divergence"
    vort.name = "relative_vorticity"

    return div, vort


class Kinematics:
    """
    Computes spatial derivatives and kinematic properties of the wind field.
    """

    def __init__(
        self,
        R: float = R_EARTH_METERS,
        lmax: int | None = None,
        geometry: str = "CC",
    ) -> None:
        """
        Initialize the kinematics calculator.

        Args:
            R: Planetary radius in meters.
            lmax: Maximum spherical harmonic degree.
            geometry: Grid geometry ('CC', 'DH', etc.).
        """
        self.R = R
        self.lmax = lmax
        self.geometry = geometry

    @overload
    def compute(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        backend: Literal["serial", "mpi", "dask", "jax", "jax-mpi", "jax-dask"] = "serial",
        nthreads: int = 0,
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    @overload
    def compute(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask", "jax", "jax-mpi", "jax-dask"] = "serial",
        nthreads: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def compute(
        self,
        u: xr.DataArray | NDArray[np.float64],
        v: xr.DataArray | NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask", "jax", "jax-mpi", "jax-dask"] = "serial",
        nthreads: int = 0,
    ) -> tuple[xr.DataArray | NDArray[np.float64], xr.DataArray | NDArray[np.float64]]:
        """
        Computes vorticity and divergence from wind components.

        Args:
            u: Zonal wind component.
            v: Meridional wind component.
            backend: Parallelization backend ('serial', 'mpi', 'dask', 'jax', 'jax-mpi', 'jax-dask').
            nthreads: Number of threads (for local computation).

        Returns:
            div, vort: Divergence and relative vorticity.
        """
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            core_func = (
                compute_vort_div_jax if backend.startswith("jax") else compute_vort_div
            )
            return core_func(
                u,
                v,
                R=self.R,
                lmax=self.lmax,
                geometry=self.geometry,
                nthreads=nthreads,
            )

        if isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray):
            return apply_vort_div(
                u,
                v,
                R=self.R,
                lmax=self.lmax,
                geometry=self.geometry,
                nthreads=nthreads,
                backend=backend,
            )

        raise TypeError("u and v must be both numpy arrays or both xarray DataArrays")
