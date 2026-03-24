from __future__ import annotations

import threading
from typing import Literal, TypedDict

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..models.constants import R_EARTH_METERS


class DerivativeKwargs(TypedDict, total=False):
    R: float
    lmax: int | None
    geometry: str
    nthreads: int
    engine: str


try:
    import ducc0  # type: ignore[import-not-found]

    DUCC0_AVAILABLE = True
except ImportError:
    DUCC0_AVAILABLE = False

try:
    import shtns  # type: ignore[import-untyped]

    SHTNS_AVAILABLE = True
except ImportError:
    SHTNS_AVAILABLE = False

_thread_local = threading.local()


def _get_shtns_plan(nlat: int, nlon: int, lmax: int) -> shtns.sht:
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}
    key = (nlat, nlon, lmax)
    if key not in _thread_local.cache:
        grid_lmax = (nlat - 1) // 2
        actual_lmax = min(lmax, grid_lmax)
        mmax = min(actual_lmax, nlon // 2 - 1)
        sh = shtns.sht(actual_lmax, mmax, norm=shtns.sht_fourpi)
        sh.set_grid(nlat, nlon, shtns.sht_reg_poles | shtns.SHT_PHI_CONTIGUOUS)
        _thread_local.cache[key] = sh
    return _thread_local.cache[key]


def compute_relative_vorticity_divergence(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    engine: Literal["auto", "shtns", "ducc0", "shtools"] = "auto",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes spatial divergence and relative vorticity from u and v wind components.

    Args:
        u: Zonal wind (ntheta, nphi).
        v: Meridional wind (ntheta, nphi).
        R: Planetary radius in meters. Default is R_EARTH_METERS.
        lmax: Maximum spherical harmonic degree. If None, derived from ntheta.
        geometry: Grid geometry (for ducc0/shtools). Default 'CC'.
        nthreads: Number of threads (for ducc0).
        engine: Transform engine ('auto', 'shtns', 'ducc0', 'shtools').

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

    # Resolve engine
    resolved_engine = engine
    if resolved_engine == "auto":
        resolved_engine = "shtns" if SHTNS_AVAILABLE else "ducc0"

    if resolved_engine == "shtns":
        if not SHTNS_AVAILABLE:
            raise ImportError("shtns is requested but not available.")
        sh = _get_shtns_plan(ntheta, nphi, lmax)
        # SHTns: spat_to_SHsphtor expects (v_theta, v_phi)
        # v_theta = v, v_phi = u (pointing North and East)
        v_theta = np.ascontiguousarray(v, dtype=np.float64)
        v_phi = np.ascontiguousarray(u, dtype=np.float64)
        # Returns S (divergence-like) and T (vorticity-like) coeffs
        slm = np.zeros(sh.nlm, dtype=np.complex128)
        tlm = np.zeros(sh.nlm, dtype=np.complex128)
        sh.spat_to_SHsphtor(v_theta, v_phi, slm, tlm)

        # Scale to get div and vort coefficients
        # SHTns returns coefficients s, t such that:
        # V = sum (s Ylm_grad + t Ylm_curl)
        # In meteorology:
        # div = -sqrt(l(l+1))/R * s
        # vort = sqrt(l(l+1))/R * t
        l_arr = sh.l
        eigen = np.sqrt(l_arr * (l_arr + 1.0)) / R
        div_lm = -slm * eigen
        vort_lm = tlm * eigen

        div = sh.synth(div_lm)
        vort = sh.synth(vort_lm)
        return div, vort

    elif resolved_engine in ("ducc0", "shtools"):
        import pyshtools as pysh  # type: ignore[import-untyped]

        backend = "ducc" if resolved_engine == "ducc0" else "shtools"
        mmax = min(lmax, (nphi - 1) // 2)

        # Standard spectral derivation from wind components:
        # V = (u, v)
        # div = 1/(R cos lat) * (du/dlon + d(v cos lat)/dlat)
        # vo  = 1/(R cos lat) * (dv/dlon - d(u cos lat)/dlat)
        # In spectral space, for a vector field expanded into E/B modes:
        # div_lm = -sqrt(l(l+1))/R * E_lm
        # vort_lm = sqrt(l(l+1))/R * B_lm

        # Analysis
        if backend == "ducc" and DUCC0_AVAILABLE:
            # v_theta = v, v_phi = u (matches NCL Spherepack convention)
            vec_map = np.stack((v, u), axis=0).astype(np.float64)
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
        else:
            # Manual E/B (Spheroidal/Toroidal) decomposition using Fortran SHTOOLS
            # SHExpandVDH expects (v_theta, v_phi). v_theta is southward wind.
            try:
                cilm_s, cilm_t = pysh.backends.shtools.SHExpandVDH(-v, u, lmax=lmax)

                # Spectral Scaling
                # div = -l(l+1)/R * s, vort = -l(l+1)/R * t
                div_lm_raw = np.zeros_like(cilm_s)
                vort_lm_raw = np.zeros_like(cilm_t)
                for l_val in range(lmax + 1):
                    scale = -float(l_val * (l_val + 1)) / R
                    div_lm_raw[:, l_val, : l_val + 1] = (
                        cilm_s[:, l_val, : l_val + 1] * scale
                    )
                    vort_lm_raw[:, l_val, : l_val + 1] = (
                        cilm_t[:, l_val, : l_val + 1] * scale
                    )

                # Synthesis
                div = pysh.backends.shtools.MakeGridDH(div_lm_raw, sampling=2)
                vort = pysh.backends.shtools.MakeGridDH(vort_lm_raw, sampling=2)

                # Handle potential shape mismatch (DH vs CC)
                if div.shape != (ntheta, nphi):
                    # Fallback to ducc0 if shapes are weird
                    return compute_relative_vorticity_divergence(
                        u, v, R, lmax, geometry, nthreads, engine="ducc0"
                    )

                return div, vort
            except Exception:
                # Fallback to ducc0 if Fortran path fails
                return compute_relative_vorticity_divergence(
                    u, v, R, lmax, geometry, nthreads, engine="ducc0"
                )

        # Spectral Scaling
        l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])
        eigen_scale = np.sqrt(l_arr * (l_arr + 1.0)) / R
        alm_div = -eigen_scale * alm_E
        alm_vort = eigen_scale * alm_B

        # Synthesis
        if backend == "ducc" and DUCC0_AVAILABLE:
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
        else:
            # This part shouldn't really be reached if we have ducc0
            raise NotImplementedError("Pure shtools vector synthesis not implemented.")

        return div, vort

    raise ValueError(f"Unknown engine: {engine}")


def apply_wind_derivatives(
    u: xr.DataArray,
    v: xr.DataArray,
    R: float = R_EARTH_METERS,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    engine: Literal["auto", "shtns", "ducc0", "shtools"] = "auto",
    backend: Literal["serial", "mpi", "dask"] = "serial",
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
        engine: Transform engine.
        backend: Parallelization backend.

    Returns:
        div, vort: Divergence and relative vorticity DataArrays.
    """
    # Logic for handling parallel dimensions if needed (ufunc)
    # For now, similar to apply_sh_filter
    kwargs: DerivativeKwargs = {
        "R": R,
        "lmax": lmax,
        "geometry": geometry,
        "nthreads": nthreads if backend not in ("mpi", "dask") else 1,
        "engine": engine,
    }

    # Identify spatial dimensions
    lat_dim = u.dims[-2]
    lon_dim = u.dims[-1]

    # Use apply_ufunc for broad support
    div_vort = xr.apply_ufunc(
        compute_relative_vorticity_divergence,
        u,
        v,
        input_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        vectorize=True,
        kwargs=kwargs,
        dask="parallelized" if backend == "dask" else "forbidden",
        output_dtypes=[u.dtype, u.dtype],
    )

    div = div_vort[0].copy()
    vort = div_vort[1].copy()

    div.name = "divergence"
    vort.name = "relative_vorticity"

    return div, vort
