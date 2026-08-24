"""Spherical vector-harmonic kinematic diagnostics.

The divergence/vorticity relations are standard spherical vector-harmonic
mathematics.  ``ducc0`` supplies the numerical spin-weighted SHT machinery;
relevant transform lineage includes Reinecke and Seljebotn (2013),
https://doi.org/10.1051/0004-6361/201321494, and Ishioka (2018),
https://doi.org/10.2151/jmsj.2018-019.  The surrounding xarray and backend
integration is PyStormTracker engineering.
"""

from __future__ import annotations

from typing import Literal, TypedDict, cast, overload

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..backends import Backend
from ..models.geo import R_EARTH_M


class KinematicsKwargs(TypedDict, total=False):
    R: float
    lmax: int | None
    geometry: str
    nthreads: int
    lat_reverse: bool


def _compute_vorticity_divergence_frame(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    lat_reverse: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spatial divergence and relative vorticity from 2D wind components.

    The spherical vector-harmonic relations are standard; ``ducc0`` provides
    the numerical spin-weighted transform implementation.

    Args:
        u: Zonal wind (ntheta, nphi).
        v: Meridional wind (ntheta, nphi).
        R: Planetary radius in meters. Default is R_EARTH_M.
        lmax: Maximum spherical harmonic degree. If None, derived from ntheta.
        geometry: Grid geometry (for ducc0). Default 'CC'.
        nthreads: Number of threads (for ducc0).
        lat_reverse: If True, assume latitude is North to South (reversed).

    Returns:
        divergence: Divergence (ntheta, nphi)
        vorticity: Relative vorticity (ntheta, nphi)
    """
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: u is {u.shape}, v is {v.shape}")

    if lat_reverse:
        u = u[::-1, :]
        v = v[::-1, :]

    ntheta, nphi = u.shape
    if lmax is None:
        if geometry == "CC":
            lmax = ntheta - 2
        elif geometry == "DH":
            lmax = (ntheta - 2) // 2
        else:
            lmax = ntheta - 1

    mmax = min(lmax, (nphi - 1) // 2)

    # parity: (v_theta, v_phi) = (-v, u)
    import ducc0

    vec_map = np.stack((-v, u), axis=0).astype(np.float64)
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
    l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])
    eigen_scale = np.sqrt(l_arr * (l_arr + 1.0)) / R
    alm_div = -eigen_scale * alm_E
    alm_vort = -eigen_scale * alm_B

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

    if not lat_reverse:
        div = div[::-1, :]
        vort = vort[::-1, :]

    return cast(NDArray[np.float64], div), cast(NDArray[np.float64], vort)


def _compute_vorticity_divergence_xarray(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    backend: Backend = "serial",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Private Xarray adapter for computing relative vorticity and divergence.

    Args:
        u: Zonal wind DataArray.
        v: Meridional wind DataArray.
        R: Planetary radius in meters. Default is R_EARTH_M.
        lmax: Maximum spherical harmonic degree.
        geometry: Grid geometry (default 'CC').
        nthreads: Number of threads.
        backend: Parallelization backend. Options: 'serial', 'mpi', 'dask'.

    Returns:
        divergence, vorticity: Divergence and relative vorticity DataArrays.
    """
    from ..io.data_loader import DataLoader

    loader = DataLoader(u.dataset if hasattr(u, "dataset") else u)
    # Identify spatial dimensions
    lat_dim = loader.find_coordinate_dimension(u, "latitude")
    lon_dim = loader.find_coordinate_dimension(u, "longitude")

    if not lat_dim or not lon_dim:
        # Fallback to positional if not found
        lat_dim = str(u.dims[-2])
        lon_dim = str(u.dims[-1])

    # Ensure latitude is North to South for ducc0
    # Store original order to restore it later if needed
    is_ascending = not loader.is_lat_reversed()
    u_sorted = u.sortby(lat_dim, ascending=False)
    v_sorted = v.sortby(lat_dim, ascending=False)

    # Logic for handling parallel dimensions if needed (ufunc)
    kwargs: KinematicsKwargs = {
        "R": R,
        "lmax": lmax,
        "geometry": geometry,
        "nthreads": nthreads if backend not in ("mpi", "dask") else 1,
        "lat_reverse": True,  # Already sorted to N-to-S (90 to -90)
    }

    core_func = _compute_vorticity_divergence_frame

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = "forbidden"
    if u_sorted.chunks or v_sorted.chunks:
        dask_mode = "parallelized"

    # Use apply_ufunc for broad support
    div_vort = xr.apply_ufunc(
        core_func,
        u_sorted,
        v_sorted,
        input_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim], [lat_dim, lon_dim]],
        vectorize=True,
        kwargs=kwargs,
        dask=dask_mode,
        output_dtypes=[u.dtype, u.dtype],
    )

    divergence = div_vort[0].copy()
    vorticity = div_vort[1].copy()

    divergence.name = "divergence"
    vorticity.name = "relative_vorticity"

    if is_ascending:
        divergence = divergence.sortby(lat_dim, ascending=True)
        vorticity = vorticity.sortby(lat_dim, ascending=True)

    return divergence, vorticity


@overload
def compute_vorticity_divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    lat_reverse: bool = False,
    backend: Backend = "serial",
) -> tuple[xr.DataArray, xr.DataArray]: ...


@overload
def compute_vorticity_divergence(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    *,
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    lat_reverse: bool = False,
    backend: Backend = "serial",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def compute_vorticity_divergence(
    u: xr.DataArray | NDArray[np.float64],
    v: xr.DataArray | NDArray[np.float64],
    *,
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: str = "CC",
    nthreads: int = 0,
    lat_reverse: bool = False,
    backend: Backend = "serial",
) -> tuple[xr.DataArray | NDArray[np.float64], xr.DataArray | NDArray[np.float64]]:
    """
    Computes spatial divergence and relative vorticity from u and v wind components.
    Accepts either a pair of xarray DataArrays or a pair of 2D NumPy arrays.

    Args:
        u: Zonal wind component (DataArray or 2D NumPy array).
        v: Meridional wind component (DataArray or 2D NumPy array).
        R: Planetary radius in meters. Default is R_EARTH_M.
        lmax: Maximum spherical harmonic degree.
        geometry: Grid geometry ('CC', 'DH', etc.). Default 'CC'.
        nthreads: Number of threads.
        lat_reverse: If True, assume latitude is North to South (NumPy only).
        backend: Parallelization backend ('serial', 'mpi', 'dask') for DataArray.

    Returns:
        divergence, vorticity: Tuple of divergence and relative vorticity.
    """
    if isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray):
        return _compute_vorticity_divergence_xarray(
            u,
            v,
            R=R,
            lmax=lmax,
            geometry=geometry,
            nthreads=nthreads,
            backend=backend,
        )

    if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
        return _compute_vorticity_divergence_frame(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            R=R,
            lmax=lmax,
            geometry=geometry,
            nthreads=nthreads,
            lat_reverse=lat_reverse,
        )

    raise TypeError("u and v must be both numpy arrays or both xarray DataArrays")


class Kinematics:
    """
    Computes spatial derivatives and kinematic properties of the wind field.
    """

    def __init__(
        self,
        R: float = R_EARTH_M,
        lmax: int | None = None,
        geometry: str = "CC",
        lat_reverse: bool = False,
    ) -> None:
        """
        Initialize the kinematics calculator.

        Args:
            R: Planetary radius in meters.
            lmax: Maximum spherical harmonic degree.
            geometry: Grid geometry ('CC', 'DH', etc.).
            lat_reverse: If True, assume latitude is North to South (reversed).
        """
        self.R = R
        self.lmax = lmax
        self.geometry = geometry
        self.lat_reverse = lat_reverse

    @overload
    def compute(
        self,
        u: xr.DataArray,
        v: xr.DataArray,
        backend: Backend = "serial",
        nthreads: int = 0,
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    @overload
    def compute(
        self,
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        backend: Backend = "serial",
        nthreads: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def compute(
        self,
        u: xr.DataArray | NDArray[np.float64],
        v: xr.DataArray | NDArray[np.float64],
        backend: Backend = "serial",
        nthreads: int = 0,
    ) -> tuple[xr.DataArray | NDArray[np.float64], xr.DataArray | NDArray[np.float64]]:
        """
        Computes vorticity and divergence from wind components.

        Args:
            u: Zonal wind component.
            v: Meridional wind component.
            backend: Parallelization backend ('serial', 'mpi', 'dask').
            nthreads: Number of threads (for local computation).

        Returns:
            divergence, vorticity: Divergence and relative vorticity.
        """
        if isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray):
            return compute_vorticity_divergence(
                u,
                v,
                R=self.R,
                lmax=self.lmax,
                geometry=self.geometry,
                nthreads=nthreads,
                lat_reverse=self.lat_reverse,
                backend=backend,
            )
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            return compute_vorticity_divergence(
                cast("NDArray[np.float64]", u),  # type: ignore[redundant-cast]
                cast("NDArray[np.float64]", v),  # type: ignore[redundant-cast]
                R=self.R,
                lmax=self.lmax,
                geometry=self.geometry,
                nthreads=nthreads,
                lat_reverse=self.lat_reverse,
                backend=backend,
            )
        raise TypeError("u and v must be both numpy arrays or both xarray DataArrays")
