# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spherical vector-harmonic kinematic diagnostics.

The divergence/vorticity relations are standard spherical vector-harmonic
mathematics. Public ``spharmgrid`` supplies default rectangular CC/GL
kinematics. Explicit ``lmax`` calculations use direct ``ducc0`` spin-weighted
SHT machinery because the released public vector composition has a different
numerical result. Relevant transform lineage includes Reinecke and
Seljebotn (2013),
https://doi.org/10.1051/0004-6361/201321494, and Ishioka (2018),
https://doi.org/10.2151/jmsj.2018-019.  The surrounding xarray and backend
integration is PyStormTracker engineering.
"""

from __future__ import annotations

from typing import Literal, cast, overload

import numpy as np
import spharmgrid as sg
import xarray as xr
from numpy.typing import NDArray

from ..backends import Backend, configure_sht_threads, resolve_sht_threads
from ..models.geo import R_EARTH_M
from .spectral import SHTGeometry


def _compute_vorticity_divergence_lmax_frame(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    R: float = R_EARTH_M,
    lmax: int = 0,
    geometry: Literal["CC", "GL"] = "CC",
    nthreads: int = 0,
    lat_reverse: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spatial divergence and relative vorticity from 2D wind components.

    The spherical vector-harmonic relations are standard. This narrow path
    preserves the public explicit-``lmax`` behavior that is not numerically
    reproduced by the released spharmgrid vector composition.

    Args:
        u: Zonal wind (ntheta, nphi).
        v: Meridional wind (ntheta, nphi).
        R: Planetary radius in meters. Default is R_EARTH_M.
        lmax: Maximum spherical harmonic degree.
        geometry: Rectangular grid geometry.
        nthreads: Number of threads for the direct transform.
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
    geometry: SHTGeometry = "auto",
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
    grid = sg.detect_grid(u)
    if geometry != "auto" and geometry.lower() != grid.kind:
        raise ValueError(
            f"geometry={geometry!r} does not match the "
            f"coordinate-defined {grid.kind.upper()} grid"
        )

    if lmax is None:
        result = sg.kinematics(
            u,
            v,
            radius=R,
            sht_threads=_spharmgrid_threads(nthreads, backend),
        )
        divergence = result["d"].copy(deep=False)
        vorticity = result["vo"].copy(deep=False)
        divergence.name = "divergence"
        vorticity.name = "relative_vorticity"
        return divergence, vorticity

    latitude_name = _coordinate_name(u, "latitude")
    longitude_name = _coordinate_name(u, "longitude")
    latitude_dim = str(u[latitude_name].dims[0])
    longitude_dim = str(u[longitude_name].dims[0])
    latitude_values = np.asarray(u[latitude_name].values, dtype=np.float64)
    lat_reverse = bool(
        latitude_values.size > 1 and latitude_values[0] > latitude_values[-1]
    )
    direct_geometry: Literal["CC", "GL"] = "CC" if grid.kind == "cc" else "GL"
    direct_threads = _direct_sht_threads(nthreads, backend)
    configure_sht_threads(direct_threads)
    kwargs = {
        "R": R,
        "lmax": lmax,
        "geometry": direct_geometry,
        "nthreads": direct_threads,
        "lat_reverse": lat_reverse,
    }

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = "forbidden"
    if u.chunks or v.chunks:
        dask_mode = "parallelized"

    div_vort = xr.apply_ufunc(
        _compute_vorticity_divergence_lmax_frame,
        u,
        v,
        input_core_dims=[[latitude_dim, longitude_dim]] * 2,
        output_core_dims=[[latitude_dim, longitude_dim]] * 2,
        vectorize=True,
        kwargs=kwargs,
        dask=dask_mode,
        output_dtypes=[np.float64, np.float64],
    )

    divergence = div_vort[0].copy(deep=False)
    vorticity = div_vort[1].copy(deep=False)

    divergence.name = "divergence"
    vorticity.name = "relative_vorticity"
    return divergence, vorticity


def _coordinate_name(
    data: xr.DataArray,
    axis: Literal["latitude", "longitude"],
) -> str:
    """Find a one-dimensional latitude or longitude coordinate."""
    aliases = (axis, "lat") if axis == "latitude" else (axis, "lon")
    for name in aliases:
        if name in data.coords and data[name].ndim == 1:
            return name
    for coordinate_name, coordinate in data.coords.items():
        if coordinate.ndim == 1 and coordinate.attrs.get("standard_name") == axis:
            return str(coordinate_name)
    raise ValueError(f"could not identify {axis} coordinate")


def _direct_sht_threads(nthreads: int, backend: Backend) -> int:
    """Resolve threads for the explicit-``lmax`` direct path."""
    requested = None if nthreads == 0 else nthreads
    return resolve_sht_threads(requested, backend)


def _spharmgrid_threads(nthreads: int, backend: Backend) -> int | None:
    """Map the legacy zero-thread default to spharmgrid's default."""
    requested = None if nthreads == 0 else nthreads
    resolved = resolve_sht_threads(requested, backend)
    return None if resolved == 0 else resolved


def _validate_geometry(geometry: SHTGeometry) -> None:
    """Validate the runtime geometry domain used by kinematics."""
    if geometry not in ("CC", "GL", "auto"):
        raise ValueError("geometry must be 'CC', 'GL', or 'auto'")


def _numpy_grid(
    geometry: SHTGeometry,
    nlat: int,
    nlon: int,
    lat_reverse: bool,
) -> sg.Grid:
    """Construct coordinates for coordinate-free NumPy input."""
    resolved_geometry: Literal["CC", "GL"] = "CC" if geometry == "auto" else geometry
    latitude_order: Literal["ascending", "descending"] = (
        "descending" if lat_reverse else "ascending"
    )
    if resolved_geometry == "CC":
        return sg.clenshaw_curtis_grid(
            nlat,
            nlon,
            latitude_order=latitude_order,
        )
    return sg.gaussian_grid(
        nlat,
        nlon,
        latitude_order=latitude_order,
    )


@overload
def compute_vorticity_divergence(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    R: float = R_EARTH_M,
    lmax: int | None = None,
    geometry: SHTGeometry = "auto",
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
    geometry: SHTGeometry = "auto",
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
    geometry: SHTGeometry = "auto",
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
        geometry: Grid geometry ('CC', 'GL', or 'auto'). The default detects
            the xarray grid and uses CC for coordinate-free NumPy input.
        nthreads: Number of threads.
        lat_reverse: If True, assume latitude is North to South (NumPy only).
        backend: Parallelization backend ('serial', 'mpi', 'dask') for DataArray.

    Returns:
        divergence, vorticity: Tuple of divergence and relative vorticity.
    """
    _validate_geometry(geometry)
    if lmax is not None and (isinstance(lmax, bool) or lmax < 0):
        raise ValueError("lmax must be a nonnegative integer or None")

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
        if u.shape != v.shape:
            raise ValueError(f"Shape mismatch: u is {u.shape}, v is {v.shape}")
        if u.ndim not in (2, 3):
            raise ValueError("NumPy wind components must be 2D or 3D")
        grid = _numpy_grid(geometry, u.shape[-2], u.shape[-1], lat_reverse)
        dimensions = (
            ("latitude", "longitude")
            if u.ndim == 2
            else ("time", "latitude", "longitude")
        )
        u_xarray = xr.DataArray(
            np.asarray(u, dtype=np.float64),
            dims=dimensions,
            coords={"latitude": grid.latitude, "longitude": grid.longitude},
        )
        v_xarray = xr.DataArray(
            np.asarray(v, dtype=np.float64),
            dims=dimensions,
            coords={"latitude": grid.latitude, "longitude": grid.longitude},
        )
        divergence, vorticity = _compute_vorticity_divergence_xarray(
            u_xarray,
            v_xarray,
            R=R,
            lmax=lmax,
            geometry=geometry,
            nthreads=nthreads,
            backend=backend,
        )
        return (
            np.asarray(divergence.values),
            np.asarray(vorticity.values),
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
        geometry: SHTGeometry = "auto",
        lat_reverse: bool = False,
    ) -> None:
        """
        Initialize the kinematics calculator.

        Args:
            R: Planetary radius in meters.
            lmax: Maximum spherical harmonic degree.
            geometry: Grid geometry ('CC', 'GL', or 'auto').
            lat_reverse: If True, assume NumPy input latitude is North to South
                (reversed). Xarray input uses its latitude coordinate order.
        """
        _validate_geometry(geometry)
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
                u,
                v,
                R=self.R,
                lmax=self.lmax,
                geometry=self.geometry,
                nthreads=nthreads,
                lat_reverse=self.lat_reverse,
                backend=backend,
            )
        raise TypeError("u and v must be both numpy arrays or both xarray DataArrays")
