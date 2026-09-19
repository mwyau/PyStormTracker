# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spectral filtering for global spherical and regional grids.

The global spherical-harmonic taper follows the published spherical
methodology of Sardeshmukh and Hoskins (1984), “Spatial Smoothing on the
Sphere,” *Monthly Weather Review*, 112(12), 2524--2529:
https://doi.org/10.1175/1520-0493(1984)112<2524:SSOTS>2.0.CO;2

Supported rectangular GL/CC operations use public ``spharmgrid``; NumPy input
uses a coordinate-aware adapter around the same operations. Reduced-grid SHT
and regional DCT machinery uses ``ducc0``. Relevant numerical lineage includes
Reinecke and Seljebotn
(2013), *Libsharp -- spherical harmonic transforms revisited*,
https://doi.org/10.1051/0004-6361/201321494, and Ishioka (2018), “A New
Recurrence Formula for Efficient Computation of Spherical Harmonic
Transform,” https://doi.org/10.2151/jmsj.2018-019.

The regional effective-total-wavenumber filter is a PyStormTracker regional
adaptation using standard DCT machinery.  It is not presented as an exact
implementation of the global spherical derivation.
"""

from __future__ import annotations

import warnings
from typing import Literal, cast, overload

import numpy as np
import spharmgrid as sg
import xarray as xr
from numpy.typing import NDArray

from ..backends import Backend, configure_sht_threads, resolve_sht_threads

type SHTGeometry = Literal["CC", "GL", "auto"]


def _apply_reduced_bandpass_mask(
    alm: NDArray[np.complex128],
    lmin: int,
    lmax: int,
    mmax: int | None = None,
    taper_val: float = 0.1,
) -> None:
    """
    Apply the PST tapered bandpass mask to reduced-grid coefficients.

    Args:
        alm: Spherical harmonic coefficients.
        lmin: Minimum wave number (hard cutoff).
        lmax: Maximum wave number (tapered to taper_val).
        mmax: Maximum m wave number.
        taper_val: Value of the taper at lmax (default 0.1).
    """
    if mmax is None:
        mmax = lmax

    # Total number of coefficients
    l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])

    # Initialize mask as zeros
    weights = np.zeros_like(l_arr, dtype=np.float64)

    # 1. Identity within the band [lmin, lmax], but with exponential smoothing
    # w(l) = exp(-K * (l*(l+1))^2)
    # such that w(lmax) = taper_val. This acts like a ∇⁴ hyper-diffusion operator,
    # significantly reducing Gibbs phenomenon ringing at the truncation boundary.
    if lmax > 0:
        k_val = -np.log(taper_val) / (lmax * (lmax + 1)) ** 2
        mask_band = (l_arr >= lmin) & (l_arr <= lmax)
        weights[mask_band] = np.exp(
            -k_val * (l_arr[mask_band] * (l_arr[mask_band] + 1)) ** 2
        )

    if alm.ndim == 2:
        alm[0, :] *= weights
    else:
        alm[:] *= weights


def _filter_reduced_gaussian_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    theta: NDArray[np.float64],
    nphi: NDArray[np.uint64],
    phi0: NDArray[np.float64],
    ringstart: NDArray[np.uint64],
    nthreads: int = 1,
    taper_val: float = 0.1,
    out_geometry: Literal["CC", "GL"] | None = None,
    out_ntheta: int | None = None,
    out_nphi: int | None = None,
) -> NDArray[np.float64]:
    """Filter one reduced-Gaussian frame with DUCC's pseudo-analysis."""
    if frame.ndim != 1:
        raise ValueError("reduced-Gaussian frames must be one-dimensional")

    nlat = len(theta)
    nlon = int(np.max(nphi))
    if nlat < lmax + 1:
        raise ValueError(
            f"Unsupported shape for spectral filter: {frame.shape} cannot "
            f"represent lmax={lmax}."
        )

    mmax = min(lmax, nlon // 2 - 1)
    import ducc0

    try:
        alm, _, _, _, _ = ducc0.sht.pseudo_analysis(
            map=np.expand_dims(frame, axis=0),
            spin=0,
            lmax=lmax,
            mmax=mmax,
            theta=theta,
            nphi=nphi,
            phi0=phi0,
            ringstart=ringstart,
            nthreads=nthreads,
            maxiter=100,
            epsilon=1e-6,
        )

        _apply_reduced_bandpass_mask(alm, lmin, lmax, mmax, taper_val=taper_val)

        if out_geometry is None:
            return cast(
                NDArray[np.float64],
                ducc0.sht.synthesis(
                    alm=alm,
                    spin=0,
                    lmax=lmax,
                    mmax=mmax,
                    theta=theta,
                    nphi=nphi,
                    phi0=phi0,
                    ringstart=ringstart,
                    geometry="GL",
                    nthreads=nthreads,
                )[0],
            )

        if out_ntheta is None or out_nphi is None:
            raise ValueError(
                "out_ntheta and out_nphi are required when out_geometry is supplied"
            )
        return cast(
            NDArray[np.float64],
            ducc0.sht.synthesis_2d(
                alm=alm,
                spin=0,
                lmax=lmax,
                mmax=mmax,
                ntheta=out_ntheta,
                nphi=out_nphi,
                geometry=out_geometry,
                nthreads=nthreads,
            )[0],
        )
    except Exception as e:
        msg = f"Spectral filter failed for shape {frame.shape}: {e}"
        raise ValueError(msg) from e


def _filter_dct_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    taper_val: float,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Filter a regional frame using a 2D DCT adaptation.

    The effective-total-wavenumber construction is a PyStormTracker regional
    adaptation using standard DCT machinery from ``ducc0``; it is not the
    global spherical-harmonic algorithm of Sardeshmukh and Hoskins (1984).
    """
    ny, nx = frame.shape
    import ducc0

    # 1. Forward 2D DCT (Type 2)
    # We use inorm=2 (divide by N) to match standard definitions
    coeffs = ducc0.fft.dct(frame, axes=(0, 1), type=2, inorm=2)

    # 2. Compute radial wavenumbers
    # Physical dimensions in degrees. We assume a regular lat-lon grid.
    # in the wavenumber mapping.
    dlat = abs(lat[1] - lat[0]) if ny > 1 else 1.0
    dlon = abs(lon[1] - lon[0]) if nx > 1 else 1.0
    width_lat = ny * dlat
    width_lon = nx * dlon

    # Wavenumbers in "total wavenumber l" units.
    # Mirroring the spherical harmonic total wavenumber l ~ sqrt(kx^2 + ky^2).
    # k_x = n_x * (180 / width_lon), k_y = n_y * (180 / width_lat)
    ky = np.arange(ny)[:, None] * (180.0 / width_lat)
    kx = np.arange(nx)[None, :] * (180.0 / width_lon)
    l_eff = np.sqrt(kx**2 + ky**2)

    # 3. Apply Tapered Mask.
    # We use the l_eff(l_eff+1) form to match the SHT Laplacian smoother behavior.
    weights = np.zeros_like(l_eff)
    if lmax > 0:
        k_val = -np.log(taper_val) / (lmax * (lmax + 1)) ** 2
        mask_band = (l_eff >= lmin) & (l_eff <= lmax)
        weights[mask_band] = np.exp(
            -k_val * (l_eff[mask_band] * (l_eff[mask_band] + 1)) ** 2
        )

    coeffs *= weights

    # 4. Inverse 2D DCT (Type 3)
    # Type 3 is the inverse of Type 2. inorm=0 because forward was inorm=2.
    return cast(
        NDArray[np.float64], ducc0.fft.dct(coeffs, axes=(0, 1), type=3, inorm=0)
    )


def _filter_dct_xarray(
    data: xr.DataArray,
    lmin: int,
    lmax: int,
    backend: Literal["serial", "mpi", "dask"] = "serial",
    taper_val: float = 0.1,
) -> xr.DataArray:
    """
    Private Xarray adapter for DCT-based spectral bandpass filter on regional
    DataArrays.

    Args:
        data: Input DataArray (regional).
        lmin: Minimum total wave number equivalent to retain.
        lmax: Maximum total wave number equivalent to retain.
        backend: Parallelization backend.
        taper_val: Value of the taper at lmax.

    Returns:
        xr.DataArray: The filtered regional data.
    """
    from ..io.data_loader import DataLoader

    loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
    # Identify spatial dimensions
    lat_dim = loader.find_coordinate_dimension(data, "latitude")
    lon_dim = loader.find_coordinate_dimension(data, "longitude")

    if not lat_dim or not lon_dim:
        raise ValueError("Input DataArray must have latitude and longitude dimensions.")

    # Prepare for xarray
    lat = data[lat_dim].values
    lon = data[lon_dim].values

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = (
        "parallelized" if data.chunks and backend == "dask" else "allowed"
    )

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            _filter_dct_frame,
            data,
            input_core_dims=[[lat_dim, lon_dim]],
            output_core_dims=[[lat_dim, lon_dim]],
            vectorize=True,
            kwargs={
                "lmin": lmin,
                "lmax": lmax,
                "taper_val": taper_val,
                "lat": lat,
                "lon": lon,
            },
            dask=dask_mode,
            output_dtypes=[data.dtype],
        ),
    )

    filtered.attrs.update(data.attrs)
    filtered.name = data.name
    return filtered


class DCTFilter:
    """Regional DCT bandpass filter.

    This is the PyStormTracker regional adaptation of a spectral cutoff.  It
    uses standard DCT machinery supplied by ``ducc0`` and should not be read
    as an exact implementation of the global spherical derivation of
    Sardeshmukh and Hoskins (1984).
    """

    def __init__(
        self,
        lmin: int,
        lmax: int,
        taper_val: float = 0.1,
    ) -> None:
        self.lmin = lmin
        self.lmax = lmax
        self.taper_val = taper_val

    def filter(
        self,
        data: xr.DataArray,
        backend: Literal["serial", "mpi", "dask"] = "serial",
    ) -> xr.DataArray:
        if not isinstance(data, xr.DataArray):
            raise TypeError("DCTFilter.filter requires xarray.DataArray")

        return _filter_dct_xarray(
            data,
            self.lmin,
            self.lmax,
            backend=backend,
            taper_val=self.taper_val,
        )


class SHTFilter:
    """Global spherical-harmonic bandpass filter.

    The published spherical smoothing lineage is Sardeshmukh and Hoskins
    (1984).  Rectangular Gauss--Legendre and Clenshaw--Curtis inputs use the
    public :func:`spharmgrid.filter` or :func:`spharmgrid.regrid` operation.
    Reduced-Gaussian inputs use the specialized DUCC pseudo-analysis path.
    """

    def __init__(
        self,
        lmin: int,
        lmax: int,
        lat_reverse: bool = False,
        taper_val: float = 0.1,
        geometry: SHTGeometry = "auto",
        out_geometry: Literal["CC", "GL"] | None = None,
        out_ntheta: int | None = None,
        out_nphi: int | None = None,
        sht_threads: int | None = None,
    ) -> None:
        """
        Initialize the filter with wave number bounds.

        Args:
            lmin (int): Minimum total wave number to retain.
            lmax (int): Maximum total wave number to retain.
            lat_reverse (bool): For NumPy input, select North-to-South
                coordinates when true. Xarray input uses its latitude
                coordinate order.
            taper_val (float): Value of the taper at lmax.
            geometry (str): Grid geometry ('CC', 'GL', or 'auto').
            out_geometry (str | None): Target geometry for regridding.
            out_ntheta (int | None): Number of latitudes in output grid.
            out_nphi (int | None): Number of longitudes in output grid.
            sht_threads: Threads per transform. Rectangular operations pass
                this value to spharmgrid; reduced-Gaussian operations pass it
                to DUCC.
        """
        if geometry not in ("CC", "GL", "auto"):
            raise ValueError("geometry must be 'CC', 'GL', or 'auto'")
        if out_geometry not in (None, "CC", "GL"):
            raise ValueError("out_geometry must be 'CC', 'GL', or None")
        has_output_size = out_ntheta is not None or out_nphi is not None
        if out_geometry is None and has_output_size:
            raise ValueError("out_geometry is required with output grid sizes")
        if out_geometry is not None and (out_ntheta is None or out_nphi is None):
            raise ValueError(
                "out_ntheta and out_nphi are required when out_geometry is supplied"
            )
        if out_ntheta is not None and out_ntheta <= 0:
            raise ValueError("out_ntheta must be positive")
        if out_nphi is not None and out_nphi <= 0:
            raise ValueError("out_nphi must be positive")
        self.lmin = lmin
        self.lmax = lmax
        self.lat_reverse = lat_reverse
        self.taper_val = taper_val
        self.geometry = geometry
        self.out_geometry = out_geometry
        self.out_ntheta = out_ntheta
        self.out_nphi = out_nphi
        if sht_threads is not None:
            resolve_sht_threads(sht_threads, "serial")
        self.sht_threads = sht_threads

    @overload
    def filter(
        self,
        data: xr.DataArray,
        backend: Literal["serial", "mpi", "dask"] = "serial",
    ) -> xr.DataArray: ...

    @overload
    def filter(
        self,
        data: NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask"] = "serial",
    ) -> NDArray[np.float64]: ...

    def filter(
        self,
        data: xr.DataArray | NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask"] = "serial",
    ) -> xr.DataArray | NDArray[np.float64]:
        """
        Applies the filter to the input data.

        Args:
            data (xr.DataArray | np.ndarray): Input data.
            backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.

        Returns:
            xr.DataArray | np.ndarray: The filtered data.
        """
        if isinstance(data, np.ndarray):
            return _filter_numpy_rectangular(
                data,
                geometry="CC" if self.geometry == "auto" else self.geometry,
                lat_reverse=self.lat_reverse,
                lmin=self.lmin,
                lmax=self.lmax,
                taper_val=self.taper_val,
                out_geometry=self.out_geometry,
                out_ntheta=self.out_ntheta,
                out_nphi=self.out_nphi,
                sht_threads=self.sht_threads,
                backend=backend,
            )

        if not isinstance(data, xr.DataArray):
            raise TypeError("SHTFilter.filter requires a NumPy array or DataArray")

        if _is_reduced_gaussian(data):
            return _filter_reduced_gaussian_xarray(
                data,
                lmin=self.lmin,
                lmax=self.lmax,
                taper_val=self.taper_val,
                out_geometry=self.out_geometry,
                out_ntheta=self.out_ntheta,
                out_nphi=self.out_nphi,
                sht_threads=self.sht_threads,
                backend=backend,
            )

        detected_grid = sg.detect_grid(data)
        if self.geometry != "auto" and self.geometry.lower() != detected_grid.kind:
            raise ValueError(
                f"geometry={self.geometry!r} does not match the "
                f"coordinate-defined {detected_grid.kind.upper()} grid"
            )
        return _filter_rectangular_xarray(
            data,
            grid=detected_grid,
            lmin=self.lmin,
            lmax=self.lmax,
            taper_val=self.taper_val,
            out_geometry=self.out_geometry,
            out_ntheta=self.out_ntheta,
            out_nphi=self.out_nphi,
            sht_threads=self.sht_threads,
            backend=backend,
        )


def _is_reduced_gaussian(data: xr.DataArray) -> bool:
    """Return whether ``data`` uses the reduced-Gaussian representation."""
    from ..io.data_loader import DataLoader

    variable_name = str(data.name) if data.name is not None else ""
    loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
    return loader.is_reduced_gaussian(variable_name)


def _spharmgrid_sht_threads(
    sht_threads: int | None,
    backend: Backend,
) -> int | None:
    """Map PST's serial zero-thread default to spharmgrid's ``None`` default."""
    resolved = resolve_sht_threads(sht_threads, backend)
    return None if resolved == 0 else resolved


def _rectangular_target_grid(
    out_geometry: Literal["CC", "GL"],
    ntheta: int,
    nphi: int,
    *,
    latitude_order: Literal["ascending", "descending"],
) -> sg.Grid:
    """Construct a public spharmgrid target descriptor for SHTFilter output."""
    if out_geometry == "CC":
        return sg.clenshaw_curtis_grid(
            ntheta,
            nphi,
            latitude_order=latitude_order,
        )
    if out_geometry == "GL":
        return sg.gaussian_grid(
            ntheta,
            nphi,
            latitude_order=latitude_order,
        )
    raise ValueError("spharmgrid output geometry must be 'CC' or 'GL'")


def _filter_numpy_rectangular(
    data: NDArray[np.float64],
    *,
    geometry: Literal["CC", "GL"],
    lat_reverse: bool,
    lmin: int,
    lmax: int,
    taper_val: float,
    out_geometry: Literal["CC", "GL"] | None,
    out_ntheta: int | None,
    out_nphi: int | None,
    sht_threads: int | None,
    backend: Backend,
) -> NDArray[np.float64]:
    """Adapt coordinate-free NumPy input to the rectangular xarray path."""
    if data.ndim not in (2, 3):
        raise ValueError("numpy array must be 2D or 3D")
    nlat, nlon = data.shape[-2:]
    latitude_order: Literal["ascending", "descending"] = (
        "descending" if lat_reverse else "ascending"
    )
    grid = _rectangular_target_grid(geometry, nlat, nlon, latitude_order=latitude_order)
    dimensions = (
        ("latitude", "longitude")
        if data.ndim == 2
        else ("time", "latitude", "longitude")
    )
    wrapped = xr.DataArray(
        data,
        dims=dimensions,
        coords={
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        },
    )
    result = _filter_rectangular_xarray(
        wrapped,
        grid=grid,
        lmin=lmin,
        lmax=lmax,
        taper_val=taper_val,
        out_geometry=out_geometry,
        out_ntheta=out_ntheta,
        out_nphi=out_nphi,
        sht_threads=sht_threads,
        backend=backend,
        # Standalone NumPy APIs must not inherit xarray/tracker frame slicing.
        partition_mpi=False,
    )
    return cast(NDArray[np.float64], np.asarray(result.values))


def _filter_rectangular_xarray(
    data: xr.DataArray,
    *,
    grid: sg.Grid,
    lmin: int,
    lmax: int,
    taper_val: float,
    out_geometry: Literal["CC", "GL"] | None,
    out_ntheta: int | None,
    out_nphi: int | None,
    sht_threads: int | None,
    backend: Backend,
    partition_mpi: bool = True,
) -> xr.DataArray:
    """Adapt the SHTFilter facade to spharmgrid's rectangular operations."""
    latitude_name = _grid_coordinate_name(data, "latitude")
    longitude_name = _grid_coordinate_name(data, "longitude")
    latitude_dim = str(data[latitude_name].dims[0])
    longitude_dim = str(data[longitude_name].dims[0])
    if partition_mpi:
        data = _partition_mpi_xarray(data, latitude_dim, longitude_dim, backend)
    sg_threads = _spharmgrid_sht_threads(sht_threads, backend)
    if out_geometry is None:
        return sg.filter(
            data,
            lmin=lmin,
            lmax=lmax,
            taper=taper_val,
            sht_threads=sg_threads,
        )

    if out_ntheta is None or out_nphi is None:
        raise ValueError(
            "out_ntheta and out_nphi are required when out_geometry is supplied"
        )
    source_latitude_order: Literal["ascending", "descending"] = (
        "ascending" if grid.latitude[0] < grid.latitude[-1] else "descending"
    )
    # Preserve the established output orientation: CC targets are ascending;
    # GL targets follow the source representation. spharmgrid keeps values
    # associated with the coordinates in either orientation.
    latitude_order: Literal["ascending", "descending"] = (
        "ascending" if out_geometry == "CC" else source_latitude_order
    )
    target = _rectangular_target_grid(
        out_geometry,
        out_ntheta,
        out_nphi,
        latitude_order=latitude_order,
    )
    result = sg.regrid(
        data,
        target,
        lmin=lmin,
        lmax=lmax,
        taper=taper_val,
        sht_threads=sg_threads,
    )

    # The historical SHTFilter regridding path exposes canonical latitude and
    # longitude dimension names.  Keep that facade while spharmgrid owns the
    # target grid and transform.
    rename_dims: dict[str, str] = {}
    if latitude_dim != "latitude":
        rename_dims[latitude_dim] = "latitude"
    if longitude_dim != "longitude":
        rename_dims[longitude_dim] = "longitude"
    if rename_dims:
        result = result.rename(rename_dims)
    result.attrs.update(data.attrs)
    result.name = data.name
    return result


def _filter_reduced_gaussian_xarray(
    data: xr.DataArray,
    *,
    lmin: int,
    lmax: int,
    taper_val: float,
    out_geometry: Literal["CC", "GL"] | None,
    out_ntheta: int | None,
    out_nphi: int | None,
    sht_threads: int | None,
    backend: Backend,
) -> xr.DataArray:
    """Filter/regrid a reduced-Gaussian DataArray with direct DUCC."""
    from ..io.data_loader import DataLoader

    variable_name = str(data.name) if data.name is not None else ""
    loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
    spatial_dim = "values" if "values" in data.dims else str(data.dims[-1])
    metadata = loader.get_grid_metadata(variable_name)
    data = _partition_mpi_xarray(data, spatial_dim, spatial_dim, backend)

    nthreads = resolve_sht_threads(sht_threads, backend)
    configure_sht_threads(nthreads)
    if out_geometry is None:
        output_core_dims = [[spatial_dim]]
        output_sizes = None
    else:
        if out_ntheta is None or out_nphi is None:
            raise ValueError(
                "out_ntheta and out_nphi are required when out_geometry is supplied"
            )
        output_core_dims = [["latitude", "longitude"]]
        output_sizes = {"latitude": out_ntheta, "longitude": out_nphi}

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = "forbidden"
    if data.chunks:
        dask_mode = "parallelized" if backend == "dask" else "allowed"

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            _filter_reduced_gaussian_frame,
            data,
            input_core_dims=[[spatial_dim]],
            output_core_dims=output_core_dims,
            vectorize=True,
            kwargs={
                "lmin": lmin,
                "lmax": lmax,
                "theta": metadata["theta"],
                "nphi": metadata["nphi"],
                "phi0": metadata["phi0"],
                "ringstart": metadata["ringstart"],
                "nthreads": nthreads,
                "taper_val": taper_val,
                "out_geometry": out_geometry,
                "out_ntheta": out_ntheta,
                "out_nphi": out_nphi,
            },
            dask=dask_mode,
            output_dtypes=[np.float64],
            dask_gufunc_kwargs=(
                {"output_sizes": output_sizes}
                if dask_mode == "parallelized" and output_sizes is not None
                else None
            ),
        ),
    )

    if out_geometry is not None:
        assert out_ntheta is not None
        assert out_nphi is not None
        target = _rectangular_target_grid(
            out_geometry,
            out_ntheta,
            out_nphi,
            latitude_order="descending",
        )
        filtered = filtered.assign_coords(
            latitude=target.latitude,
            longitude=target.longitude,
        )
    filtered.attrs.update(data.attrs)
    filtered.name = data.name
    return filtered


def _grid_coordinate_name(
    data: xr.DataArray, axis: Literal["latitude", "longitude"]
) -> str:
    """Return the coordinate name used by a detected rectangular grid."""
    for name in (axis, "lat" if axis == "latitude" else "lon"):
        if name in data.coords and data[name].ndim == 1:
            return name
    for coordinate_name, coordinate in data.coords.items():
        if coordinate.ndim == 1 and coordinate.attrs.get("standard_name") == axis:
            return str(coordinate_name)
    raise ValueError(f"could not identify {axis} coordinate")


def _partition_mpi_xarray(
    data: xr.DataArray,
    latitude_dim: str,
    longitude_dim: str,
    backend: Backend,
) -> xr.DataArray:
    """Keep the existing rank-local SHT preprocessing partition."""
    if backend != "mpi":
        return data
    try:
        from mpi4py import MPI
    except ImportError:
        warnings.warn(
            "mpi4py not installed. Proceeding serially.",
            stacklevel=3,
        )
        return data

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    time_dims = [
        dimension
        for dimension in data.dims
        if dimension not in (latitude_dim, longitude_dim)
    ]
    if not time_dims:
        return data
    time_dim = time_dims[0]
    total_len = int(data.sizes[time_dim])
    chunk_size = total_len // size
    remainder = total_len % size
    start = rank * chunk_size + min(rank, remainder)
    stop = (rank + 1) * chunk_size + min(rank + 1, remainder)
    if start < stop:
        return data.isel({time_dim: slice(start, stop)})
    return data.isel({time_dim: slice(0, 0)})


def is_global_grid(data: xr.DataArray) -> bool:
    """
    Heuristic to determine if an xarray DataArray represents a global grid.
    Checks longitude range and presence of healpix attributes.
    """
    from ..io.data_loader import DataLoader

    # HEALPix is always global
    if data.attrs.get("grid_type") == "healpix":
        return True

    # Check longitude range
    loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
    try:
        _, lon_dim, _ = loader.get_coords()
        lon = data[lon_dim]
        nx = len(lon)
        if nx < 2:
            return False

        lon_min, lon_max = float(lon.min()), float(lon.max())
        dlon = abs(float(lon[1] - lon[0]))

        # A grid is global if the coverage + one grid spacing is >= 360
        return (lon_max - lon_min + dlon) >= 359.0
    except (ValueError, KeyError):
        return False
