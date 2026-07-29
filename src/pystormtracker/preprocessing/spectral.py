from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Literal, TypedDict, cast, overload

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray


class FilterKwargs(TypedDict, total=False):
    lmin: int
    lmax: int
    lat_reverse: bool
    nthreads: int
    taper_val: float
    geometry: str
    theta: NDArray[np.float64]
    nphi: NDArray[np.uint64]
    phi0: NDArray[np.float64]
    ringstart: NDArray[np.uint64]
    out_geometry: str
    out_ntheta: int
    out_nphi: int


def _get_filter_config(
    lmin: int,
    lmax: int,
    lat_reverse: bool,
    nthreads: int = 1,
    taper_val: float = 0.1,
    geometry: str = "CC",
) -> tuple[Callable[..., NDArray[np.float64]], FilterKwargs]:
    """Returns the filter function and kwargs for the requested engine."""
    kwargs: FilterKwargs = {
        "lmin": lmin,
        "lmax": lmax,
        "lat_reverse": lat_reverse,
        "nthreads": nthreads,
        "taper_val": taper_val,
        "geometry": geometry,
    }

    return _filter_ducc0_frame, kwargs


def apply_bandpass_mask_to_alm(
    alm: NDArray[np.complex128],
    lmin: int,
    lmax: int,
    mmax: int | None = None,
    taper_val: float = 0.1,
) -> None:
    """
    Applies a tapered bandpass mask in-place to spherical harmonic coefficients.
    Uses the form from Hoskins and Sardeshmukh (1984), acting like a ∇⁴ smoother.

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


def _filter_ducc0_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    lat_reverse: bool = False,
    nthreads: int = 1,
    taper_val: float = 0.1,
    geometry: str = "CC",
    # Grid metadata for 1D maps (Reduced Gaussian / HEALPix)
    theta: NDArray[np.float64] | None = None,
    nphi: NDArray[np.uint64] | None = None,
    phi0: NDArray[np.float64] | None = None,
    ringstart: NDArray[np.uint64] | None = None,
    # Regridding options
    out_geometry: str | None = None,
    out_ntheta: int | None = None,
    out_nphi: int | None = None,
) -> NDArray[np.float64]:
    """
    Filters a single spatial frame using ducc0.
    Supports both 2D structured grids and 1D unstructured grids.
    """
    is_1d = frame.ndim == 1
    if not is_1d and not lat_reverse:
        frame = frame[::-1, :]

    nlat: int
    nlon: int
    if is_1d:
        if theta is None or nphi is None or phi0 is None or ringstart is None:
            raise ValueError("Grid metadata required for 1D maps.")
        nlat = len(theta)
        nlon = int(np.max(nphi))
    else:
        nlat, nlon = frame.shape

    if nlat < lmax + 1:
        raise ValueError(
            f"Unsupported shape for spectral filter: {frame.shape} cannot "
            f"represent lmax={lmax}."
        )

    mmax = min(lmax, nlon // 2 - 1)

    try:
        # 1. Analysis: Map -> Spherical Harmonics (ALM)
        alm: NDArray[np.complex128]
        if is_1d:
            # Use iterative pseudo-analysis for unstructured/reduced grids
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
        else:
            alm = ducc0.sht.analysis_2d(
                map=np.expand_dims(frame, axis=0),
                spin=0,
                lmax=lmax,
                mmax=mmax,
                geometry=geometry,
                nthreads=nthreads,
            )

        # 2. Filter: Apply tapered mask to ALMs
        apply_bandpass_mask_to_alm(alm, lmin, lmax, mmax, taper_val=taper_val)

        # 3. Synthesis: ALM -> Map
        synth_geometry = out_geometry or geometry
        synth_ntheta = out_ntheta or nlat
        synth_nphi = out_nphi or nlon

        out: NDArray[np.float64]
        if is_1d and out_geometry is None:
            # Synthesize back to the SAME 1D grid
            out = cast(
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
                    geometry=synth_geometry,
                    nthreads=nthreads,
                )[0],
            )
        else:
            # Synthesize to a 2D structured grid
            out = cast(
                NDArray[np.float64],
                ducc0.sht.synthesis_2d(
                    alm=alm,
                    spin=0,
                    lmax=lmax,
                    mmax=mmax,
                    ntheta=synth_ntheta,
                    nphi=synth_nphi,
                    geometry=synth_geometry,
                    nthreads=nthreads,
                )[0],
            )

        if not is_1d and not lat_reverse:
            out = out[::-1, :]

        return out
    except Exception as e:
        msg = f"Spectral filter failed for shape {frame.shape}: {e}"
        raise ValueError(msg) from e


def apply_dct_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    backend: Literal["serial", "mpi", "dask"] = "serial",
    taper_val: float = 0.1,
) -> xr.DataArray:
    """
    Applies a DCT-based spectral bandpass filter to a regional DataArray.
    Uses ducc0.fft.dct for high-performance 2D DCT.

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

    # Identify spatial dimensions
    lat_dim = next(
        (c for c in DataLoader.VAR_MAPPING["latitude"] if c in data.dims), None
    )
    lon_dim = next(
        (c for c in DataLoader.VAR_MAPPING["longitude"] if c in data.dims), None
    )

    if not lat_dim or not lon_dim:
        raise ValueError("Input DataArray must have latitude and longitude dimensions.")

    # DCT filter function for xr.apply_ufunc
    def _dct_filter_2d(
        frame: NDArray[np.float64],
        lmin: int,
        lmax: int,
        taper_val: float,
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ny, nx = frame.shape

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

    # Prepare for xarray
    lat = data[lat_dim].values
    lon = data[lon_dim].values

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = (
        "parallelized" if data.chunks and backend == "dask" else "allowed"
    )

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            _dct_filter_2d,
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
    filtered.name = f"{data.name}_dct_filtered" if data.name else "dct_filtered"
    return filtered


class DCTFilter:
    """
    Spectral bandpass filter for regional grids using Discrete Cosine Transform.
    """

    def __init__(
        self,
        lmin: int = 5,
        lmax: int = 42,
        taper_val: float = 0.1,
    ) -> None:
        self.lmin = lmin
        self.lmax = lmax
        self.taper_val = taper_val

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
        if isinstance(data, np.ndarray):
            # For numpy arrays, we'd need lat/lon or assume a default scaling.
            # In PyStormTracker, we prefer DataArrays for regional metadata.
            raise NotImplementedError("DCTFilter currently requires xarray.DataArray")

        return apply_dct_filter(
            data,
            self.lmin,
            self.lmax,
            backend=backend,
            taper_val=self.taper_val,
        )


class SHTFilter:
    """
    Spectral bandpass filter (truncation) for lat-lon grid data using ducc0.
    """

    def __init__(
        self,
        lmin: int = 5,
        lmax: int = 42,
        lat_reverse: bool = False,
        taper_val: float = 0.1,
        geometry: Literal["CC", "GL", "DH", "auto"] = "auto",
    ) -> None:
        """
        Initialize the filter with wave number bounds.

        Args:
            lmin (int): Minimum total wave number to retain.
            lmax (int): Maximum total wave number to retain.
            lat_reverse (bool): If True, assume latitude is North to South (reversed).
            taper_val (float): Value of the taper at lmax.
            geometry (str): Grid geometry ('CC', 'GL', 'DH', or 'auto').
        """
        self.lmin = lmin
        self.lmax = lmax
        self.lat_reverse = lat_reverse
        self.taper_val = taper_val
        self.geometry = geometry

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
            nthreads = 1 if backend in ("mpi", "dask") else 0
            # For numpy arrays, we can't auto-detect geometry easily without lat array.
            # Default to CC or use provided geometry if not auto.
            geom = "CC" if self.geometry == "auto" else str(self.geometry)
            filter_func, kwargs = _get_filter_config(
                self.lmin,
                self.lmax,
                self.lat_reverse,
                nthreads,
                self.taper_val,
                geometry=geom,
            )
            if data.ndim == 2:
                return filter_func(data, **kwargs)
            if data.ndim == 3:
                out = np.empty_like(data)
                for i in range(data.shape[0]):
                    out[i] = filter_func(data[i], **kwargs)
                return out
            raise ValueError("numpy array must be 2D or 3D")

        return apply_sht_filter(
            data,
            self.lmin,
            self.lmax,
            lat_reverse=self.lat_reverse,
            backend=backend,
            taper_val=self.taper_val,
            geometry=self.geometry,
        )


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


def apply_sht_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    lat_reverse: bool = False,
    backend: Literal["serial", "mpi", "dask"] = "serial",
    taper_val: float = 0.1,
    geometry: Literal["CC", "GL", "DH", "auto"] = "auto",
    out_geometry: str | None = None,
    out_ntheta: int | None = None,
    out_nphi: int | None = None,
) -> xr.DataArray:
    """
    Applies a spectral bandpass filter to the input DataArray.
    Supports regular 2D grids and reduced Gaussian 1D grids.

    Args:
        data (xr.DataArray): Input data.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        lat_reverse (bool): If True, assume latitude is South to North.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.
        taper_val (float): Value of the taper at lmax.
        geometry (str): Grid geometry ('CC', 'GL', 'DH', or 'auto').
        out_geometry (str): Output geometry (e.g., 'CC', 'GL', 'HEALPix').
        out_ntheta (int): Number of latitudes in output grid.
        out_nphi (int): Number of longitudes in output grid.

    Returns:
        xr.DataArray: The filtered (and possibly regridded) data.
    """
    from ..io.data_loader import DataLoader

    varname = str(data.name) if data.name is not None else ""
    loader = DataLoader(data.dataset if hasattr(data, "dataset") else data)
    is_reduced = loader.is_reduced_gaussian(varname)

    # Identify spatial dimensions
    lat_dim: str | None = None
    lon_dim: str | None = None
    spatial_dim: str | None = None

    nthreads = 1 if backend in ("mpi", "dask") else 0

    # Grid geometry detection
    detected_geometry = str(geometry)
    if geometry == "auto":
        if is_reduced:
            # cfgrib usually provides N parameter for Gaussian grids
            n_gauss = data.attrs.get("GRIB_N")
            detected_geometry = "GL" if n_gauss else "CC"
        else:
            lat_search_dim = next(
                (c for c in DataLoader.VAR_MAPPING["latitude"] if c in data.dims),
                "latitude",
            )
            lat_vals = data[lat_search_dim].values
            if len(lat_vals) > 1:
                dlat = np.diff(lat_vals)
                # Gaussian grids have non-uniform latitude spacing
                detected_geometry = "GL" if np.ptp(dlat) > 1e-4 else "CC"
            else:
                detected_geometry = "CC"

    kwargs: FilterKwargs = {
        "lmin": lmin,
        "lmax": lmax,
        "lat_reverse": True,
        "nthreads": nthreads,
        "taper_val": taper_val,
        "geometry": detected_geometry,
    }

    if out_geometry:
        kwargs["out_geometry"] = out_geometry
    if out_ntheta:
        kwargs["out_ntheta"] = out_ntheta
    if out_nphi:
        kwargs["out_nphi"] = out_nphi

    if is_reduced:
        spatial_dim = "values" if "values" in data.dims else str(data.dims[-1])
        grid_meta = loader.get_grid_metadata(varname)
        # We know grid_meta keys match FilterKwargs keys for 1D maps
        kwargs.update(grid_meta)  # type: ignore[typeddict-item]
    else:
        lat_dim = next(
            (c for c in DataLoader.VAR_MAPPING["latitude"] if c in data.dims), None
        )
        lon_dim = next(
            (c for c in DataLoader.VAR_MAPPING["longitude"] if c in data.dims), None
        )

        if not lat_dim or not lon_dim:
            raise ValueError(
                f"Input DataArray must have latitude and longitude dimensions. "
                f"Found: {list(data.dims)}"
            )

    # Ensure latitude is North to South for ducc0
    # Store original order to restore it later if needed
    is_ascending = not loader.is_lat_reversed()
    if not is_reduced:
        data = data.sortby(lat_dim, ascending=False)

    dask_mode: Literal["forbidden", "allowed", "parallelized"] = "forbidden"

    if data.chunks:
        # If data is chunked, we must allow or parallelize dask handling
        if backend == "dask":
            # Prevent OpenMP oversubscription when Dask is handling parallelism
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            dask_mode = "parallelized"
        else:
            # For serial or MPI with chunked data, use 'allowed' to run on chunks
            dask_mode = "allowed"

    if backend == "mpi":
        try:
            from mpi4py import MPI

            # Prevent OpenMP oversubscription when MPI is handling parallelism
            os.environ.setdefault("OMP_NUM_THREADS", "1")

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Find the time dimension to split across
            exclude_dims = (spatial_dim,) if is_reduced else (lat_dim, lon_dim)
            time_dims = [d for d in data.dims if d not in exclude_dims]
            if time_dims:
                time_dim = time_dims[0]
                total_len = len(data[time_dim])
                chunk_size = total_len // size
                remainder = total_len % size

                s_idx = rank * chunk_size + min(rank, remainder)
                e_idx = (rank + 1) * chunk_size + min(rank + 1, remainder)

                if s_idx < e_idx:
                    data = data.isel({time_dim: slice(s_idx, e_idx)})
                else:
                    data = data.isel({time_dim: slice(0, 0)})
        except ImportError:
            warnings.warn(
                "mpi4py not installed. Proceeding serially.",
                stacklevel=2,
            )

    input_core_dims = [[spatial_dim]] if is_reduced else [[lat_dim, lon_dim]]

    # Determine output core dims and coords
    output_core_dims = [["lat_out", "lon_out"]] if out_geometry else input_core_dims

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            _filter_ducc0_frame,
            data,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            vectorize=True,
            kwargs=kwargs,
            dask=dask_mode,
            output_dtypes=[data.dtype],
        ),
    )

    # Handle coordinates for regridded output
    if out_geometry:
        # We need to construct actual coordinates for the output
        ntheta_out = int(out_ntheta) if out_ntheta else 0
        nphi_out = int(out_nphi) if out_nphi else 0

        if out_geometry == "CC":
            lats_out = np.linspace(-90, 90, ntheta_out)
            lons_out = np.linspace(0, 360, nphi_out, endpoint=False)
        elif out_geometry == "GL":
            lats_out = 90.0 - np.degrees(ducc0.misc.GL_thetas(ntheta_out))
            lons_out = np.linspace(0, 360, nphi_out, endpoint=False)
        else:
            # Placeholder for other geometries
            lats_out = np.arange(ntheta_out, dtype=np.float64)
            lons_out = np.arange(nphi_out, dtype=np.float64)

        filtered = filtered.rename({"lat_out": "latitude", "lon_out": "longitude"})
        filtered = filtered.assign_coords(latitude=lats_out, longitude=lons_out)

    filtered.attrs.update(data.attrs)
    filtered.name = (
        f"{data.name}_spectral_filtered" if data.name else "spectral_filtered"
    )

    if not is_reduced and is_ascending and not out_geometry:
        filtered = filtered.sortby(lat_dim, ascending=True)

    return filtered
