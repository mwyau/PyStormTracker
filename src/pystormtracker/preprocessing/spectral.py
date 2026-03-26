from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Callable
from typing import Literal, TypedDict, cast, overload

import ducc0
import numpy as np
import xarray as xr
from numpy.typing import NDArray

try:
    import shtns  # type: ignore[import-untyped]

    SHTNS_AVAILABLE = True
except ImportError:
    SHTNS_AVAILABLE = False


class FilterKwargs(TypedDict, total=False):
    lmin: int
    lmax: int
    lat_reverse: bool
    backend: str
    nthreads: int
    sht_engine: str


def _resolve_engine(
    sht_engine: Literal["auto", "shtns", "ducc0"],
) -> Literal["shtns", "ducc0"]:
    """Resolves 'auto' engine to the best available backend."""
    if sht_engine == "auto":
        return "ducc0"
    return sht_engine


def _get_filter_config(
    sht_engine: Literal["auto", "shtns", "ducc0"],
    lmin: int,
    lmax: int,
    lat_reverse: bool,
    nthreads: int = 1,
) -> tuple[Callable[..., NDArray[np.float64]], FilterKwargs]:
    """Returns the filter function and kwargs for the resolved engine."""
    resolved_engine = _resolve_engine(sht_engine)
    kwargs: FilterKwargs = {
        "lmin": lmin,
        "lmax": lmax,
        "lat_reverse": lat_reverse,
    }

    if resolved_engine == "shtns":
        if not SHTNS_AVAILABLE:
            raise ImportError("shtns is requested but not available.")
        return _filter_shtns_frame, kwargs

    # ducc0
    kwargs["nthreads"] = nthreads

    return _filter_ducc0_frame, kwargs


# Thread-local storage to ensure each Dask thread has its own SHTns object.
# Sharing one SHTns object across threads is UNSAFE due to internal buffers.
_thread_local = threading.local()


def _get_shtns_plan(nlat: int, nlon: int, lmax: int) -> shtns.sht:
    """
    Retrieves or creates an SHTns plan for the current thread.
    """
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}

    key = (nlat, nlon, lmax)
    if key not in _thread_local.cache:
        # mmax is derived from the longitude grid resolution to satisfy the
        # sampling theorem (mmax < nlon/2).
        mmax = min(lmax, nlon // 2 - 1)

        # Use 4pi normalization (shtns.sht_fourpi) to ensure parity with
        # NCL/Spherepack and Hodges TRACK. This ensures coefficients are
        # scaled such that they represent the amplitude of the harmonics.
        sh = shtns.sht(lmax, mmax, norm=shtns.sht_fourpi)

        # shtns.sht_reg_poles is the standard equidistant latitude grid
        # including poles, which is the most common format for climate data (e.g. ERA5).
        # SHT_PHI_CONTIGUOUS matches standard NumPy/C row-major layout (nlat, nlon).
        # polar_opt=0.0 (equivalent to eps=0.0 in C API) disables polar
        # optimization to ensure maximum accuracy for low-resolution grids.
        sh.set_grid(nlat, nlon, flags=shtns.sht_reg_poles | shtns.SHT_PHI_CONTIGUOUS, polar_opt=0.0)
        _thread_local.cache[key] = sh

    return _thread_local.cache[key]


def _filter_shtns_frame(
    frame: NDArray[np.float64], lmin: int, lmax: int, lat_reverse: bool = False
) -> NDArray[np.float64]:
    """Filters a single 2D frame using SHTns."""
    if lat_reverse:
        frame = frame[::-1, :]

    nlat, nlon = frame.shape
    # Basic check against the sampling theorem limit for the latitude grid.
    grid_lmax = (nlat - 1) // 2
    if lmin > grid_lmax:
        raise ValueError(
            f"Unsupported shape for spectral filter: {frame.shape}. "
            f"Grid resolution (lmax={grid_lmax}) is too low for lmin={lmin}."
        )

    # Ensure frame is contiguous for SHTns C-routines
    frame = np.ascontiguousarray(frame, dtype=np.float64)

    # Resolve plan for this thread.
    sh = _get_shtns_plan(nlat, nlon, lmax)

    # Forward transform (Spatial -> Spectral)
    ylm = sh.analys(frame)

    # Apply Bandpass Mask (Zero out coefficients outside [lmin, lmax])
    mask = (sh.l < lmin) | (sh.l > lmax)
    ylm[mask] = 0.0

    # Backward transform (Spectral -> Spatial)
    out = cast(NDArray[np.float64], sh.synth(ylm))

    if lat_reverse:
        out = out[::-1, :]

    return out


def _filter_ducc0_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    lat_reverse: bool = False,
    nthreads: int = 1,
) -> NDArray[np.float64]:
    """Filters a single 2D frame using ducc0."""
    if lat_reverse:
        frame = frame[::-1, :]

    nlat, nlon = frame.shape

    # geometry='CC' (Clenshaw-Curtis) assumes an equidistant grid including
    # the poles, matching standard lat-lon climate data.
    # For Gaussian grids, 'GL' (Gauss-Legendre) would be more appropriate.
    geometry = "CC"
    mmax = min(lmax, nlon // 2 - 1)

    try:
        alm = ducc0.sht.analysis_2d(
            map=np.expand_dims(frame, axis=0),
            spin=0,
            lmax=lmax,
            mmax=mmax,
            geometry=geometry,
            nthreads=nthreads,
        )

        # Apply Bandpass Mask
        # Coefficients are stored in a packed format: for each m, l goes from m to lmax.
        l_arr = np.concatenate([np.arange(m, lmax + 1) for m in range(mmax + 1)])
        if lmin > 0:
            mask = l_arr < lmin
            alm[0, mask] = 0.0

        out = cast(
            NDArray[np.float64],
            ducc0.sht.synthesis_2d(
                alm=alm,
                spin=0,
                lmax=lmax,
                mmax=mmax,
                ntheta=nlat,
                nphi=nlon,
                geometry=geometry,
                nthreads=nthreads,
            )[0],
        )

        if lat_reverse:
            out = out[::-1, :]

        return out
    except Exception as e:
        msg = f"Unsupported shape for spectral filter: {frame.shape}. {e}"
        raise ValueError(msg) from e


class SpectralFilter:
    """
    Spectral bandpass filter (truncation) for lat-lon grid data.
    Backends (in order of preference):
    - shtns: Highly optimized for performance, requires C library installation.
    - ducc0: Pure Python/C++ library with no external C dependencies, easy installation.
    """

    def __init__(
        self,
        lmin: int = 5,
        lmax: int = 42,
        lat_reverse: bool = False,
        sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
    ) -> None:
        """
        Initialize the filter with wave number bounds.

        Args:
            lmin (int): Minimum total wave number to retain.
            lmax (int): Maximum total wave number to retain.
            lat_reverse (bool): If True, assume latitude is stored from South to North.
            sht_engine (str): Engine to use ('auto', 'shtns', 'ducc0').
                - 'shtns' is preferred for high-performance iterative filters.
                - 'ducc0' is used for robust, thread-safe transforms.
        """
        self.lmin = lmin
        self.lmax = lmax
        self.lat_reverse = lat_reverse
        self.sht_engine = sht_engine

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
            filter_func, kwargs = _get_filter_config(
                self.sht_engine, self.lmin, self.lmax, self.lat_reverse, nthreads
            )

            if data.ndim == 2:
                return filter_func(data, **kwargs)
            elif data.ndim == 3:
                out = np.empty_like(data)
                for i in range(data.shape[0]):
                    out[i] = filter_func(data[i], **kwargs)
                return out
            else:
                raise ValueError("numpy array must be 2D or 3D")

        return apply_spectral_filter(
            data,
            self.lmin,
            self.lmax,
            lat_reverse=self.lat_reverse,
            backend=backend,
            sht_engine=self.sht_engine,
        )


def apply_spectral_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    lat_reverse: bool = False,
    backend: Literal["serial", "mpi", "dask"] = "serial",
    sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
) -> xr.DataArray:
    """
    Applies a spectral bandpass filter to the input DataArray.

    Args:
        data (xr.DataArray): Input data with lat/lon dimensions.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        lat_reverse (bool): If True, assume latitude is South to North.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.
        sht_engine (str): Engine. Options: 'auto', 'shtns', 'ducc0'.

    Returns:
        xr.DataArray: The filtered data.
    """
    from ..io.loader import DataLoader

    # Identify spatial dimensions
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

    nthreads = 1 if backend in ("mpi", "dask") else 0
    filter_func, kwargs = _get_filter_config(
        sht_engine, lmin, lmax, lat_reverse, nthreads
    )

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
            time_dims = [d for d in data.dims if d not in (lat_dim, lon_dim)]
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

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            filter_func,
            data,
            input_core_dims=[[lat_dim, lon_dim]],
            output_core_dims=[[lat_dim, lon_dim]],
            vectorize=True,
            kwargs=kwargs,
            dask=dask_mode,
            output_dtypes=[data.dtype],
        ),
    )

    filtered.attrs.update(data.attrs)
    filtered.name = (
        f"{data.name}_spectral_filtered" if data.name else "spectral_filtered"
    )
    return filtered
