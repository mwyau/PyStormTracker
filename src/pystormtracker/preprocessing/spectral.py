from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Literal, TypedDict, cast, overload

import ducc0
import jax.numpy as jnp
import numpy as np
import xarray as xr
from numpy.typing import NDArray


class FilterKwargs(TypedDict, total=False):
    lmin: int
    lmax: int
    lat_reverse: bool
    nthreads: int


def _get_filter_config(
    lmin: int,
    lmax: int,
    lat_reverse: bool,
    nthreads: int = 1,
    sht_engine: Literal["ducc0", "jax"] = "ducc0",
) -> tuple[Callable[..., NDArray[np.float64]], FilterKwargs]:
    """Returns the filter function and kwargs for the requested engine."""
    kwargs: FilterKwargs = {
        "lmin": lmin,
        "lmax": lmax,
        "lat_reverse": lat_reverse,
        "nthreads": nthreads,
    }

    if sht_engine == "jax":
        return _filter_jax_frame, kwargs

    return _filter_ducc0_frame, kwargs


def _filter_jax_frame(
    frame: NDArray[np.float64],
    lmin: int,
    lmax: int,
    lat_reverse: bool = False,
    nthreads: int = 1,
) -> NDArray[np.float64]:
    """Filters a single 2D frame using JAX-native SHT parity backend."""
    try:
        import jax

        from .jax_sht import jax_analysis_2d, jax_synthesis_2d

        jax.config.update("jax_enable_x64", True)
    except ImportError as e:
        raise ImportError(
            "The 'jax' backend requires 'jax'. "
            "Install via 'pip install pystormtracker[jax]'."
        ) from e

    if lat_reverse:
        frame = frame[::-1, :]

    ny, nx = frame.shape
    mmax = min(lmax, nx // 2 - 1)
    frame_jax = jax.device_put(frame)

    # Forward transform
    alm = cast(jnp.ndarray, jax_analysis_2d(frame_jax, lmax, mmax=mmax, geometry="CC"))

    # Apply Bandpass Mask
    # We need to calculate the degree 'l' for each coefficient to mask it.
    l_list = []
    for m in range(mmax + 1):
        l_list.append(jnp.arange(m, lmax + 1))
    l_arr = jnp.concatenate(l_list)

    mask = l_arr >= lmin
    alm_filtered = alm * mask

    # Inverse transform
    out = jax_synthesis_2d(alm_filtered, ny, nx, lmax, mmax=mmax, geometry="CC")
    out_np = np.asarray(out)

    if lat_reverse:
        out_np = out_np[::-1, :]

    return out_np


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
    Spectral bandpass filter (truncation) for lat-lon grid data using ducc0.
    """

    def __init__(
        self,
        lmin: int = 5,
        lmax: int = 42,
        lat_reverse: bool = False,
    ) -> None:
        """
        Initialize the filter with wave number bounds.

        Args:
            lmin (int): Minimum total wave number to retain.
            lmax (int): Maximum total wave number to retain.
            lat_reverse (bool): If True, assume latitude is stored from South to North.
        """
        self.lmin = lmin
        self.lmax = lmax
        self.lat_reverse = lat_reverse

    @overload
    def filter(
        self,
        data: xr.DataArray,
        backend: Literal["serial", "mpi", "dask"] = "serial",
        sht_engine: Literal["ducc0", "jax"] = "ducc0",
    ) -> xr.DataArray: ...

    @overload
    def filter(
        self,
        data: NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask"] = "serial",
        sht_engine: Literal["ducc0", "jax"] = "ducc0",
    ) -> NDArray[np.float64]: ...

    def filter(
        self,
        data: xr.DataArray | NDArray[np.float64],
        backend: Literal["serial", "mpi", "dask"] = "serial",
        sht_engine: Literal["ducc0", "jax"] = "ducc0",
    ) -> xr.DataArray | NDArray[np.float64]:
        """
        Applies the filter to the input data.

        Args:
            data (xr.DataArray | np.ndarray): Input data.
            backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.
            sht_engine (str): SHT engine. Options: 'ducc0', 'jax'.

        Returns:
            xr.DataArray | np.ndarray: The filtered data.
        """
        if isinstance(data, np.ndarray):
            nthreads = 1 if backend in ("mpi", "dask") else 0
            filter_func, kwargs = _get_filter_config(
                self.lmin,
                self.lmax,
                self.lat_reverse,
                nthreads,
                sht_engine=sht_engine,
            )

            if data.ndim == 2:
                return filter_func(data, **kwargs)
            if data.ndim == 3:
                out = np.empty_like(data)
                for i in range(data.shape[0]):
                    out[i] = filter_func(data[i], **kwargs)
                return out
            raise ValueError("numpy array must be 2D or 3D")

        return apply_spectral_filter(
            data,
            self.lmin,
            self.lmax,
            lat_reverse=self.lat_reverse,
            backend=backend,
            sht_engine=sht_engine,
        )


def apply_spectral_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    lat_reverse: bool = False,
    backend: Literal["serial", "mpi", "dask"] = "serial",
    sht_engine: Literal["ducc0", "jax"] = "ducc0",
) -> xr.DataArray:
    """
    Applies a spectral bandpass filter to the input DataArray.

    Args:
        data (xr.DataArray): Input data with lat/lon dimensions.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        lat_reverse (bool): If True, assume latitude is South to North.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.
        sht_engine (str): SHT engine. Options: 'ducc0', 'jax'.

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
        lmin, lmax, lat_reverse, nthreads, sht_engine=sht_engine
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
