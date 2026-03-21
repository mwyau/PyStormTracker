from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Literal, cast, overload

import numpy as np
import shtns
import xarray as xr
from numpy.typing import NDArray


@lru_cache(maxsize=32)
def _get_shtns_plan(nlat: int, nlon: int, lmax: int) -> shtns.sht:
    """
    Creates and caches an SHTns plan.
    The plan is thread-safe for concurrent analys/synth calls.
    """
    # Sampling theorem limit for regular grid with poles: nlat > 2*lmax
    grid_lmax = (nlat - 1) // 2
    actual_lmax = min(lmax, grid_lmax)
    mmax = min(actual_lmax, nlon // 2 - 1)

    sh = shtns.sht(actual_lmax, mmax)
    # shtns.sht_reg_poles is the standard equidistant latitude grid including poles.
    # SHT_PHI_CONTIGUOUS matches standard NumPy/C row-major layout (nlat, nlon).
    sh.set_grid(nlat, nlon, shtns.sht_reg_poles | shtns.SHT_PHI_CONTIGUOUS)

    return sh


def _filter_shtns_frame(
    frame: NDArray[np.float64], lmin: int, lmax: int, lat_reverse: bool = False
) -> NDArray[np.float64]:
    """Filters a single 2D frame using SHTns."""
    if lat_reverse:
        frame = frame[::-1, :]

    nlat, nlon = frame.shape
    grid_lmax = (nlat - 1) // 2
    if lmin > grid_lmax:
        raise ValueError(
            f"Unsupported shape for SH filter: {frame.shape}. "
            f"Grid resolution (lmax={grid_lmax}) is too low for lmin={lmin}."
        )
    
    # Ensure frame is contiguous for SHTns C-routines
    frame = np.ascontiguousarray(frame, dtype=np.float64)

    # Resolve plan. We use the maximum of the requested lmax and grid limit
    # to ensure the plan covers the desired filtering range.
    sh = _get_shtns_plan(nlat, nlon, lmax)

    # Forward transform (Spatial -> Spectral)
    # ylm is a 1D complex array
    ylm = sh.analys(frame)

    # Apply Bandpass Mask
    # sh.l contains the degree for each element in ylm.
    mask = (sh.l < lmin) | (sh.l > lmax)
    ylm[mask] = 0.0

    # Backward transform (Spectral -> Spatial)
    out = cast(NDArray[np.float64], sh.synth(ylm))

    if lat_reverse:
        out = out[::-1, :]

    return out


class SphericalHarmonicFilter:
    """
    High-performance Spherical harmonic bandpass filter for lat-lon grid data
    powered by SHTns.
    """

    def __init__(
        self, lmin: int = 5, lmax: int = 42, lat_reverse: bool = False
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
            if data.ndim == 2:
                return _filter_shtns_frame(
                    data, self.lmin, self.lmax, lat_reverse=self.lat_reverse
                )
            elif data.ndim == 3:
                out = np.empty_like(data)
                for i in range(data.shape[0]):
                    out[i] = _filter_shtns_frame(
                        data[i], self.lmin, self.lmax, lat_reverse=self.lat_reverse
                    )
                return out
            else:
                raise ValueError("numpy array must be 2D or 3D")

        return apply_sh_filter(
            data, self.lmin, self.lmax, lat_reverse=self.lat_reverse, backend=backend
        )


def apply_sh_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    lat_reverse: bool = False,
    backend: Literal["serial", "mpi", "dask"] = "serial",
) -> xr.DataArray:
    """
    Applies a spherical harmonic bandpass filter to the input data using SHTns.

    Args:
        data (xr.DataArray): Input data with dimensions containing 'lat' and 'lon'.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        lat_reverse (bool): If True, assume latitude is South to North.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.

    Returns:
        xr.DataArray: The filtered data.
    """
    if "lat" not in data.dims or "lon" not in data.dims:
        raise ValueError("Input DataArray must have 'lat' and 'lon' dimensions.")

    kwargs = {"lmin": lmin, "lmax": lmax, "lat_reverse": lat_reverse}
    dask_mode: Literal["forbidden", "allowed", "parallelized"] = "forbidden"

    if backend == "dask":
        if not data.chunks:
            warnings.warn(
                "Backend is 'dask' but data is not chunked. Proceeding serially.",
                stacklevel=2,
            )
        else:
            dask_mode = "parallelized"
    elif backend == "mpi":
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Find the time dimension to split across
            time_dims = [d for d in data.dims if d not in ("lat", "lon")]
            if time_dims:
                time_dim = time_dims[0]
                total_len = len(data[time_dim])
                chunk_size = total_len // size
                remainder = total_len % size

                s_idx = rank * chunk_size + min(rank, remainder)
                e_idx = (rank + 1) * chunk_size + min(rank, remainder)

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
            _filter_shtns_frame,
            data,
            input_core_dims=[["lat", "lon"]],
            output_core_dims=[["lat", "lon"]],
            vectorize=True,
            kwargs=kwargs,
            dask=dask_mode,
            output_dtypes=[data.dtype],
        ),
    )

    filtered.attrs.update(data.attrs)
    filtered.name = f"{data.name}_sh_filtered" if data.name else "sh_filtered"
    return filtered
