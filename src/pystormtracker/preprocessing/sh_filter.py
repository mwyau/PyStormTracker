from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pyshtools as pysh
import xarray as xr


def _filter_sh_frame(frame: np.ndarray, lmin: int, lmax: int) -> np.ndarray:
    """Filters a single 2D frame using spherical harmonics."""
    nlat, nlon = frame.shape
    pad_pole = False

    # Check for expected regular lat-lon grid shapes (Driscoll-Healy)
    if nlon == 2 * (nlat - 1):
        pad_pole = True
        frame_proc = frame[:-1, :]
    elif nlon == 2 * nlat:
        frame_proc = frame
    else:
        raise ValueError(
            f"Unsupported shape for SH filter: {frame.shape}. "
            f"Expected nlon=2*(nlat-1) or nlon=2*nlat for regular lat-lon grid."
        )

    grid = pysh.SHGrid.from_array(frame_proc, grid="DH")
    coeffs = grid.expand()

    # Filter operations
    coeffs_filtered = coeffs.copy()
    for l_val in range(min(lmin, coeffs_filtered.lmax + 1)):
        coeffs_filtered.coeffs[:, l_val, :] = 0.0

    if lmax < coeffs_filtered.lmax:
        for l_val in range(lmax + 1, coeffs_filtered.lmax + 1):
            coeffs_filtered.coeffs[:, l_val, :] = 0.0

    filtered_grid = coeffs_filtered.expand()
    out = filtered_grid.to_array()

    if pad_pole:
        return out[:, :-1]
    else:
        return out[:-1, :-1]


def apply_sh_filter(
    data: xr.DataArray,
    lmin: int = 5,
    lmax: int = 42,
    backend: Literal["serial", "mpi", "dask"] = "serial",
) -> xr.DataArray:
    """
    Applies a spherical harmonic bandpass filter to the input data.

    Args:
        data (xr.DataArray): Input data with dimensions containing 'lat' and 'lon'.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.

    Returns:
        xr.DataArray: The filtered data. If 'mpi', the returned DataArray
        may be a subset corresponding to the MPI rank's chunk.
    """
    if "lat" not in data.dims or "lon" not in data.dims:
        raise ValueError("Input DataArray must have 'lat' and 'lon' dimensions.")

    kwargs = {"lmin": lmin, "lmax": lmax}
    dask_mode = "forbidden"

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
                e_idx = (rank + 1) * chunk_size + min(rank + 1, remainder)

                if s_idx < e_idx:
                    data = data.isel({time_dim: slice(s_idx, e_idx)})
                else:
                    # Empty slice for this rank
                    data = data.isel({time_dim: slice(0, 0)})
        except ImportError:
            warnings.warn(
                "mpi4py not installed. Proceeding serially.",
                stacklevel=2,
            )

    filtered = xr.apply_ufunc(
        _filter_sh_frame,
        data,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        kwargs=kwargs,
        dask=dask_mode,
        output_dtypes=[data.dtype],
    )

    filtered.attrs.update(data.attrs)
    filtered.name = f"{data.name}_sh_filtered" if data.name else "sh_filtered"
    return filtered
