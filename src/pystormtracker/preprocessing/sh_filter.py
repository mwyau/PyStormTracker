from __future__ import annotations

import warnings
from typing import Literal, cast, overload

import numpy as np
import pyshtools as pysh  # type: ignore[import-untyped]
import xarray as xr
from numpy.typing import NDArray


def _filter_sh_frame(
    frame: NDArray[np.float64], lmin: int, lmax: int, lat_reverse: bool = False
) -> NDArray[np.float64]:
    """Filters a single 2D frame using spherical harmonics."""
    if lat_reverse:
        frame = frame[::-1, :]

    nlat, nlon = frame.shape

    # pyshtools DH grid requires nlon = 2*nlat-1 (extended) or nlon = 2*nlat.
    # Typical ERA5 2.5x2.5 is 73x144. nlon=144, nlat=73. 2*nlat-1 = 145.
    # Typical ERA5 0.25x0.25 is 721x1440. nlon=1440, nlat=721. 2*nlat-1 = 1441.

    if nlon == 2 * nlat - 2:
        # Pad longitude by repeating first column at the end to get 2*nlat-1
        frame_proc = np.empty((nlat, nlon + 1), dtype=frame.dtype)
        frame_proc[:, :-1] = frame
        frame_proc[:, -1] = frame[:, 0]
        slice_back = True
    elif nlon == 2 * nlat - 1 or nlon == 2 * nlat:
        frame_proc = frame
        slice_back = False
    else:
        # Try to use as is, let pyshtools raise if incompatible
        frame_proc = frame
        slice_back = False

    try:
        grid = pysh.SHGrid.from_array(frame_proc, grid="DH")
        coeffs = grid.expand()

        # Filter operations
        coeffs_filtered = coeffs.copy()
        if lmin > 0:
            for l_val in range(min(lmin, coeffs_filtered.lmax + 1)):
                coeffs_filtered.coeffs[:, l_val, :] = 0.0

        if lmax < coeffs_filtered.lmax:
            for l_val in range(lmax + 1, coeffs_filtered.lmax + 1):
                coeffs_filtered.coeffs[:, l_val, :] = 0.0

        filtered_grid = coeffs_filtered.expand()
        out = cast(NDArray[np.float64], filtered_grid.to_array())

        if slice_back:
            out = out[:, :-1]

        if lat_reverse:
            out = out[::-1, :]

        return out
    except Exception as e:
        raise ValueError(f"Unsupported shape for SH filter: {frame.shape}. {e}")


class SphericalHarmonicFilter:
    """
    Spherical harmonic bandpass filter for lat-lon grid data.
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
                return _filter_sh_frame(
                    data, self.lmin, self.lmax, lat_reverse=self.lat_reverse
                )
            elif data.ndim == 3:
                out = np.empty_like(data)
                for i in range(data.shape[0]):
                    out[i] = _filter_sh_frame(
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
    Applies a spherical harmonic bandpass filter to the input data.

    Args:
        data (xr.DataArray): Input data with dimensions containing 'lat' and 'lon'.
        lmin (int): Minimum total wave number to retain. Defaults to 5.
        lmax (int): Maximum total wave number to retain. Defaults to 42.
        lat_reverse (bool): If True, assume latitude is South to North.
        backend (str): Parallelization backend. Options: 'serial', 'mpi', 'dask'.

    Returns:
        xr.DataArray: The filtered data. If 'mpi', the returned DataArray
        may be a subset corresponding to the MPI rank's chunk.
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
                    # Empty slice for this rank
                    data = data.isel({time_dim: slice(0, 0)})
        except ImportError:
            warnings.warn(
                "mpi4py not installed. Proceeding serially.",
                stacklevel=2,
            )

    filtered = cast(
        xr.DataArray,
        xr.apply_ufunc(
            _filter_sh_frame,
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
