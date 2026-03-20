from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class TaperFilter:
    """
    A filter that applies a cosine taper to the edges of the domain.
    This is often used in TRACK before spherical harmonic filtering to
    minimize ringing at the boundaries.
    """

    def __init__(self, n_points: int = 10) -> None:
        """
        Initialize the taper filter.

        Args:
            n_points (int): Number of points to taper at the edges.
        """
        self.n_points = n_points

    def filter(self, data: xr.DataArray | NDArray[np.float64]) -> xr.DataArray | NDArray[np.float64]:
        """
        Applies a cosine taper to the edges of the input data.

        Args:
            data (xr.DataArray | np.ndarray): Input data.

        Returns:
            xr.DataArray | np.ndarray: The tapered data.
        """
        if isinstance(data, xr.DataArray):
            return self._filter_xarray(data)
        elif isinstance(data, np.ndarray):
            return self._filter_numpy(data)
        else:
            raise TypeError("data must be an xarray.DataArray or a numpy.ndarray")

    def _filter_numpy(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies tapering to a numpy array."""
        if data.ndim < 2:
            raise ValueError("numpy array must be at least 2D")

        # Tapering only for the spatial dimensions (assumed to be the last two)
        ny, nx = data.shape[-2:]
        taper_y = self._get_taper(ny)
        taper_x = self._get_taper(nx)

        # Create a 2D taper mask
        mask = np.outer(taper_y, taper_x)

        # Apply to all leading dimensions (e.g., time)
        return data * mask

    def _filter_xarray(self, data: xr.DataArray) -> xr.DataArray:
        """Applies tapering to an xarray DataArray."""
        if "lat" not in data.dims or "lon" not in data.dims:
            raise ValueError("Input DataArray must have 'lat' and 'lon' dimensions.")

        ny = len(data.lat)
        nx = len(data.lon)
        taper_y = self._get_taper(ny)
        taper_x = self._get_taper(nx)

        # Mask as a DataArray for easy broadcasting
        mask = xr.DataArray(
            np.outer(taper_y, taper_x),
            dims=("lat", "lon"),
            coords={"lat": data.lat, "lon": data.lon},
        )

        return data * mask

    def _get_taper(self, n: int) -> NDArray[np.float64]:
        """Generates a 1D cosine taper vector."""
        taper = np.ones(n, dtype=np.float64)
        if self.n_points <= 0:
            return taper

        # Ensure n_points doesn't exceed half the dimension size
        n_eff = min(self.n_points, n // 2)

        # Cosine taper from 0 to 1 over n_eff points
        weights = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_eff) / n_eff))

        taper[:n_eff] = weights
        taper[-n_eff:] = weights[::-1]

        return taper
