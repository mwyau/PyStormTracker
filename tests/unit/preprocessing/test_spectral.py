from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.preprocessing import SpectralFilter


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_serial(ny: int, nx: int) -> None:
    # Test with both 2.5-deg and 0.25-deg
    data: NDArray[np.float64] = np.random.rand(2, ny, nx)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0, 1],
            "lat": np.linspace(90, -90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    filt = SpectralFilter(lmin=5, lmax=42)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (2, ny, nx)
    assert filtered.dims == ("time", "lat", "lon")
    assert filtered.name == "msl_spectral_filtered"


def test_spectral_filter_invalid_shape() -> None:
    # 10 x 15 is invalid for SHT
    data: NDArray[np.float64] = np.random.rand(1, 10, 15)
    da = xr.DataArray(data, dims=["time", "lat", "lon"])

    filt = SpectralFilter()
    with pytest.raises(ValueError, match="Unsupported shape for spectral filter"):
        filt.filter(da, backend="serial")


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_lat_reverse(ny: int, nx: int) -> None:
    # Test latitude South to North
    data: NDArray[np.float64] = np.random.rand(1, ny, nx)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    filt = SpectralFilter(lmin=5, lmax=42, lat_reverse=True)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (1, ny, nx)
    assert filtered.lat[0] == -90


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_numpy_ndarray(ny: int, nx: int) -> None:
    # Test passing a raw numpy array
    data: NDArray[np.float64] = np.random.rand(ny, nx)

    filt = SpectralFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (ny, nx)


@pytest.mark.parametrize(("ny", "nx"), [(73, 144), (721, 1440)])
def test_spectral_filter_numpy_ndarray_3d(ny: int, nx: int) -> None:
    # Test passing a 3D numpy array (T, ny, nx)
    data: NDArray[np.float64] = np.random.rand(3, ny, nx)

    filt = SpectralFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (3, ny, nx)
