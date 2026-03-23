from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from pystormtracker.preprocessing import SphericalHarmonicFilter


def test_sh_filter_serial() -> None:
    # 73 x 144 (matches ERA5 2.5x2.5)
    data: NDArray[np.float64] = np.random.rand(2, 73, 144)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0, 1],
            "lat": np.linspace(90, -90, 73),
            "lon": np.linspace(0, 360, 144, endpoint=False),
        },
        name="msl",
    )

    filt = SphericalHarmonicFilter(lmin=5, lmax=42)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (2, 73, 144)
    assert filtered.dims == ("time", "lat", "lon")
    assert filtered.name == "msl_sh_filtered"


def test_sh_filter_invalid_shape() -> None:
    # 10 x 15 is invalid because nlon (15) != 10, 20, or 19.
    data: NDArray[np.float64] = np.random.rand(1, 10, 15)
    da = xr.DataArray(data, dims=["time", "lat", "lon"])

    filt = SphericalHarmonicFilter()
    with pytest.raises(ValueError, match="Unsupported shape for SH filter"):
        filt.filter(da, backend="serial")


def test_sh_filter_lat_reverse() -> None:
    # 73 x 144, but latitude from -90 to 90 (reversed)
    data: NDArray[np.float64] = np.random.rand(1, 73, 144)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": [0],
            "lat": np.linspace(-90, 90, 73),
            "lon": np.linspace(0, 360, 144, endpoint=False),
        },
        name="msl",
    )

    filt = SphericalHarmonicFilter(lmin=5, lmax=42, lat_reverse=True)
    filtered = filt.filter(da, backend="serial")

    assert filtered.shape == (1, 73, 144)
    assert filtered.lat[0] == -90


def test_sh_filter_numpy_ndarray() -> None:
    # Test passing a raw numpy array (must be 73x144 for 2.5 degree)
    data: NDArray[np.float64] = np.random.rand(73, 144)

    filt = SphericalHarmonicFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (73, 144)


def test_sh_filter_numpy_ndarray_3d() -> None:
    # Test passing a 3D numpy array (T, 73, 144)
    data: NDArray[np.float64] = np.random.rand(3, 73, 144)

    filt = SphericalHarmonicFilter(lmin=5, lmax=42)
    filtered = filt.filter(data)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == (3, 73, 144)


def test_sh_filter_engines_compare() -> None:
    pytest.importorskip("shtns")

    # Use a smooth field rather than random noise to avoid high-frequency
    # aliasing differences between the two transform implementations.
    nlat, nlon = 73, 144
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)[:, None]
    lon = np.linspace(0, 2 * np.pi, nlon)[None, :]
    data: NDArray[np.float64] = np.cos(5 * lon) * np.sin(lat) ** 5 * np.cos(lat) ** 5

    # Use a more conservative lmax (30) for the comparison test to minimize
    # the impact of grid-sampling differences (DH vs Regular-with-Poles)
    # between the two backend implementations.
    filt_pyshtools = SphericalHarmonicFilter(lmin=5, lmax=30, engine="pyshtools")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtns")

    filtered_pysh = filt_pyshtools.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # 2 orders of magnitude back from ~1e-10 limit
    np.testing.assert_allclose(filtered_pysh, filtered_shtns, rtol=1e-8, atol=1e-8)


def test_sh_filter_engines_compare_gaussian_blob() -> None:
    pytest.importorskip("shtns")

    nlat, nlon = 73, 144
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Gaussian blob at 45N, 180E
    sigma = 10.0
    data = np.exp(-((lat_grid - 45) ** 2 + (lon_grid - 180) ** 2) / (2 * sigma**2))

    filt_pyshtools = SphericalHarmonicFilter(lmin=5, lmax=30, engine="pyshtools")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtns")

    filtered_pysh = filt_pyshtools.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # ~1.5 orders of magnitude back from ~1.4e-7 abs and ~3e-3 rel limits
    np.testing.assert_allclose(filtered_pysh, filtered_shtns, rtol=5e-2, atol=1e-5)


def test_sh_filter_engines_compare_random() -> None:
    pytest.importorskip("shtns")

    # Random noise test (harder to match exactly due to aliasing)
    np.random.seed(42)
    data = np.random.rand(73, 144)

    filt_pyshtools = SphericalHarmonicFilter(lmin=5, lmax=20, engine="pyshtools")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=20, engine="shtns")

    filtered_pysh = filt_pyshtools.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # Use larger tolerance for random noise (limit was ~5e-3 abs)
    np.testing.assert_allclose(filtered_pysh, filtered_shtns, rtol=1e-1, atol=5e-2)
