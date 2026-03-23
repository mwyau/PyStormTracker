from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pystormtracker.preprocessing import SphericalHarmonicFilter


@pytest.fixture(autouse=True)
def skip_if_no_shtns() -> None:
    pytest.importorskip("shtns")


def test_sh_filter_engines_compare() -> None:
    # Use a smooth field rather than random noise to avoid high-frequency
    # aliasing differences between the two transform implementations.
    nlat, nlon = 73, 144
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)[:, None]
    lon = np.linspace(0, 2 * np.pi, nlon)[None, :]
    data: NDArray[np.float64] = np.cos(5 * lon) * np.sin(lat) ** 5 * np.cos(lat) ** 5

    # Use a more conservative lmax (30) for the comparison test to minimize
    # the impact of grid-sampling differences (DH vs Regular-with-Poles)
    # between the two backend implementations.
    filt_shtools = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtools")
    filt_ducc0 = SphericalHarmonicFilter(lmin=5, lmax=30, engine="ducc0")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtns")

    filtered_shtools = filt_shtools.filter(data)
    filtered_ducc0 = filt_ducc0.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # 2 orders of magnitude back from ~1e-10 limit
    np.testing.assert_allclose(filtered_shtools, filtered_shtns, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(filtered_ducc0, filtered_shtns, rtol=1e-8, atol=1e-8)


def test_sh_filter_engines_compare_gaussian_blob() -> None:
    nlat, nlon = 73, 144
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Gaussian blob at 45N, 180E
    sigma = 10.0
    data = np.exp(-((lat_grid - 45) ** 2 + (lon_grid - 180) ** 2) / (2 * sigma**2))

    filt_shtools = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtools")
    filt_ducc0 = SphericalHarmonicFilter(lmin=5, lmax=30, engine="ducc0")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=30, engine="shtns")

    filtered_shtools = filt_shtools.filter(data)
    filtered_ducc0 = filt_ducc0.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # ~1.5 orders of magnitude back from ~1.4e-7 abs and ~3e-3 rel limits
    np.testing.assert_allclose(filtered_shtools, filtered_shtns, rtol=5e-2, atol=1e-5)
    np.testing.assert_allclose(filtered_ducc0, filtered_shtns, rtol=5e-2, atol=1e-5)


def test_sh_filter_engines_compare_random() -> None:
    # Random noise test (harder to match exactly due to aliasing)
    np.random.seed(42)
    data = np.random.rand(73, 144)

    filt_shtools = SphericalHarmonicFilter(lmin=5, lmax=20, engine="shtools")
    filt_ducc0 = SphericalHarmonicFilter(lmin=5, lmax=20, engine="ducc0")
    filt_shtns = SphericalHarmonicFilter(lmin=5, lmax=20, engine="shtns")

    filtered_shtools = filt_shtools.filter(data)
    filtered_ducc0 = filt_ducc0.filter(data)
    filtered_shtns = filt_shtns.filter(data)

    # Use larger tolerance for random noise (limit was ~5e-3 abs)
    np.testing.assert_allclose(filtered_shtools, filtered_shtns, rtol=1e-1, atol=5e-2)
    np.testing.assert_allclose(filtered_ducc0, filtered_shtns, rtol=1e-1, atol=5e-2)
