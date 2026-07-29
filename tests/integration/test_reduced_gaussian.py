from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pystormtracker.hodges.tracker import HodgesTracker
from pystormtracker.io.data_loader import DataLoader
from pystormtracker.preprocessing.regrid import SpectralRegridder
from pystormtracker.preprocessing.spectral import apply_sht_filter


@pytest.fixture
def n320_msl_path() -> str:
    path = "/home/albert/PyStormTracker-Data/era5_msl_2025-2026_djf_n320.grib"
    if not Path(path).exists():
        pytest.skip(f"Local test data {path} not found")
    return path


@pytest.mark.integration
def test_reduced_gaussian_loader(n320_msl_path: str) -> None:
    loader = DataLoader(n320_msl_path)
    loader.ensure_open()

    assert loader.is_reduced_gaussian("msl")
    pl = loader.get_reduced_grid_pl("msl")
    assert pl is not None
    assert len(pl) == 640
    assert np.sum(pl) == 542080


@pytest.mark.integration
def test_reduced_gaussian_filter_to_cc(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    # Process only first 2 time steps to keep it fast
    data = ds.msl.isel(time=slice(0, 2))

    # Filter and regrid to a 1 degree Lat-Lon grid (181x360)
    filtered = apply_sht_filter(
        data, lmin=5, lmax=42, out_geometry="CC", out_ntheta=181, out_nphi=360
    )

    assert filtered.dims == ("time", "latitude", "longitude")
    assert filtered.shape == (2, 181, 360)
    assert not np.isnan(filtered.values).any()


@pytest.mark.integration
def test_reduced_gaussian_filter_to_gl(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    data = ds.msl.isel(time=0)

    # Filter and regrid to a regular N80 Gaussian grid (160x320)
    filtered = apply_sht_filter(
        data, lmin=0, lmax=80, out_geometry="GL", out_ntheta=160, out_nphi=320
    )

    assert filtered.dims == ("latitude", "longitude")
    assert filtered.shape == (160, 320)


@pytest.mark.integration
def test_reduced_gaussian_regridder(n320_msl_path: str) -> None:
    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    data = ds.msl.isel(time=0)

    regridder = SpectralRegridder(lmax=80)

    # Regrid to HEALPix nside=32
    hp_data = regridder.to_healpix(
        data,
        nside=32,
        in_geometry="GL",  # Reduced grids use GL theta values
    )

    assert hp_data.dims == ("cell",)
    assert hp_data.size == 12 * 32**2


@pytest.mark.integration
def test_reduced_gaussian_tracking_pipeline(n320_msl_path: str, tmp_path: Path) -> None:
    """End-to-end test tracking on reduced Gaussian input."""
    tracker = HodgesTracker(min_lifetime=3)

    # We must regrid during tracking for reduced grids.
    # Currently, tracker.track expects 2D structured data or does its own filtering.
    # Let's ensure it handles it.
    # Actually, we need to update tracker.track to handle this.
    # For now, let's just verify the components.

    ds = xr.open_dataset(n320_msl_path, engine="cfgrib")
    # Take a small slice for speed
    data = ds.msl.isel(time=slice(0, 10))

    # 1. Manually filter and regrid to CC
    data_filtered = apply_sht_filter(
        data, lmin=5, lmax=42, out_geometry="CC", out_ntheta=181, out_nphi=360
    )

    # 2. Track on the regridded data
    tracks = tracker.track(
        infile=data_filtered,
        varname="msl",
        mode="min",
        threshold=101000.0,  # Pa
        filter=False,  # Already filtered
    )

    assert len(tracks) > 0
    assert any(len(t) >= 3 for t in tracks)
