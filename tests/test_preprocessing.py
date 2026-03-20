from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.preprocessing import SphericalHarmonicFilter


def test_sh_filter_serial() -> None:
    # 73 x 144 (matches ERA5 2.5x2.5)
    data = np.random.rand(2, 73, 144)
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
    data = np.random.rand(1, 10, 10)
    da = xr.DataArray(data, dims=["time", "lat", "lon"])

    filt = SphericalHarmonicFilter()
    with pytest.raises(ValueError, match="Unsupported shape for SH filter"):
        filt.filter(da, backend="serial")


def test_sh_filter_dask() -> None:
    pytest.importorskip("dask")

    data = np.random.rand(4, 73, 144)
    da = xr.DataArray(data, dims=["time", "lat", "lon"]).chunk({"time": 2})

    filt = SphericalHarmonicFilter(lmin=5, lmax=42)
    filtered = filt.filter(da, backend="dask")

    # Check that it remains a dask array
    assert filtered.chunks is not None

    # Evaluate
    result = filtered.compute()
    assert result.shape == (4, 73, 144)
