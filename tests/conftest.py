from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def reduced_gaussian_data() -> xr.DataArray:
    """Create a small deterministic reduced Gaussian field."""
    pl = np.array([4, 8, 12, 16, 16, 12, 8, 4], dtype=np.int32)
    values = np.sin(np.arange(int(pl.sum()), dtype=np.float64) / 7.0)
    return xr.DataArray(
        values[np.newaxis, :],
        dims=("time", "values"),
        coords={"time": [np.datetime64("2000-01-01")]},
        name="msl",
        attrs={"GRIB_gridType": "reduced_gg", "GRIB_pl": pl.tolist()},
    )
