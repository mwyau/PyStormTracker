from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

# Routine MPI integration tests intentionally launch four local ranks. Allow
# Open MPI / PRRTE to run them on CI hosts that expose fewer processor slots.
os.environ.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
os.environ.setdefault("PRTE_MCA_mapby", ":oversubscribe")


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
