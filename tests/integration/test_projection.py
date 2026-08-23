from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from pystormtracker.backends import Backend
from pystormtracker.models.geo import Projection
from pystormtracker.models.tracks import Tracks
from pystormtracker.simple.tracker import SimpleTracker
from tests.utils import get_integration_msl_path


@pytest.mark.integration
@pytest.mark.parametrize("projection", ["nh_stereo", "sh_stereo"])
def test_simple_stereographic_dask_matches_serial(
    projection: Literal["nh_stereo", "sh_stereo"], tmp_path: Path
) -> None:
    source = get_integration_msl_path()
    with xr.open_dataset(source, engine="h5netcdf") as dataset:
        data = dataset.msl.isel(valid_time=slice(0, 2)).rename({"valid_time": "time"})
        data = data.load()

    input_path = tmp_path / "projection_input.nc"
    data.to_dataset(name="msl").to_netcdf(input_path, engine="h5netcdf")

    def run(
        input_path: Path,
        *,
        projection: Projection,
        backend: Backend,
        n_workers: int | None = None,
    ) -> Tracks:
        tracker = SimpleTracker(
            projection=projection,
            stereo_grid_spacing_km=300.0,
            extent=(-3000.0, 3000.0, -3000.0, 3000.0),
            search_window_size=5,
            feature_refinement="quadratic",
            backend=backend,
            workers=n_workers,
        )
        return tracker.track(
            data=input_path,
            variable="msl",
            detection_mode="min",
            feature_threshold=0.0,
        )

    serial = run(
        input_path,
        projection=projection,
        backend="serial",
    )
    dask = run(
        input_path,
        projection=projection,
        backend="dask",
        n_workers=4,
    )

    assert serial.metadata == dask.metadata
    np.testing.assert_array_equal(serial.ids, dask.ids)
    np.testing.assert_array_equal(serial.offsets, dask.offsets)
    np.testing.assert_array_equal(serial.times, dask.times)
    np.testing.assert_allclose(serial.lats, dask.lats, atol=1e-6)
    np.testing.assert_allclose(serial.lons, dask.lons, atol=1e-6)
    np.testing.assert_allclose(
        serial.variables["msl"], dask.variables["msl"], atol=1e-6
    )
