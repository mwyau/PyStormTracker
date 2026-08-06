from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from pystormtracker.models.tracker import Backend
from pystormtracker.models.tracks import Tracks
from pystormtracker.preprocessing.tracking import Projection
from pystormtracker.simple.tracker import SimpleTracker


@pytest.mark.integration
@pytest.mark.parametrize("map_proj", ["nh_stereo", "sh_stereo"])
def test_simple_stereographic_dask_matches_serial(
    map_proj: Literal["nh_stereo", "sh_stereo"], tmp_path: Path
) -> None:
    source = (
        Path(__file__).parents[1] / "data" / "era5" / "era5_msl_2025120100_2.5x2.5.nc"
    )
    field = xr.open_dataarray(source)

    times = np.array(
        ["2025-12-01T00", "2025-12-01T06"],
        dtype="datetime64[h]",
    )

    data = field.expand_dims(time=times)

    input_path = tmp_path / "projection_input.nc"
    data.to_dataset(name="msl").to_netcdf(input_path)

    def run(
        input_path: Path,
        *,
        map_proj: Projection,
        backend: Backend,
        n_workers: int | None = None,
    ) -> Tracks:
        tracker = SimpleTracker(
            projection=map_proj,
            stereo_grid_spacing_km=300.0,
            extent=(-3000.0, 3000.0, -3000.0, 3000.0),
            feature_point_method="quadratic",
            backend=backend,
            workers=n_workers,
        )
        return tracker.track(
            data=input_path,
            variable="msl",
            detection_mode="min",
        )

    serial = run(
        input_path,
        map_proj=map_proj,
        backend="serial",
    )
    dask = run(
        input_path,
        map_proj=map_proj,
        backend="dask",
        n_workers=2,
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
