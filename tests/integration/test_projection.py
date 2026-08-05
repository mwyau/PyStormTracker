from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from pystormtracker.models.tracks import Tracks
from pystormtracker.preprocessing.tracking import Projection
from pystormtracker.simple.tracker import SimpleTracker
from pystormtracker.track import Backend


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
        tracker: SimpleTracker,
        input_path: Path,
        *,
        map_proj: Projection,
        backend: Backend,
        n_workers: int | None = None,
    ) -> Tracks:
        return tracker.track(
            input_path,
            "msl",
            backend=backend,
            n_workers=n_workers,
            mode="min",
            map_proj=map_proj,
            extent=(-3000.0, 3000.0, -3000.0, 3000.0),
            resolution=300.0,
            subgrid_refine=True,
        )

    tracker = SimpleTracker()

    serial = run(
        tracker,
        input_path,
        map_proj=map_proj,
        backend="serial",
    )
    dask = run(
        tracker,
        input_path,
        map_proj=map_proj,
        backend="dask",
        n_workers=2,
    )

    assert serial == dask
    if serial.lats.size:
        if map_proj == "nh_stereo":
            assert np.all(serial.lats > 0.0)
        else:
            assert np.all(serial.lats < 0.0)
