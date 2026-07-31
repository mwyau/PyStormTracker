from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pystormtracker.simple.tracker import SimpleTracker


@pytest.mark.integration
@pytest.mark.parametrize("map_proj", ["nh_stereo", "sh_stereo"])
def test_simple_stereographic_dask_matches_serial(
    map_proj: str, tmp_path: Path
) -> None:
    source = (
        Path(__file__).parents[1] / "data" / "era5" / "era5_msl_2025120100_2.5x2.5.nc"
    )
    field = xr.open_dataarray(source)
    data = xr.concat(
        [field, field],
        dim=xr.IndexVariable(
            "time",
            np.array(["2025-12-01T00", "2025-12-01T06"], dtype="datetime64[h]"),
        ),
    )
    input_path = tmp_path / "projection_input.nc"
    data.to_dataset(name="msl").to_netcdf(input_path)

    kwargs = {
        "mode": "min",
        "map_proj": map_proj,
        "extent": (-3000.0, 3000.0, -3000.0, 3000.0),
        "resolution": 300.0,
        "filter": False,
        "lmax": 7,
        "subgrid_refine": True,
    }
    tracker = SimpleTracker()
    serial = tracker.track(input_path, "msl", backend="serial", **kwargs)  # type: ignore[arg-type]
    dask = tracker.track(
        input_path,
        "msl",
        backend="dask",
        n_workers=2,
        **kwargs,  # type: ignore[arg-type]
    )

    assert serial == dask
    if serial.lats.size:
        if map_proj == "nh_stereo":
            assert np.all(serial.lats > 0.0)
        else:
            assert np.all(serial.lats < 0.0)
