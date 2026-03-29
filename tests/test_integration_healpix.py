from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pystormtracker.healpix.tracker import HealpixTracker
from pystormtracker.preprocessing.regrid import SpectralRegridder


@pytest.mark.integration
def test_healpix_tracker_serial_integration() -> None:
    # 1. Create a 2D grid (2.5 deg) with a known local minimum
    ny, nx = 73, 144
    time = np.arange(3)
    data = np.zeros((len(time), ny, nx), dtype=np.float64)
    # Background
    data[:] = 1013.0

    # Track coordinates over time
    expected_lats = [45.0, 47.5, 50.0]
    expected_lons = [10.0, 12.5, 15.0]

    for t in range(len(time)):
        # Calculate grid indices nearest to expected lat/lon
        lat_vals = np.linspace(-90, 90, ny)
        lon_vals = np.linspace(0, 360, nx, endpoint=False)
        lat_idx = np.argmin(np.abs(lat_vals - expected_lats[t]))
        lon_idx = np.argmin(np.abs(lon_vals - expected_lons[t]))
        # Simple dip to create an extremum
        data[t, lat_idx, lon_idx] = 980.0
        # Add some surrounding points to make it a local minimum
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                data[t, lat_idx + i, (lon_idx + j) % nx] = 1000.0

    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": time,
            "lat": np.linspace(-90, 90, ny),
            "lon": np.linspace(0, 360, nx, endpoint=False),
        },
        name="msl",
    )

    # 2. Regrid to HEALPix
    regridder = SpectralRegridder()
    nside = 16
    # Regrid per frame
    hp_frames = []
    for t in range(len(time)):
        hp_frames.append(regridder.to_healpix(da.isel(time=t), nside=nside, lat_reverse=True))

    da_hp = xr.concat(hp_frames, dim="time")
    da_hp.coords["time"] = time

    # Save to dummy file to simulate Tracker protocol input
    import os
    import uuid
    tmp_file = f"/tmp/test_hp_{uuid.uuid4().hex}.nc"
    da_hp.to_netcdf(tmp_file)

    try:
        # 3. Track on HEALPix
        tracker = HealpixTracker()
        tracks = tracker.track(
            infile=tmp_file,
            varname="msl",
            mode="min",
            threshold=1000.0,
        )

        # 4. Verify results
        assert len(tracks) == 1
        track = tracks[0]
        assert len(track) == 3

        # Verify coordinates (with some tolerance due to regridding and HEALPix resolution)
        # Nside=16 pixel resolution is ~3.7 degrees.
        for t in range(len(time)):
            np.testing.assert_allclose(track[t].lat, expected_lats[t], atol=5.0)
            np.testing.assert_allclose(track[t].lon, expected_lons[t], atol=5.0)

    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
