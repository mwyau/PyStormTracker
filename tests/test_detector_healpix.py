from __future__ import annotations

import numpy as np
import xarray as xr

from pystormtracker.healpix.detector import HealpixDetector


def test_healpix_detector_init() -> None:
    # 1D xarray data
    nside = 4
    npix = 12 * nside**2
    data = np.ones((1, npix))
    da = xr.DataArray(
        data,
        dims=["time", "cell"],
        coords={"time": [0], "cell": np.arange(npix)},
        name="msl"
    )

    detector = HealpixDetector.from_xarray(da)
    assert detector.varname == "msl"
    assert detector._hp_base is not None
    assert detector._hp_base.nside() == nside
    assert detector._neighbor_table is not None
    assert detector._neighbor_table.shape == (8, npix)

def test_healpix_detector_detect() -> None:
    nside = 4
    npix = 12 * nside**2
    # Perturb background
    data = np.ones((1, npix)) * 1000.0 + np.arange(npix) * 0.01
    # Minimum at pixel 50
    data[0, 50] = 950.0

    da = xr.DataArray(
        data,
        dims=["time", "cell"],
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "cell": np.arange(npix),
        },
        name="msl"
    )

    detector = HealpixDetector.from_xarray(da)
    # Threshold 990.0 ensures background points are filtered out
    raw_results = detector.detect(threshold=990.0, minmaxmode="min")

    assert len(raw_results) == 1
    _time_val, lats_out, _lons_out, vars_dict = raw_results[0]

    assert len(lats_out) == 1
    assert vars_dict["msl"][0] == 950.0
