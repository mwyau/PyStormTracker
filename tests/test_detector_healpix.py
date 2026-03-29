from __future__ import annotations

import ducc0
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
        name="msl",
    )

    detector = HealpixDetector.from_xarray(da)
    assert detector.varname == "msl"
    assert detector._hp_base is not None
    assert detector._hp_base.nside() == nside
    assert detector._neighbor_table is not None
    assert detector._neighbor_table.shape == (8, npix)


def test_healpix_detector_detect() -> None:
    nside = 16
    npix = 12 * nside**2

    hp_base = ducc0.healpix.Healpix_Base(nside, "RING")
    all_pix = np.arange(npix, dtype=np.int64)
    ang = hp_base.pix2ang(all_pix)
    lats = 90.0 - np.degrees(ang[:, 0])
    lons = np.degrees(ang[:, 1])

    # Create a smooth minimum at pixel 100
    p0 = 100
    lat0_rad = np.radians(lats[p0])
    lon0_rad = np.radians(lons[p0])
    data_1d = np.ones(npix) * 1013.0
    for i in range(npix):
        lat_i = np.radians(lats[i])
        lon_i = np.radians(lons[i])
        dlat = lat_i - lat0_rad
        dlon = (lon_i - lon0_rad) * np.cos(lat0_rad)
        dist_sq = dlat**2 + dlon**2
        if dist_sq < 0.01:
            data_1d[i] = 980.0 + 1000.0 * dist_sq

    data = data_1d.reshape(1, npix)

    da = xr.DataArray(
        data,
        dims=["time", "cell"],
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "cell": np.arange(npix),
        },
        name="msl",
    )

    detector = HealpixDetector.from_xarray(da)
    # Threshold 1000.0
    raw_results = detector.detect(threshold=1000.0, minmaxmode="min", min_points=1)

    assert len(raw_results) == 1
    _time_val, lats_out, _lons_out, vars_dict = raw_results[0]
    assert len(lats_out) >= 1
    # Check if the true minimum was found (among others if any)
    min_val = np.min(vars_dict["msl"])
    assert min_val < 985.0
