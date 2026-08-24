#!/usr/bin/env python3
"""Verify ERA5 MSLP input geometry and cadence for TRACK comparison runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

CASES = {
    "f320-2024": {
        "frames": 1464,
        "nlon": 1280,
        "nlat": 640,
        "start": "2024-01-01T00:00:00",
        "end": "2024-12-31T18:00:00",
        "regular_spacing": None,
    },
    "f320-jan2024": {
        "frames": 124,
        "nlon": 1280,
        "nlat": 640,
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-31T18:00:00",
        "regular_spacing": None,
    },
    "2p5-dec2025": {
        "frames": 124,
        "nlon": 144,
        "nlat": 73,
        "start": "2025-12-01T00:00:00",
        "end": "2025-12-31T18:00:00",
        "regular_spacing": 2.5,
    },
    "2p5-djf2025-2026": {
        "frames": 360,
        "nlon": 144,
        "nlat": 73,
        "start": "2025-12-01T00:00:00",
        "end": "2026-02-28T18:00:00",
        "regular_spacing": 2.5,
    },
    "0p25-dec2025": {
        "frames": 124,
        "nlon": 1440,
        "nlat": 721,
        "start": "2025-12-01T00:00:00",
        "end": "2025-12-31T18:00:00",
        "regular_spacing": 0.25,
    },
    "0p25-djf2025-2026": {
        "frames": 360,
        "nlon": 1440,
        "nlat": 721,
        "start": "2025-12-01T00:00:00",
        "end": "2026-02-28T18:00:00",
        "regular_spacing": 0.25,
    },
}


def coord_name(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise SystemExit(f"None of {names!r} found; dimensions={dict(ds.sizes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    expected = CASES[args.case]
    with xr.open_dataset(args.path) as ds:
        time = coord_name(ds, ("time", "valid_time"))
        lat = coord_name(ds, ("latitude", "lat"))
        lon = coord_name(ds, ("longitude", "lon"))

        sizes = ds.sizes
        actual = (sizes[time], sizes[lon], sizes[lat])
        wanted = (expected["frames"], expected["nlon"], expected["nlat"])
        if actual != wanted:
            raise SystemExit(
                f"size mismatch: got time/lon/lat={actual}, expected={wanted}"
            )

        times = np.asarray(ds[time].values).astype("datetime64[s]")
        start = np.datetime64(expected["start"])
        end = np.datetime64(expected["end"])
        if times[0] != start or times[-1] != end:
            raise SystemExit(
                f"time range mismatch: got {times[0]} .. {times[-1]}, "
                f"expected {start} .. {end}"
            )
        if len(times) > 1 and not np.all(np.diff(times) == np.timedelta64(6, "h")):
            raise SystemExit("time cadence is not exactly 6 hours")

        spacing = expected["regular_spacing"]
        if spacing is not None:
            lats = np.asarray(ds[lat].values, dtype=float)
            lons = np.asarray(ds[lon].values, dtype=float)
            if not np.allclose(np.abs(np.diff(lats)), spacing):
                raise SystemExit(f"latitude spacing is not {spacing} degrees")
            if not np.allclose(np.mod(np.diff(lons), 360.0), spacing):
                raise SystemExit(f"longitude spacing is not {spacing} degrees")

        variable = "msl" if "msl" in ds.data_vars else None
        if variable is None and "mean_sea_level_pressure" in ds.data_vars:
            variable = "mean_sea_level_pressure"
        if variable is None:
            raise SystemExit(f"MSLP variable not found; variables={list(ds.data_vars)}")

        units = str(ds[variable].attrs.get("units", ""))
        if units and units.lower() not in {"pa", "pascal", "pascals"}:
            raise SystemExit(f"unexpected MSLP units: {units!r}")

        print(f"OK case={args.case}")
        print(f"path={args.path.resolve()}")
        print(f"variable={variable}")
        print(f"dimensions=time:{sizes[time]} lon:{sizes[lon]} lat:{sizes[lat]}")
        print(f"time={times[0]} .. {times[-1]} cadence=6h")
        print(f"units={units or '<not declared>'}")


if __name__ == "__main__":
    main()
