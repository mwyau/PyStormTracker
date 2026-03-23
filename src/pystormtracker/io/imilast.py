from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..models.tracks import Tracks


def read_imilast(filename: Path | str) -> Tracks:
    """Loads tracks from an IMILAST format text file."""
    track_ids = []
    times = []
    lats = []
    lons = []
    vars_vals = []

    with open(filename) as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "90":
                continue
            elif parts[0] == "00":
                track_id = int(parts[1])
                lon, lat, var = float(parts[8]), float(parts[9]), float(parts[10])
                s_time = parts[3]
                if len(s_time) == 10:
                    dt = datetime(
                        int(s_time[:4]),
                        int(s_time[4:6]),
                        int(s_time[6:8]),
                        int(s_time[8:10]),
                        tzinfo=UTC,
                    )
                    time_val = np.datetime64(dt.replace(tzinfo=None), "s")
                else:
                    try:
                        dt = datetime.fromtimestamp(float(s_time), tz=UTC)
                        time_val = np.datetime64(dt.replace(tzinfo=None), "s")
                    except ValueError:
                        time_val = np.datetime64(s_time, "s")

                track_ids.append(track_id)
                times.append(time_val)
                lats.append(lat)
                lons.append(lon)
                vars_vals.append(var)

    obj = Tracks(
        track_ids=np.array(track_ids, dtype=np.int64),
        times=np.array(times, dtype="datetime64[s]"),
        lats=np.array(lats, dtype=np.float64),
        lons=np.array(lons, dtype=np.float64),
        vars_dict={"Intensity1": np.array(vars_vals, dtype=np.float64)},
    )
    obj.sort()
    return obj


def write_imilast(tracks: Tracks, outfile: str | Path, decimal_places: int = 6) -> None:
    """Exports tracks to an IMILAST format text file."""
    outfile_str = str(outfile)

    # Determine variable header based on track type
    var_header = "Intensity1"
    if tracks.track_type.lower() == "msl":
        var_header = "MSL"
    elif tracks.track_type.lower() == "vo":
        var_header = "VO"

    with open(outfile_str, "w", newline="") as f:
        header = (
            "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,LongE,LatN,"
            f"{var_header}\n"
        )
        f.write(header)

        t0 = np.datetime64("1970-01-01T00:00:00")

        for i, track in enumerate(tracks, start=1):
            f.write(f"90 {i} {len(track)}\n")
            for step, center in enumerate(track, start=1):
                try:
                    ts = (center.time - t0) / np.timedelta64(1, "s")
                    # Use integer timestamp for datetime to avoid float precision issues
                    dt = datetime.fromtimestamp(int(ts), tz=UTC)
                    yyyymmddhh = dt.strftime("%Y%m%d%H")
                    yyyy, mm, dd, hh = dt.year, dt.month, dt.day, dt.hour
                except Exception:
                    yyyymmddhh = "0000000000"
                    yyyy, mm, dd, hh = 0, 0, 0, 0

                # Ensure intensity is formatted with enough precision for VO
                val = next(iter(center.vars.values())) if center.vars else np.nan
                var_val = f"{float(val):.{decimal_places}f}"

                lon = center.lon
                if lon > 180:
                    lon -= 360

                f.write(
                    f"00 {i} {step} {yyyymmddhh} {yyyy} {mm:02d} "
                    f"{dd:02d} {hh:02d} {lon:.2f} {center.lat:.2f} "
                    f"{var_val}\n"
                )
