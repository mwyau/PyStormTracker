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
    track_type = "unknown"
    var_key = "Intensity1"

    with open(filename) as f:
        header = f.readline()
        # Standard IMILAST header uses commas
        if "," in header:
            parts_h = [p.strip() for p in header.split(",")]
            if len(parts_h) > 10:
                var_key = parts_h[10]

        # Determine track type from variable key or header content
        if "MSL" in var_key.upper() or "MSL" in header.upper():
            track_type = "msl"
        elif "VO" in var_key.upper() or "VO" in header.upper():
            track_type = "vo"

        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "90":
                continue
            elif parts[0] == "00":
                track_id = int(parts[1])
                lon, lat, var = float(parts[8]), float(parts[9]), float(parts[10])

                # Unit conversion back to SI
                # Heuristic: only scale if values look like they are in hPa or
                # 10^-5 units to avoid double-scaling legacy files.
                if track_type == "msl" and abs(var) < 5000.0:
                    # Likely hPa (e.g., 1013 vs 101325)
                    var *= 100.0  # hPa -> Pa
                elif track_type == "vo" and abs(var) > 0.1:
                    # Likely 10^-5 units (e.g., 10 vs 1e-4)
                    var *= 1e-5  # 10^-5 s^-1 -> s^-1

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
        vars_dict={var_key: np.array(vars_vals, dtype=np.float64)},
        track_type=track_type,
    )
    obj.sort()
    return obj


def write_imilast(tracks: Tracks, outfile: str | Path, decimal_places: int = 6) -> None:
    """Exports tracks to an IMILAST format text file."""
    outfile_str = str(outfile)

    # Determine variable header name
    if tracks.track_type != "unknown":
        var_header = tracks.track_type.upper()
    else:
        var_names = list(tracks.vars.keys())
        var_header = var_names[0] if var_names else "Intensity1"

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
                # Use the key corresponding to var_header if it exists in vars
                val = center.vars.get(var_header.lower(), np.nan)
                if np.isnan(val):
                    val = next(iter(center.vars.values())) if center.vars else np.nan

                # Unit conversion for standard variables
                if tracks.track_type.lower() == "msl":
                    val *= 0.01  # Pa -> hPa
                elif tracks.track_type.lower() == "vo":
                    val *= 1e5  # s^-1 -> 10^-5 s^-1

                var_val = f"{float(val):.{decimal_places}f}"

                lon = center.lon
                if lon > 180:
                    lon -= 360

                f.write(
                    f"00 {i} {step} {yyyymmddhh} {yyyy} {mm:02d} "
                    f"{dd:02d} {hh:02d} {lon:.2f} {center.lat:.2f} "
                    f"{var_val}\n"
                )
