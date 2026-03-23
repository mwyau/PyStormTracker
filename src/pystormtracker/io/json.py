import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..models.tracks import Tracks


def infer_track_type(tracks: Tracks) -> str:
    """
    Heuristic to infer track type (msl or vo) from track variables.
    """
    if tracks.track_type != "unknown":
        return tracks.track_type

    var_key = None
    if "msl" in tracks.vars:
        return "msl"
    elif "vo" in tracks.vars:
        return "vo"
    elif "Intensity1" in tracks.vars:
        var_key = "Intensity1"
    elif tracks.vars:
        var_key = next(iter(tracks.vars.keys()))

    if var_key:
        # Heuristic: MSL values are typically > 800 hPa
        # MSL anomalies can be very negative (e.g., -1000 to -10000 Pa)
        # VO values are typically very small (< 0.1)
        sample_val = np.nanmedian(tracks.vars[var_key])
        if sample_val > 500 or sample_val < -100:
            return "msl"
        else:
            return "vo"

    return "unknown"


def write_json(tracks: Tracks, outfile: str | Path) -> None:
    """
    Writes a Tracks object to the 'json' Hybrid Index format.
    This structure separates data into flat Struct-of-Arrays (SoA) for fast WebGL
    rendering, with NaN separators between tracks.
    """
    if len(tracks.track_ids) == 0:
        with open(outfile, "w") as f:
            json.dump({"points": {}, "tracks": [], "metadata": {}}, f)
        return

    # 1. Determine var_key and track_type
    var_key = None
    if "msl" in tracks.vars:
        var_key = "msl"
    elif "vo" in tracks.vars:
        var_key = "vo"
    elif "Intensity1" in tracks.vars:
        var_key = "Intensity1"
    elif tracks.vars:
        var_key = next(iter(tracks.vars.keys()))

    track_type = infer_track_type(tracks)
    tracks.track_type = track_type

    # 2. Scale MSL if in Pa (Heuristic: max abs > 500)
    raw_strength = tracks.vars[var_key] if var_key else np.zeros_like(tracks.lats)
    if track_type == "msl" and np.nanmax(np.abs(raw_strength)) > 500:
        raw_strength = raw_strength / 100.0

    # 3. Prepare SoA with NaN separators
    diff = np.diff(tracks.track_ids)
    boundaries = np.where(diff != 0)[0] + 1

    def inject_nans(
        arr: NDArray[np.float64], insert_val: float = np.nan
    ) -> NDArray[np.float64]:
        return np.insert(arr, boundaries, insert_val)

    points_lat = inject_nans(tracks.lats).tolist()
    points_lon = inject_nans(tracks.lons).tolist()

    timestamps = tracks.times.astype("datetime64[ms]").astype(np.int64)
    points_time = np.insert(timestamps.astype(float), boundaries, np.nan)
    points_time_list = [None if np.isnan(x) else int(x) for x in points_time]

    points_strength = inject_nans(raw_strength)
    points_strength_list = [None if np.isnan(x) else float(x) for x in points_strength]

    points_lat_list = [None if np.isnan(x) else float(x) for x in points_lat]
    points_lon_list = [None if np.isnan(x) else float(x) for x in points_lon]

    # 4. Generate metadata
    tracks_meta = []
    u_ids = tracks.unique_track_ids

    start_indices = np.concatenate(([0], boundaries))
    end_indices = np.concatenate((boundaries, [len(tracks.track_ids)]))

    offsets = np.arange(len(start_indices))
    new_start_indices = start_indices + offsets
    new_end_indices = end_indices + offsets - 1

    min_time = float("inf")
    max_time = float("-inf")
    min_strength = float("inf")
    max_strength = float("-inf")

    for i, tid in enumerate(u_ids):
        orig_s = start_indices[i]
        orig_e = end_indices[i]

        t_times = timestamps[orig_s:orig_e]
        t_lats = tracks.lats[orig_s:orig_e]
        t_lons = tracks.lons[orig_s:orig_e]
        t_vals = raw_strength[orig_s:orig_e]

        if track_type == "msl":
            t_strength = float(np.min(t_vals))
        else:
            t_strength = float(np.max(t_vals))

        min_strength = min(min_strength, float(np.min(t_vals)))
        max_strength = max(max_strength, float(np.max(t_vals)))
        min_time = min(min_time, int(t_times[0]))
        max_time = max(max_time, int(t_times[-1]))

        duration_hrs = float((t_times[-1] - t_times[0]) / (3600 * 1000))
        from ..utils.geo import geod_dist_km

        disp = float(geod_dist_km(t_lats[0], t_lons[0], t_lats[-1], t_lons[-1]))

        tracks_meta.append(
            {
                "track_id": int(tid),
                "start": int(new_start_indices[i]),
                "end": int(new_end_indices[i]),
                "strength": t_strength,
                "duration": duration_hrs,
                "displacement": disp,
            }
        )

    metadata = {
        "track_type": track_type,
        "min_time": int(min_time) if min_time != float("inf") else 0,
        "max_time": int(max_time) if max_time != float("-inf") else 0,
        "min_strength": float(min_strength) if min_strength != float("inf") else 0.0,
        "max_strength": float(max_strength) if max_strength != float("-inf") else 0.0,
        "max_duration": float(max(t["duration"] for t in tracks_meta))
        if tracks_meta
        else 0.0,
        "max_displacement": float(max(t["displacement"] for t in tracks_meta))
        if tracks_meta
        else 0.0,
    }

    json_data = {
        "metadata": metadata,
        "points": {
            "lat": points_lat_list,
            "lon": points_lon_list,
            "time": points_time_list,
            "strength": points_strength_list,
        },
        "tracks": tracks_meta,
    }

    with open(outfile, "w") as f:
        json.dump(json_data, f, separators=(",", ":"))


def read_json(infile: str | Path) -> Tracks:
    """
    Reads a 'json' format file back into a Tracks object.
    Note: If MSL was scaled to hPa during write, it remains hPa here.
    """
    with open(infile) as f:
        data = json.load(f)

    points = data.get("points", {})
    tracks_meta = data.get("tracks", [])

    if not points or not tracks_meta:
        return Tracks()

    lats_raw = np.array(points.get("lat", []), dtype=float)
    lons_raw = np.array(points.get("lon", []), dtype=float)
    times_raw = np.array(points.get("time", []), dtype=float)
    strength_raw = np.array(points.get("strength", []), dtype=float)

    valid_mask = ~np.isnan(lats_raw)

    lats = lats_raw[valid_mask]
    lons = lons_raw[valid_mask]
    times = (times_raw[valid_mask]).astype("datetime64[ms]").astype("datetime64[s]")
    strength = strength_raw[valid_mask]

    track_ids = []
    for t in tracks_meta:
        length = t["end"] - t["start"] + 1
        track_ids.extend([t["track_id"]] * length)

    track_ids_arr = np.array(track_ids, dtype=np.int64)

    metadata = data.get("metadata", {})
    track_type = metadata.get("track_type", "unknown")
    var_key = (
        "msl" if track_type == "msl" else ("vo" if track_type == "vo" else "intensity")
    )

    vars_dict = {var_key: strength} if len(strength) > 0 else {}

    return Tracks(
        track_ids=track_ids_arr,
        times=times,
        lats=lats,
        lons=lons,
        vars_dict=vars_dict,
        track_type=track_type,
    )
