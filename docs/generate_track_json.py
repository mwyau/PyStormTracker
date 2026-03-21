import os
import json
import numpy as np
import pandas as pd
from pystormtracker.io.imilast import read_imilast
from pystormtracker.utils.geo import geod_dist_km

def main():
    # Load track data
    track_file = "../data/test/tracks/era5_msl_2025-2026_djf_2.5x2.5_hodges_imilast.txt"
    # Resolve relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_file_path = os.path.normpath(os.path.join(script_dir, track_file))
    
    print(f"Reading tracks from {track_file_path}...")
    tracks = read_imilast(track_file_path)

    track_type = 'vo' if 'vo' in os.path.basename(track_file_path).lower() else 'msl'
    
    raw_intensity = tracks.vars.get('Intensity1', np.zeros_like(tracks.lats))

    data = {
        'track_id': tracks.track_ids,
        'time': tracks.times,
        'lat': tracks.lats,
        'lon': tracks.lons,
        'intensity': raw_intensity
    }
    df = pd.DataFrame(data)

    if track_type == 'vo':
        df['display_intensity'] = df['intensity'] * 1e5
    else:
        df['display_intensity'] = df['intensity'] / 100.0

    df = df.sort_values(['track_id', 'time'])

    grouped = df.groupby('track_id')

    if track_type == 'vo':
        track_strength = grouped['display_intensity'].max().to_dict()
    else:
        track_strength = grouped['display_intensity'].min().to_dict()
    
    df['track_strength'] = df['track_id'].map(track_strength)

    track_durations = (grouped['time'].max() - grouped['time'].min()) / np.timedelta64(1, 'h')
    df['track_duration_hrs'] = df['track_id'].map(track_durations.to_dict())

    def calc_displacement(g):
        return geod_dist_km(g.iloc[0]['lat'], g.iloc[0]['lon'], g.iloc[-1]['lat'], g.iloc[-1]['lon'])
    
    track_displacements = grouped.apply(calc_displacement)
    df['track_displacement_km'] = df['track_id'].map(track_displacements.to_dict())

    print("Formatting data for JSON...")
    
    json_data = {
        "metadata": {
            "track_type": track_type,
            "min_time": int(df['time'].min().timestamp() * 1000),
            "max_time": int(df['time'].max().timestamp() * 1000),
            "min_strength": float(df['display_intensity'].min()),
            "max_strength": float(df['display_intensity'].max()),
            "max_duration": float(df['track_duration_hrs'].max()),
            "max_displacement": float(df['track_displacement_km'].max())
        },
        "tracks": []
    }

    for track_id, group in grouped:
        if len(group) > 1:
            track_obj = {
                "id": int(track_id),
                "lats": [float(x) for x in group['lat'].values],
                "lons": [float(x) for x in group['lon'].values],
                "times": [int(x.timestamp() * 1000) for x in group['time']],
                "intensities": [float(x) for x in group['display_intensity'].values],
                "strength": float(group['track_strength'].iloc[0]),
                "duration": float(group['track_duration_hrs'].iloc[0]),
                "displacement": float(group['track_displacement_km'].iloc[0])
            }
            json_data["tracks"].append(track_obj)

    static_dir = os.path.join(script_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    out_file = os.path.join(static_dir, "tracks_data.js")
    
    print(f"Writing to {out_file}...")
    with open(out_file, "w") as f:
        f.write("window.TRACKS_DATA = ")
        json.dump(json_data, f, separators=(',', ':'))
        f.write(";")

    print("Done!")

if __name__ == "__main__":
    main()
