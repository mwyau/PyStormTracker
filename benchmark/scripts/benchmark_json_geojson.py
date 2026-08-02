"""Empirical multi-scale I/O benchmark comparing TrackJSON v1.0 and GeoJSON."""

from __future__ import annotations

import gzip
import json
import time

import numpy as np


def generate_mock_tracks(
    n_tracks: int = 1000, points_per_track: int = 30
) -> tuple[dict[str, object], dict[str, object]]:
    """Generate synthetic trajectory dataset in both TrackJSON and GeoJSON formats."""
    np.random.seed(42)
    tracks_list: list[dict[str, object]] = []

    lats_flat: list[float | None] = []
    lons_flat: list[float | None] = []
    times_flat: list[int | None] = []
    msl_flat: list[float | None] = []
    vo_flat: list[float | None] = []

    tracks_meta: list[dict[str, object]] = []
    current_idx = 0
    start_time_base = 1577836800000

    min_lat, max_lat = float("inf"), float("-inf")
    min_lon, max_lon = float("inf"), float("-inf")
    min_time, max_time = float("inf"), float("-inf")

    for tid in range(1, n_tracks + 1):
        start_lat = float(np.random.uniform(-60, 60))
        start_lon = float(np.random.uniform(-180, 180))

        t_lats = (
            start_lat + np.cumsum(np.random.normal(0.2, 0.5, points_per_track))
        ).tolist()
        t_lons = (
            start_lon + np.cumsum(np.random.normal(0.5, 0.5, points_per_track))
        ).tolist()
        t_times = (start_time_base + np.arange(points_per_track) * 21600000).tolist()
        t_msl = (
            101000.0
            - np.abs(np.sin(np.linspace(0, np.pi, points_per_track))) * 3500.0
            + np.random.normal(0, 100, points_per_track)
        ).tolist()
        t_vo = (
            1.0e-4
            + np.abs(np.sin(np.linspace(0, np.pi, points_per_track))) * 3.0e-4
            + np.random.normal(0, 1.0e-5, points_per_track)
        ).tolist()

        start_pos = current_idx
        end_pos = current_idx + points_per_track - 1

        lats_flat.extend(t_lats)
        lats_flat.append(None)

        lons_flat.extend(t_lons)
        lons_flat.append(None)

        times_flat.extend(t_times)
        times_flat.append(None)

        msl_flat.extend(t_msl)
        msl_flat.append(None)

        vo_flat.extend(t_vo)
        vo_flat.append(None)

        min_lat = min(min_lat, min(t_lats))
        max_lat = max(max_lat, max(t_lats))
        min_lon = min(min_lon, min(t_lons))
        max_lon = max(max_lon, max(t_lons))
        min_time = min(min_time, t_times[0])
        max_time = max(max_time, t_times[-1])

        peak_idx = int(np.argmin(t_msl))

        tracks_meta.append(
            {
                "track_id": tid,
                "start": start_pos,
                "end": end_pos,
                "start_lat": t_lats[0],
                "start_lon": t_lons[0],
                "start_time": t_times[0],
                "end_lat": t_lats[-1],
                "end_lon": t_lons[-1],
                "end_time": t_times[-1],
                "peak_lat": t_lats[peak_idx],
                "peak_lon": t_lons[peak_idx],
                "peak_time": t_times[peak_idx],
                "peak_value": t_msl[peak_idx],
                "duration_hours": float((t_times[-1] - t_times[0]) / 3600000.0),
            }
        )

        current_idx += points_per_track + 1

        tracks_list.append(
            {
                "id": tid,
                "lats": t_lats,
                "lons": t_lons,
                "times": t_times,
                "msl": t_msl,
                "vo": t_vo,
            }
        )

    trackjson_data: dict[str, object] = {
        "format": "TrackJSON/1.0",
        "metadata": {
            "primary_var": "msl",
            "mode": "min",
            "units": {"msl": "Pa", "vo": "s^-1"},
            "bounds": {
                "min_time": int(min_time),
                "max_time": int(max_time),
                "min_lat": float(min_lat),
                "max_lat": float(max_lat),
                "min_lon": float(min_lon),
                "max_lon": float(max_lon),
            },
        },
        "points": {
            "lat": lats_flat,
            "lon": lons_flat,
            "time": times_flat,
            "variables": {
                "msl": msl_flat,
                "vo": vo_flat,
            },
        },
        "tracks": tracks_meta,
    }

    geojson_features: list[dict[str, object]] = []
    for tr in tracks_list:
        lons = tr["lons"]
        lats = tr["lats"]
        times = tr["times"]
        msl = tr["msl"]
        if (
            not isinstance(lons, list)
            or not isinstance(lats, list)
            or not isinstance(times, list)
            or not isinstance(msl, list)
        ):
            msg = "Synthetic track values must be lists"
            raise TypeError(msg)
        numeric_times = [float(value) for value in times]
        numeric_msl = [float(value) for value in msl]
        coords = [[lon, lat] for lon, lat in zip(lons, lats, strict=True)]
        geojson_features.append(
            {
                "type": "Feature",
                "id": tr["id"],
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "track_id": tr["id"],
                    "times": tr["times"],
                    "variables": {"msl": tr["msl"], "vo": tr["vo"]},
                    "peak_value": min(numeric_msl),
                    "duration_hours": (numeric_times[-1] - numeric_times[0])
                    / 3600000.0,
                },
            }
        )

    geojson_data: dict[str, object] = {
        "type": "FeatureCollection",
        "pystormtracker": {"primary_var": "msl", "mode": "min"},
        "features": geojson_features,
    }

    return trackjson_data, geojson_data


def run_scale_benchmark(n_tracks: int, points_per_track: int = 30) -> None:
    """Run benchmark for a given track count scale."""
    tjson_data, geojson_data = generate_mock_tracks(n_tracks, points_per_track)
    total_points = n_tracks * points_per_track

    t0 = time.perf_counter()
    tjson_str = json.dumps(tjson_data, separators=(",", ":"))
    tjson_dumps_time = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    geojson_str = json.dumps(geojson_data, separators=(",", ":"))
    geojson_dumps_time = (time.perf_counter() - t0) * 1000.0

    tjson_bytes = len(tjson_str.encode("utf-8"))
    geojson_bytes = len(geojson_str.encode("utf-8"))

    tjson_gzip = len(gzip.compress(tjson_str.encode("utf-8")))
    geojson_gzip = len(gzip.compress(geojson_str.encode("utf-8")))

    t0 = time.perf_counter()
    parsed_tjson = json.loads(tjson_str)
    tjson_loads_time = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    parsed_geojson = json.loads(geojson_str)
    geojson_loads_time = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    flat_lats_tj = np.array(
        [x if x is not None else np.nan for x in parsed_tjson["points"]["lat"]],
        dtype=np.float32,
    )
    flat_lons_tj = np.array(
        [x if x is not None else np.nan for x in parsed_tjson["points"]["lon"]],
        dtype=np.float32,
    )
    tjson_webgl_time = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    flat_lats_gj: list[float] = []
    flat_lons_gj: list[float] = []
    for feat in parsed_geojson["features"]:
        coords = feat["geometry"]["coordinates"]
        for c in coords:
            flat_lons_gj.append(c[0])
            flat_lats_gj.append(c[1])
        flat_lons_gj.append(np.nan)
        flat_lats_gj.append(np.nan)
    arr_lats_gj = np.array(flat_lats_gj, dtype=np.float32)
    arr_lons_gj = np.array(flat_lons_gj, dtype=np.float32)
    geojson_webgl_time = (time.perf_counter() - t0) * 1000.0

    if flat_lats_tj.size != flat_lons_tj.size or arr_lats_gj.size != arr_lons_gj.size:
        msg = "Latitude and longitude arrays must have matching lengths"
        raise ValueError(msg)

    print(f"\n--- Scale: {n_tracks:,} tracks ({total_points:,} trajectory points) ---")
    print("Raw JSON Size:")
    print(f"  TrackJSON v1.0 SoA: {tjson_bytes / (1024 * 1024):.2f} MB")
    print(
        "  GeoJSON LineString: "
        f"{geojson_bytes / (1024 * 1024):.2f} MB "
        f"({geojson_bytes / tjson_bytes:.2f}x size)"
    )
    print("Gzip Compressed Size:")
    print(f"  TrackJSON v1.0 SoA: {tjson_gzip / (1024 * 1024):.2f} MB")
    print(
        "  GeoJSON LineString: "
        f"{geojson_gzip / (1024 * 1024):.2f} MB "
        f"({geojson_gzip / tjson_gzip:.2f}x size)"
    )
    print("JSON Serialization Time:")
    print(f"  TrackJSON v1.0 SoA: {tjson_dumps_time:.1f} ms")
    print(f"  GeoJSON LineString: {geojson_dumps_time:.1f} ms")
    print("JSON Parse Time:")
    print(f"  TrackJSON v1.0 SoA: {tjson_loads_time:.1f} ms")
    print(
        "  GeoJSON LineString: "
        f"{geojson_loads_time:.1f} ms "
        f"({geojson_loads_time / tjson_loads_time:.2f}x parse time)"
    )
    print("WebGL Float32Array Build Time:")
    print(f"  TrackJSON v1.0 SoA: {tjson_webgl_time:.1f} ms")
    print(
        "  GeoJSON LineString: "
        f"{geojson_webgl_time:.1f} ms "
        f"({geojson_webgl_time / tjson_webgl_time:.2f}x build time)"
    )


def main() -> None:
    print("=== PyStormTracker TrackJSON vs GeoJSON I/O Benchmark ===")
    for n in (1000, 10000, 50000):
        run_scale_benchmark(n, 30)


if __name__ == "__main__":
    main()
