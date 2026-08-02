"""RFC 7946 GeoJSON interchange for PyStormTracker trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..models.tracks import Tracks
from .json import infer_intensity_mode


def _nullable_values(values: NDArray[np.float64]) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in values]


def write_geojson(tracks: Tracks, outfile: str | Path) -> None:
    """Write one GeoJSON LineString feature per track.

    Timestamps and trajectory variables are stored as aligned arrays in feature
    properties.  They are GeoJSON foreign members and preserve information not
    represented by a LineString geometry alone.
    """
    features: list[dict[str, object]] = []
    for track in tracks:
        indices = track.indices
        if len(indices) == 0:
            continue
        timestamps = tracks.times[indices].astype("datetime64[ms]").astype(np.int64)
        properties: dict[str, object] = {
            "track_id": int(track.track_id),
            "times": [int(value) for value in timestamps],
            "start_time": int(timestamps[0]),
            "end_time": int(timestamps[-1]),
            "duration_hours": float((timestamps[-1] - timestamps[0]) / 3_600_000),
            "variables": {
                name: _nullable_values(values[indices])
                for name, values in tracks.vars.items()
            },
        }
        features.append(
            {
                "type": "Feature",
                "id": int(track.track_id),
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(lon), float(lat)]
                        for lon, lat in zip(
                            tracks.lons[indices], tracks.lats[indices], strict=True
                        )
                    ],
                },
                "properties": properties,
            }
        )

    document = {
        "type": "FeatureCollection",
        "pystormtracker": {
            "primary_var": tracks.track_type,
            "mode": infer_intensity_mode(tracks.track_type),
        },
        "features": features,
    }
    with open(outfile, "w", encoding="utf-8") as output:
        json.dump(document, output, separators=(",", ":"), allow_nan=False)


def _milliseconds(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "GeoJSON timestamps must be milliseconds since the Unix epoch."
        )
    if not np.isfinite(value):
        raise ValueError("GeoJSON timestamps must be finite.")
    return int(value)


def _feature_coordinates(feature: dict[str, object]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "LineString":
        raise ValueError("GeoJSON track features must have LineString geometries.")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("GeoJSON LineString coordinates must be a nonempty array.")
    parsed_coordinates: list[tuple[float, float]] = []
    for position in coordinates:
        if (
            not isinstance(position, list)
            or len(position) < 2
            or isinstance(position[0], bool)
            or isinstance(position[1], bool)
            or not isinstance(position[0], (int, float))
            or not isinstance(position[1], (int, float))
        ):
            raise ValueError(
                "GeoJSON coordinates must contain numeric longitude-latitude pairs."
            )
        parsed_coordinates.append((float(position[0]), float(position[1])))
    return parsed_coordinates


def read_geojson(infile: str | Path) -> Tracks:
    """Read a GeoJSON FeatureCollection containing LineString track features."""
    with open(infile, encoding="utf-8") as source:
        document: object = json.load(source)
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON input must be a FeatureCollection.")
    raw_features = document.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("GeoJSON FeatureCollection requires a features array.")

    root_metadata = document.get("pystormtracker")
    metadata = root_metadata if isinstance(root_metadata, dict) else {}
    primary_value = metadata.get("primary_var", "unknown")
    primary_var = primary_value if isinstance(primary_value, str) else "unknown"

    parsed: list[
        tuple[int, list[tuple[float, float]], dict[str, object], dict[str, object]]
    ] = []
    variable_names: set[str] = set()
    for feature in raw_features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("GeoJSON features must be Feature objects.")
        properties_value = feature.get("properties")
        properties = properties_value if isinstance(properties_value, dict) else {}
        raw_track_id = properties.get("track_id", feature.get("id"))
        if isinstance(raw_track_id, bool) or not isinstance(raw_track_id, int):
            raise ValueError("GeoJSON track features require an integer track_id.")
        coordinates = _feature_coordinates(feature)
        variables_value = properties.get("variables", {})
        variables = variables_value if isinstance(variables_value, dict) else {}
        for name, values in variables.items():
            if not isinstance(name, str) or not isinstance(values, list):
                raise ValueError("GeoJSON variable properties must be named arrays.")
            if len(values) != len(coordinates):
                raise ValueError(
                    "GeoJSON variable arrays must match coordinate length."
                )
            variable_names.add(name)
        parsed.append((raw_track_id, coordinates, variables, properties))

    output_ids: list[int] = []
    output_lats: list[float] = []
    output_lons: list[float] = []
    output_times: list[int] = []
    output_variables: dict[str, list[float]] = {name: [] for name in variable_names}
    for track_id, coordinates, variables, properties in parsed:
        raw_times = properties.get("times")
        if not isinstance(raw_times, list) or len(raw_times) != len(coordinates):
            raise ValueError("GeoJSON track features require aligned times properties.")
        for index, coordinate in enumerate(coordinates):
            output_ids.append(track_id)
            output_lons.append(coordinate[0])
            output_lats.append(coordinate[1])
            output_times.append(_milliseconds(raw_times[index]))
            for name in variable_names:
                values = variables.get(name)
                if isinstance(values, list):
                    value = values[index]
                    output_variables[name].append(
                        float(value) if isinstance(value, (int, float)) else np.nan
                    )
                else:
                    output_variables[name].append(np.nan)

    return Tracks(
        track_ids=np.asarray(output_ids, dtype=np.int64),
        times=np.asarray(output_times, dtype=np.int64)
        .astype("datetime64[ms]")
        .astype("datetime64[s]"),
        lats=np.asarray(output_lats, dtype=np.float64),
        lons=np.asarray(output_lons, dtype=np.float64),
        vars_dict={
            name: np.asarray(values, dtype=np.float64)
            for name, values in output_variables.items()
        },
        track_type=primary_var,
    )
