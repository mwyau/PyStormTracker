"""RFC 7946 GeoJSON interchange for PyStormTracker trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..models.tracks import Tracks
from .json import infer_intensity_mode

Coordinate = tuple[float, float]


def _nullable_values(values: NDArray[np.float64]) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _split_antimeridian(coordinates: list[Coordinate]) -> list[list[Coordinate]]:
    """Split a trajectory wherever consecutive longitudes cross ±180 degrees."""
    segments: list[list[Coordinate]] = [[coordinates[0]]]
    for coordinate in coordinates[1:]:
        previous_lon = segments[-1][-1][0]
        if abs(coordinate[0] - previous_lon) > 180.0:
            segments.append([coordinate])
        else:
            segments[-1].append(coordinate)
    return segments


def _geometry_for_coordinates(coordinates: list[Coordinate]) -> dict[str, object]:
    """Return valid GeoJSON geometry without drawing across the antimeridian."""
    if len(coordinates) == 1:
        return {"type": "Point", "coordinates": coordinates[0]}

    segments = _split_antimeridian(coordinates)
    if len(segments) == 1:
        return {"type": "LineString", "coordinates": segments[0]}
    if all(len(segment) >= 2 for segment in segments):
        return {"type": "MultiLineString", "coordinates": segments}

    geometries: list[dict[str, object]] = []
    for segment in segments:
        if len(segment) == 1:
            geometries.append({"type": "Point", "coordinates": segment[0]})
        else:
            geometries.append({"type": "LineString", "coordinates": segment})
    return {"type": "GeometryCollection", "geometries": geometries}


def write_geojson(tracks: Tracks, outfile: str | Path) -> None:
    """Write one GeoJSON feature per track.

    Timestamps and trajectory variables are stored as aligned arrays in feature
    properties. They are GeoJSON foreign members and preserve information not
    represented by a LineString or Point geometry alone.
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
        coordinates: list[Coordinate] = [
            (float(lon), float(lat))
            for lon, lat in zip(tracks.lons[indices], tracks.lats[indices], strict=True)
        ]
        geometry = _geometry_for_coordinates(coordinates)
        features.append(
            {
                "type": "Feature",
                "id": int(track.track_id),
                "geometry": geometry,
                "properties": properties,
            }
        )

    document = {
        "type": "FeatureCollection",
        "pystormtracker": {
            "primary_var": tracks.track_type,
            "mode": (
                tracks.mode
                if tracks.mode in ("min", "max")
                else infer_intensity_mode(tracks.track_type)
            ),
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


def _parse_position(position: object) -> Coordinate:
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
    return float(position[0]), float(position[1])


def _geometry_coordinates(geometry: object) -> list[Coordinate]:
    if not isinstance(geometry, dict):
        raise ValueError("GeoJSON track features must have supported geometries.")
    geometry_type = geometry.get("type")
    coordinates_value = geometry.get("coordinates")
    if geometry_type == "Point":
        if not isinstance(coordinates_value, list):
            raise ValueError("GeoJSON Point coordinates must be an array.")
        return [_parse_position(coordinates_value)]
    elif geometry_type == "LineString":
        if not isinstance(coordinates_value, list) or len(coordinates_value) < 2:
            raise ValueError("GeoJSON LineString coordinates must be an array.")
        return [_parse_position(position) for position in coordinates_value]
    elif geometry_type == "MultiLineString":
        if not isinstance(coordinates_value, list) or not coordinates_value:
            raise ValueError("GeoJSON MultiLineString coordinates must be an array.")
        coordinates: list[Coordinate] = []
        for segment in coordinates_value:
            if not isinstance(segment, list) or len(segment) < 2:
                raise ValueError(
                    "GeoJSON MultiLineString segments must have at least two positions."
                )
            coordinates.extend(_parse_position(position) for position in segment)
        return coordinates
    elif geometry_type == "GeometryCollection":
        geometries = geometry.get("geometries")
        if not isinstance(geometries, list) or not geometries:
            raise ValueError("GeoJSON GeometryCollection must contain geometries.")
        coordinates = []
        for child in geometries:
            coordinates.extend(_geometry_coordinates(child))
        if not coordinates:
            raise ValueError("GeoJSON track coordinates must be a nonempty array.")
        return coordinates
    else:
        raise ValueError(
            "GeoJSON track features must have Point, LineString, "
            "MultiLineString, or GeometryCollection geometries."
        )


def _feature_coordinates(feature: dict[str, object]) -> list[Coordinate]:
    coordinates = _geometry_coordinates(feature.get("geometry"))
    if not coordinates:
        raise ValueError("GeoJSON track coordinates must be a nonempty array.")
    return coordinates


def read_geojson(infile: str | Path) -> Tracks:
    """Read a FeatureCollection containing Point or split line geometries."""
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
    mode_value = metadata.get("mode")
    if mode_value is not None and mode_value not in ("min", "max"):
        raise ValueError("GeoJSON pystormtracker.mode must be 'min' or 'max'.")
    mode = cast(Literal["min", "max"] | None, mode_value)

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
        mode=mode,
    )
