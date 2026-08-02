"""TrackJSON v1.0 serialization for array-backed storm trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..models.tracks import Tracks


def infer_intensity_mode(
    track_type: str = "unknown",
    var_name: str | None = None,
    mode: str | None = None,
) -> Literal["min", "max"]:
    """Infer the extremum direction for a tracking variable."""
    if mode in ("min", "max"):
        return cast(Literal["min", "max"], mode)
    if var_name is not None and var_name.lower() in ("msl", "slp", "pnm", "pres"):
        return "min"
    if track_type.lower() == "msl":
        return "min"
    return "max"


def infer_track_type(tracks: Tracks) -> str:
    """Infer a primary variable name from the available trajectory variables."""
    if tracks.track_type != "unknown":
        return tracks.track_type
    if "msl" in tracks.vars:
        return "msl"
    if "vo" in tracks.vars:
        return "vo"
    if "Intensity1" in tracks.vars:
        return "Intensity1"
    if tracks.vars:
        return next(iter(tracks.vars))
    return "unknown"


def _units_for_variable(var_name: str) -> str | None:
    normalized = var_name.lower()
    if normalized in ("msl", "slp", "pnm", "pres"):
        return "Pa"
    if normalized in ("vo", "vort"):
        return "s^-1"
    return None


def _as_nullable_float(values: NDArray[np.float64]) -> list[float | None]:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _as_nullable_int(values: NDArray[np.float64]) -> list[int | None]:
    return [None if not np.isfinite(value) else int(value) for value in values]


def _primary_variable(tracks: Tracks) -> str:
    inferred = infer_track_type(tracks)
    if inferred in tracks.vars:
        return inferred
    if tracks.vars:
        return next(iter(tracks.vars))
    return inferred if inferred != "unknown" else "intensity"


def write_json(tracks: Tracks, outfile: str | Path) -> None:
    """Write ``tracks`` as TrackJSON v1.0 with separator-delimited SoA points."""
    primary_var = _primary_variable(tracks)
    mode = infer_intensity_mode(tracks.track_type, primary_var)
    units = {
        name: unit
        for name in tracks.vars
        if (unit := _units_for_variable(name)) is not None
    }

    point_lats: list[float] = []
    point_lons: list[float] = []
    point_times: list[float] = []
    point_variables: dict[str, list[float]] = {name: [] for name in tracks.vars}
    track_metadata: list[dict[str, object]] = []

    for track in tracks:
        indices = track.indices
        if len(indices) == 0:
            continue
        start = len(point_lats)
        times = tracks.times[indices].astype("datetime64[ms]").astype(np.int64)
        lats = tracks.lats[indices]
        lons = tracks.lons[indices]

        point_lats.extend(float(value) for value in lats)
        point_lons.extend(float(value) for value in lons)
        point_times.extend(float(value) for value in times)
        for name, values in tracks.vars.items():
            point_variables[name].extend(float(value) for value in values[indices])

        end = len(point_lats) - 1
        summary: dict[str, object] = {
            "track_id": int(track.track_id),
            "start": start,
            "end": end,
            "start_lat": float(lats[0]),
            "start_lon": float(lons[0]),
            "start_time": int(times[0]),
            "end_lat": float(lats[-1]),
            "end_lon": float(lons[-1]),
            "end_time": int(times[-1]),
            "duration_hours": float((times[-1] - times[0]) / 3_600_000),
        }
        if primary_var in tracks.vars:
            primary_values = tracks.vars[primary_var][indices]
            if np.any(np.isfinite(primary_values)):
                peak_index = (
                    int(np.nanargmin(primary_values))
                    if mode == "min"
                    else int(np.nanargmax(primary_values))
                )
                summary.update(
                    {
                        "peak_lat": float(lats[peak_index]),
                        "peak_lon": float(lons[peak_index]),
                        "peak_time": int(times[peak_index]),
                        "peak_value": float(primary_values[peak_index]),
                    }
                )
        track_metadata.append(summary)

        point_lats.append(float("nan"))
        point_lons.append(float("nan"))
        point_times.append(float("nan"))
        for separator_values in point_variables.values():
            separator_values.append(float("nan"))

    if track_metadata:
        point_lats.pop()
        point_lons.pop()
        point_times.pop()
        for separator_values in point_variables.values():
            separator_values.pop()

    lat_array = np.asarray(point_lats, dtype=np.float64)
    lon_array = np.asarray(point_lons, dtype=np.float64)
    time_array = np.asarray(point_times, dtype=np.float64)
    variable_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in point_variables.items()
    }
    bounds: dict[str, int | float | None] = {
        "min_time": int(np.nanmin(time_array)) if time_array.size else None,
        "max_time": int(np.nanmax(time_array)) if time_array.size else None,
        "min_lat": float(np.nanmin(lat_array)) if lat_array.size else None,
        "max_lat": float(np.nanmax(lat_array)) if lat_array.size else None,
        "min_lon": float(np.nanmin(lon_array)) if lon_array.size else None,
        "max_lon": float(np.nanmax(lon_array)) if lon_array.size else None,
    }
    variables = {
        name: _as_nullable_float(values) for name, values in variable_arrays.items()
    }
    points: dict[str, object] = {
        "lat": _as_nullable_float(lat_array),
        "lon": _as_nullable_float(lon_array),
        "time": _as_nullable_int(time_array),
        "variables": variables,
    }
    document = {
        "format": "TrackJSON/1.0",
        "metadata": {
            "primary_var": primary_var,
            "mode": mode,
            "units": units,
            "bounds": bounds,
        },
        "points": points,
        "tracks": track_metadata,
    }
    with open(outfile, "w", encoding="utf-8") as output:
        json.dump(document, output, separators=(",", ":"), allow_nan=False)


def _numeric_array(values: object, name: str, length: int) -> NDArray[np.float64]:
    if not isinstance(values, list):
        raise ValueError(f"TrackJSON points.{name} must be an array.")
    if len(values) != length:
        raise ValueError(f"TrackJSON points.{name} must have {length} values.")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"TrackJSON points.{name} must contain numbers or null."
        ) from exc


def read_json(infile: str | Path) -> Tracks:
    """Read a TrackJSON v1.0 document into array-backed trajectories."""
    with open(infile, encoding="utf-8") as source:
        raw_document: object = json.load(source)
    if not isinstance(raw_document, dict):
        raise ValueError("TrackJSON document must be a JSON object.")

    format_name = raw_document.get("format")
    if format_name != "TrackJSON/1.0":
        raise ValueError(f"Unsupported JSON track format: {format_name!r}.")
    raw_points = raw_document.get("points")
    raw_tracks = raw_document.get("tracks")
    if not isinstance(raw_points, dict) or not isinstance(raw_tracks, list):
        raise ValueError(
            "TrackJSON requires object 'points' and array 'tracks' fields."
        )
    if not raw_tracks:
        return Tracks()

    lat_values = raw_points.get("lat")
    if not isinstance(lat_values, list):
        raise ValueError("TrackJSON points.lat must be an array.")
    length = len(lat_values)
    latitudes = _numeric_array(lat_values, "lat", length)
    longitudes = _numeric_array(raw_points.get("lon"), "lon", length)
    timestamps = _numeric_array(raw_points.get("time"), "time", length)

    metadata = raw_document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("TrackJSON metadata must be an object.")
    primary_var_value = metadata.get("primary_var")
    if not isinstance(primary_var_value, str):
        raise ValueError("TrackJSON metadata.primary_var must be a string.")
    primary_var = primary_var_value
    raw_variables = raw_points.get("variables")
    if not isinstance(raw_variables, dict):
        raise ValueError("TrackJSON points.variables must be an object.")
    variables_source = raw_variables

    variable_arrays: dict[str, NDArray[np.float64]] = {}
    for name, values in variables_source.items():
        if not isinstance(name, str):
            raise ValueError("TrackJSON variable names must be strings.")
        variable_arrays[name] = _numeric_array(values, f"variables.{name}", length)

    output_ids: list[int] = []
    output_lats: list[float] = []
    output_lons: list[float] = []
    output_times: list[float] = []
    output_variables: dict[str, list[float]] = {name: [] for name in variable_arrays}
    for item in raw_tracks:
        if not isinstance(item, dict):
            raise ValueError("Each TrackJSON track entry must be an object.")
        track_id, start, end = item.get("track_id"), item.get("start"), item.get("end")
        if (
            not isinstance(track_id, int)
            or isinstance(track_id, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end >= length
        ):
            raise ValueError(
                "TrackJSON track entries require valid track_id, start, and end."
            )
        selected = slice(start, end + 1)
        if (
            not np.all(np.isfinite(latitudes[selected]))
            or not np.all(np.isfinite(longitudes[selected]))
            or not np.all(np.isfinite(timestamps[selected]))
        ):
            raise ValueError(
                "TrackJSON track ranges cannot include separator null values."
            )
        count = end - start + 1
        output_ids.extend([track_id] * count)
        output_lats.extend(latitudes[selected])
        output_lons.extend(longitudes[selected])
        output_times.extend(timestamps[selected])
        for name, values in variable_arrays.items():
            output_variables[name].extend(values[selected])

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
