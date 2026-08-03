"""Reader and writer for the supported TRACK/Hodges ASCII subset."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from ..models.geo import normalize_longitudes_360
from ..models.tracks import Tracks, TracksBuilder
from ..models.units import canonical_unit_for


def _parse_track_time(token: str) -> np.datetime64:
    if len(token) == 10 and token.isdigit():
        try:
            value = datetime.strptime(token, "%Y%m%d%H")
        except ValueError as exc:
            raise ValueError(f"Invalid TRACK calendar time {token!r}") from exc
        return np.datetime64(value, "ms")
    try:
        seconds = float(token)
    except ValueError as exc:
        raise ValueError(f"Invalid TRACK point time {token!r}") from exc
    if not np.isfinite(seconds) or not seconds.is_integer():
        raise ValueError(f"Invalid TRACK point time {token!r}")
    return np.datetime64("1970-01-01T00:00:00", "ms") + np.timedelta64(
        int(seconds * 1000), "ms"
    )


def read_hodges(
    path: str | Path,
    *,
    primary_var: str = "Intensity1",
    mode: Literal["min", "max"] = "max",
) -> Tracks:
    """Read ``TRACK_NUM/TRACK_ID/POINT_NUM`` records into packed tracks.

    The supported point layout is ``time longitude latitude intensity``. Time
    may be a ``YYYYMMDDHH`` token, Unix seconds, or a global frame index
    anchored at the Unix epoch. Generic ``Intensity1`` values are not scaled.
    """
    point_groups: dict[int, list[tuple[np.datetime64, float, float, float]]] = {}
    expected_tracks: int | None = None
    current_id: int | None = None
    expected_points = 0
    parsed_points = 0
    try:
        with Path(path).open(encoding="utf-8") as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("TRACK_NUM"):
                    fields = line.split()
                    if len(fields) < 2:
                        raise ValueError("Malformed TRACK_NUM header")
                    expected_tracks = int(fields[1])
                    continue
                if line.startswith("TRACK_ID"):
                    if current_id is not None and parsed_points != expected_points:
                        raise ValueError(
                            f"TRACK_ID {current_id} is truncated: expected "
                            f"{expected_points} points, got {parsed_points}"
                        )
                    fields = line.split()
                    if len(fields) != 2:
                        raise ValueError("Malformed TRACK_ID header")
                    current_id = int(fields[1])
                    expected_points = 0
                    parsed_points = 0
                    continue
                if line.startswith("POINT_NUM"):
                    if current_id is None:
                        raise ValueError("POINT_NUM appeared before TRACK_ID")
                    fields = line.split()
                    if len(fields) != 2:
                        raise ValueError("Malformed POINT_NUM header")
                    expected_points = int(fields[1])
                    if expected_points < 0:
                        raise ValueError("POINT_NUM must be nonnegative")
                    parsed_points = 0
                    continue
                if current_id is None or expected_points == 0:
                    continue
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError("Malformed TRACK point record")
                try:
                    point_time = _parse_track_time(fields[0])
                    lon = float(fields[1])
                    lat = float(fields[2])
                    intensity = float(fields[3])
                except ValueError as exc:
                    raise ValueError(f"Malformed TRACK point record: {line}") from exc
                if not all(np.isfinite(value) for value in (lon, lat, intensity)):
                    raise ValueError("TRACK point values must be finite")
                if not -90.0 <= lat <= 90.0:
                    raise ValueError("TRACK latitude is outside [-90, 90]")
                point_groups.setdefault(current_id, []).append(
                    (point_time, lat, lon, intensity)
                )
                parsed_points += 1
                if parsed_points > expected_points:
                    raise ValueError(f"TRACK_ID {current_id} has too many points")
            if current_id is not None and parsed_points != expected_points:
                raise ValueError(
                    f"TRACK_ID {current_id} is truncated: expected {expected_points} "
                    f"points, got {parsed_points}"
                )
    except OSError as exc:
        raise ValueError(f"Unable to read TRACK file {path}: {exc}") from exc

    if expected_tracks is not None and expected_tracks != len(point_groups):
        raise ValueError(
            f"TRACK_NUM declares {expected_tracks} tracks but parsed "
            f"{len(point_groups)}"
        )
    units = {primary_var: canonical_unit_for(primary_var) or "1"}
    builder = TracksBuilder(primary_var, mode, units)
    for track_id, points in point_groups.items():
        builder.add_track(
            track_id,
            [point[0] for point in points],
            [point[1] for point in points],
            [point[2] for point in points],
            {primary_var: [point[3] for point in points]},
        )
    return builder.finish()


def write_hodges(tracks: Tracks, outfile: str | Path) -> None:
    """Write the supported TRACK subset with each point's actual timestamp."""
    variable_name = tracks.primary_var
    if len(tracks) and variable_name not in tracks.variables:
        raise ValueError("TRACK writing requires the primary variable column")
    with Path(outfile).open("w", encoding="utf-8") as output:
        output.write("0\n")
        output.write("0 0\n")
        output.write(f"TRACK_NUM {len(tracks):10d} ADD_FLD    0   0 &\n")
        for track in tracks:
            output.write(f"TRACK_ID {track.track_id:2d}\n")
            output.write(f"POINT_NUM {len(track):3d}\n")
            values = tracks.variables[variable_name][track.point_slice]
            for _point_index, (timestamp, value, lat, lon) in enumerate(
                zip(track.times, values, track.lats, track.lons, strict=True), start=1
            ):
                if np.isnat(timestamp):
                    raise ValueError("TRACK writing requires real timestamps")
                milliseconds = int(
                    (timestamp - np.datetime64("1970-01-01T00:00:00", "ms"))
                    / np.timedelta64(1, "ms")
                )
                if milliseconds % (3_600 * 1000) == 0:
                    token = datetime.fromtimestamp(
                        milliseconds / 1000.0, tz=UTC
                    ).strftime("%Y%m%d%H")
                else:
                    token = str(milliseconds // 1000)
                lon_360 = float(normalize_longitudes_360(np.asarray([lon]))[0])
                output.write(f"{token} {lon_360:10.6f} {lat:10.6f} {value:12.6e}\n")
