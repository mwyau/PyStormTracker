"""Reader and writer for the supported TRACK/Hodges ASCII subset."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..models.geo import normalize_longitudes_360
from ..models.time import decode_time_values, encode_time_values
from ..models.tracks import DetectionMode, Tracks, TracksMetadata, _TracksBuilder
from ..models.units import canonical_unit_for, resolve_mode

type TrackNumericTime = Literal["reject", "frame_index", "unix_seconds"]


def _parse_track_time(
    token: str,
    *,
    track_numeric_time: TrackNumericTime,
    track_frame_times_ms: NDArray[np.int64] | None,
) -> np.datetime64 | int:
    """Parse one TRACK time token under an explicit numeric convention."""
    if len(token) == 10 and token.isdigit():
        try:
            value = datetime.strptime(token, "%Y%m%d%H")
        except ValueError as exc:
            if track_numeric_time != "unix_seconds":
                raise ValueError(f"Invalid TRACK calendar time {token!r}") from exc
        else:
            return np.datetime64(value, "ms")
    if track_numeric_time == "reject":
        raise ValueError(
            f"Ambiguous numeric TRACK point time {token!r}; TRACK may write "
            "a one-based frame index. Set track_numeric_time='frame_index' with "
            "track_frame_times, or explicitly request "
            "track_numeric_time='unix_seconds'."
        )
    try:
        numeric_value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid TRACK point time {token!r}") from exc
    if not numeric_value.is_finite():
        raise ValueError(f"Invalid TRACK point time {token!r}")
    if track_numeric_time == "frame_index":
        if track_frame_times_ms is None:
            raise ValueError(
                "track_numeric_time='frame_index' requires track_frame_times so "
                "TRACK frame indices can be mapped to source timestamps"
            )
        if numeric_value != numeric_value.to_integral_value():
            raise ValueError(f"TRACK frame index must be integral, got {token!r}")
        frame_index = int(numeric_value)
        if frame_index < 1 or frame_index > track_frame_times_ms.size:
            raise ValueError(
                f"TRACK frame index {frame_index} is outside supplied "
                f"track_frame_times range [1, {track_frame_times_ms.size}]"
            )
        return int(track_frame_times_ms[frame_index - 1])
    if track_numeric_time == "unix_seconds":
        return int((numeric_value * Decimal(1000)).to_integral_value())
    raise ValueError(f"Unsupported track_numeric_time {track_numeric_time!r}")


def read_track(
    path: str | Path,
    *,
    primary_variable: str = "Intensity1",
    mode: DetectionMode | None = "auto",
    track_numeric_time: TrackNumericTime = "reject",
    track_frame_times: object | None = None,
) -> Tracks:
    """Read ``TRACK_NUM/TRACK_ID/POINT_NUM`` records into packed tracks.

    The supported point layout is ``time longitude latitude intensity``. TRACK
    itself writes either a ``YYYYMMDDHH`` calendar token or a one-based global
    frame index. Numeric values are therefore rejected by default. Use
    ``track_numeric_time='frame_index'`` with the exact source ``track_frame_times``
    coordinate, or explicitly select ``track_numeric_time='unix_seconds'`` for a
    non-TRACK extension. Generic ``Intensity1`` values are not scaled.
    """
    if track_numeric_time not in ("reject", "frame_index", "unix_seconds"):
        raise ValueError(
            f"unsupported track_numeric_time {track_numeric_time!r}; "
            "expected 'reject', 'frame_index', or 'unix_seconds'"
        )
    if track_frame_times is not None and track_numeric_time != "frame_index":
        raise ValueError(
            "track_frame_times is only valid with track_numeric_time='frame_index'"
        )
    frame_times_ms = (
        encode_time_values(track_frame_times) if track_frame_times is not None else None
    )
    if track_numeric_time == "frame_index" and (
        frame_times_ms is None or frame_times_ms.size == 0
    ):
        raise ValueError(
            "track_numeric_time='frame_index' requires a non-empty "
            "track_frame_times sequence"
        )

    point_groups: dict[int, list[tuple[np.datetime64 | int, float, float, float]]] = {}
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
                    point_time = _parse_track_time(
                        fields[0],
                        track_numeric_time=track_numeric_time,
                        track_frame_times_ms=frame_times_ms,
                    )
                    lon = float(fields[1])
                    lat = float(fields[2])
                    intensity = float(fields[3])
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed TRACK point record: {line}; {exc}"
                    ) from exc
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
                if parsed_points == expected_points:
                    current_id = None
                    expected_points = 0
                    parsed_points = 0
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
    units = {primary_variable: canonical_unit_for(primary_variable) or "1"}
    builder = _TracksBuilder(
        TracksMetadata(primary_variable, resolve_mode(primary_variable, mode), units)
    )
    for track_id, points in point_groups.items():
        builder.add_track(
            track_id,
            [point[0] for point in points],
            [point[1] for point in points],
            [point[2] for point in points],
            {primary_variable: [point[3] for point in points]},
        )
    return builder.finish()


def write_track(tracks: Tracks, outfile: str | Path) -> None:
    """Write the strict TRACK subset using calendar-hour point timestamps.

    TRACK's calendar representation has no sub-hour or sub-second field. Such
    timestamps are rejected instead of being emitted as numeric Unix seconds,
    which the default strict reader does not interpret.
    """
    variable_name = tracks.primary_variable
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
                try:
                    dt = decode_time_values([int(timestamp)])[0]
                except (OverflowError, ValueError) as exc:
                    raise ValueError(
                        "TRACK output cannot represent timestamp "
                        f"{int(timestamp)!r}; expected a calendar date within "
                        "Python datetime range"
                    ) from exc
                if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
                    raise ValueError(
                        "TRACK output supports only whole-hour timestamps; "
                        f"timestamp {dt.isoformat()} has an unsupported cadence"
                    )
                token = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"
                lon_360 = float(normalize_longitudes_360(np.asarray([lon]))[0])
                output.write(f"{token} {lon_360:10.6f} {lat:10.6f} {value:12.6e}\n")


__all__ = [
    "TrackNumericTime",
    "read_track",
    "write_track",
]
