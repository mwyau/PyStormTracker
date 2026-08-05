"""IMILAST text interoperability at the packed-model boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from ..models.time import decode_time_values
from ..models.tracks import Tracks, TracksMetadata, _TracksBuilder
from ..models.units import ModeOption, canonical_unit_for, resolve_mode


def _parse_time(value: str) -> np.datetime64:
    if len(value) == 10 and value.isdigit():
        dt = datetime(
            int(value[:4]),
            int(value[4:6]),
            int(value[6:8]),
            int(value[8:10]),
            tzinfo=UTC,
        )
        return np.datetime64(dt.replace(tzinfo=None), "ms")
    try:
        milliseconds = round(float(value) * 1000.0)
        return np.datetime64(milliseconds, "ms")
    except ValueError:
        return np.datetime64(value, "ms")


def read_imilast(
    filename: Path | str,
    *,
    primary_var: str | None = None,
    mode: ModeOption | None = "auto",
) -> Tracks:
    """Read the supported IMILAST subset into packed trajectories.

    IMILAST pressure values are interpreted as hPa and converted to Pa; the
    vorticity field is interpreted in ``10^-5 s^-1`` and converted to ``s^-1``.
    Generic fields are not numerically classified or scaled.
    """
    path = Path(filename)
    track_points: dict[int, list[tuple[np.datetime64, float, float, float]]] = {}
    variable_name = "Intensity1"
    with path.open(encoding="utf-8") as source:
        header = source.readline()
        if "," in header:
            fields = [field.strip() for field in header.split(",")]
            if len(fields) > 10 and fields[10]:
                variable_name = fields[10].strip()
        if primary_var is not None:
            variable_name = primary_var
        for line in source:
            fields = line.split()
            if not fields or fields[0] != "00":
                continue
            if len(fields) < 11:
                raise ValueError("unsupported IMILAST record with fewer than 11 fields")
            track_id = int(fields[1])
            time = _parse_time(fields[3])
            lon = float(fields[8])
            lat = float(fields[9])
            value = float(fields[10])
            if variable_name.lower() in {"msl", "slp", "pnm", "pres"}:
                value *= 100.0
            elif variable_name.lower() in {"vo", "vort", "vorticity"}:
                value *= 1.0e-5
            track_points.setdefault(track_id, []).append((time, lat, lon, value))

    effective_mode = resolve_mode(variable_name, mode)
    units = {variable_name: canonical_unit_for(variable_name) or "1"}
    builder = _TracksBuilder(TracksMetadata(variable_name, effective_mode, units))
    for track_id, points in track_points.items():
        builder.add_track(
            track_id,
            [point[0] for point in points],
            [point[1] for point in points],
            [point[2] for point in points],
            {variable_name: [point[3] for point in points]},
        )
    return builder.finish()


def write_imilast(
    tracks: Tracks,
    outfile: str | Path,
    decimal_places: int = 6,
) -> None:
    """Write packed trajectories in the supported IMILAST text subset."""
    variable_name = tracks.primary_var
    if len(tracks) and variable_name not in tracks.variables:
        raise ValueError("IMILAST writing requires the primary variable column")
    values = tracks.variables.get(variable_name, np.empty(0, dtype=np.float64))
    multiplier = 0.01 if variable_name.lower() in {"msl", "slp", "pnm", "pres"} else 1.0
    if variable_name.lower() in {"vo", "vort", "vorticity"}:
        multiplier = 1.0e5
    with Path(outfile).open("w", encoding="utf-8", newline="") as output:
        output.write(
            "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,LongE,LatN,"
            f"{variable_name.upper()}\n"
        )
        for track in tracks:
            output.write(f"90 {track.track_id} {len(track)}\n")
            for point_index, center in enumerate(track, start=1):
                dt = decode_time_values([cast(int, center.time)])[0]
                date_i10 = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"
                value = values[track.point_slice][point_index - 1]
                output.write(
                    f"00 {track.track_id} {point_index} {date_i10} {dt.year} "
                    f"{dt.month:02d} {dt.day:02d} {dt.hour:02d} "
                    f"{center.lon: .2f} {center.lat: .2f} "
                    f"{float(value) * multiplier:.{decimal_places}f}\n"
                )
