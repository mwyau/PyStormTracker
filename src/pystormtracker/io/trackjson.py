"""TrackJSON/1.0 typed serialization for packed storm trajectories."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Literal, cast

import msgspec
import numpy as np
from numpy.typing import NDArray

from ..models.geo import SpatialBounds, geod_dist_km, minimal_longitude_interval
from ..models.time import (
    CANONICAL_TIME_UNITS,
    INT64_MIN,
    MAX_SAFE_JSON_INTEGER,
    PROLEPTIC_GREGORIAN,
    Calendar,
    CanonicalTimeUnits,
)
from ..models.tracks import ProcessingStep, Tracks, TracksMetadata

TRACKJSON_FORMAT: Final[str] = "TrackJSON/1.0"
TRACKJSON_STATS_VERSION: Final = 1
TRACKJSON_SCHEMA_RESOURCE: Final[str] = "trackjson.schema.json"

__all__ = [
    "TrackJSONBounds",
    "TrackJSONData",
    "TrackJSONDocument",
    "TrackJSONIndex",
    "TrackJSONMetadata",
    "TrackJSONProcessingStep",
    "TrackJSONStats",
    "TrackJSONTime",
    "compute_trackjson_stats",
    "encode_trackjson",
    "read_trackjson",
    "write_trackjson",
]

_INT64_MAX = 2**63 - 1
_STATS_RTOL = 1.0e-9
_STATS_ATOL = 1.0e-9
_JSON_SIGNIFICANT_DIGITS = 15

NonemptyString = Annotated[
    str, msgspec.Meta(min_length=1, description="A non-empty string.")
]
Latitude = Annotated[
    float,
    msgspec.Meta(ge=-90.0, le=90.0, description="Latitude in degrees north."),
]
Longitude = Annotated[
    float,
    msgspec.Meta(
        ge=-180.0,
        lt=180.0,
        description="Signed longitude in degrees east, normalized to [-180, 180).",
    ),
]
BoundLongitude = Annotated[
    float,
    msgspec.Meta(
        ge=-180.0,
        le=180.0,
        description="Longitude interval edge in degrees east.",
    ),
]
NonnegativeInteger = Annotated[
    int, msgspec.Meta(ge=0, description="A nonnegative integer.")
]
SignedInteger = Annotated[
    int,
    msgspec.Meta(
        ge=-(2**63),
        le=_INT64_MAX,
        description="A signed 64-bit integer.",
    ),
]
PositiveInteger = Annotated[int, msgspec.Meta(gt=0, description="A positive integer.")]
NonnegativeFloat = Annotated[
    float, msgspec.Meta(ge=0.0, description="A finite nonnegative value.")
]
TimeMillisecond = Annotated[
    int,
    msgspec.Meta(
        ge=-MAX_SAFE_JSON_INTEGER,
        le=MAX_SAFE_JSON_INTEGER,
        description="A signed CF millisecond offset in the JSON safe-integer range.",
    ),
]
FiniteNumber = Annotated[float, msgspec.Meta(description="A finite JSON number.")]
ProcessingScalar = str | int | float | bool | None

STAT_INT_FIELDS: tuple[str, ...] = ("point_count",)
STAT_REQUIRED_TIME_FIELDS: tuple[str, ...] = ("start_time", "end_time")
STAT_OPTIONAL_TIME_FIELDS: tuple[str, ...] = ("peak_time",)
STAT_BOOL_FIELDS: tuple[str, ...] = ("antimeridian_wrap",)
STAT_REQUIRED_FLOAT_FIELDS: tuple[str, ...] = (
    "duration_hours",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "south_lat",
    "north_lat",
    "west_lon",
    "east_lon",
    "path_length_km",
    "displacement_km",
)
STAT_OPTIONAL_FLOAT_FIELDS: tuple[str, ...] = (
    "peak_lat",
    "peak_lon",
    "peak_value",
)
STAT_ARRAY_FIELDS: tuple[str, ...] = (
    STAT_INT_FIELDS
    + STAT_REQUIRED_TIME_FIELDS
    + STAT_OPTIONAL_TIME_FIELDS
    + STAT_BOOL_FIELDS
    + STAT_REQUIRED_FLOAT_FIELDS
    + STAT_OPTIONAL_FLOAT_FIELDS
)


class TrackJSONBounds(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Optional declared spatial domain; longitude edges may cross 180 degrees."""

    south: Latitude
    north: Latitude
    west: BoundLongitude
    east: BoundLongitude


class TrackJSONTime(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Required canonical numeric-time units and calendar."""

    units: CanonicalTimeUnits
    calendar: Calendar


class TrackJSONProcessingStep(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True, omit_defaults=True
):
    """One preprocessing operation recorded independently of variable names."""

    operation: NonemptyString
    enabled: bool
    parameters: dict[NonemptyString, ProcessingScalar] = msgspec.field(
        default_factory=dict
    )


class TrackJSONMetadata(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Metadata needed to interpret canonical trajectory columns."""

    primary_var: NonemptyString
    mode: Literal["min", "max"]
    units: dict[NonemptyString, NonemptyString]
    time: TrackJSONTime
    bounds: TrackJSONBounds | msgspec.UnsetType = msgspec.UNSET
    processing: list[TrackJSONProcessingStep] = msgspec.field(default_factory=list)


class TrackJSONIndex(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Packed track IDs and complete half-open offset boundaries."""

    ids: list[SignedInteger]
    offsets: list[NonnegativeInteger]


class TrackJSONData(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Aligned packed point columns and nullable meteorological variables."""

    times: list[TimeMillisecond]
    lats: list[Latitude]
    lons: list[Longitude]
    variables: dict[NonemptyString, list[FiniteNumber | None]]


class TrackJSONStats(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """Optional columnar derived statistics; never part of the core model."""

    version: Literal[1]
    point_count: list[PositiveInteger]
    start_time: list[TimeMillisecond]
    end_time: list[TimeMillisecond]
    duration_hours: list[NonnegativeFloat]
    start_lat: list[Latitude]
    start_lon: list[Longitude]
    end_lat: list[Latitude]
    end_lon: list[Longitude]
    south_lat: list[Latitude]
    north_lat: list[Latitude]
    west_lon: list[Longitude]
    east_lon: list[Longitude]
    antimeridian_wrap: list[bool]
    peak_time: list[TimeMillisecond | None]
    peak_lat: list[Latitude | None]
    peak_lon: list[Longitude | None]
    peak_value: list[FiniteNumber | None]
    path_length_km: list[NonnegativeFloat]
    displacement_km: list[NonnegativeFloat]


class TrackJSONDocument(msgspec.Struct, forbid_unknown_fields=True, omit_defaults=True):
    """The complete TrackJSON/1.0 document."""

    format: Literal["TrackJSON/1.0"]
    metadata: TrackJSONMetadata
    index: TrackJSONIndex
    data: TrackJSONData
    stats: TrackJSONStats | msgspec.UnsetType = msgspec.UNSET


_TRACKJSON_ENCODER = msgspec.json.Encoder()
_TRACKJSON_DECODER = msgspec.json.Decoder(TrackJSONDocument)


def _clean_json_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite float, got {value!r}")
    cleaned = float(format(value, f".{_JSON_SIGNIFICANT_DIGITS}g"))
    return 0.0 if cleaned == 0.0 else cleaned


def _clean_processing_value(value: object) -> object:
    if isinstance(value, float):
        return _clean_json_float(value)
    return value


def _clean_float_values(values: list[float]) -> list[float]:
    return [_clean_json_float(value) for value in values]


def _clean_optional_float_values(values: list[float | None]) -> list[float | None]:
    return [None if value is None else _clean_json_float(value) for value in values]


def _path_error(exc: Exception) -> ValueError:
    return ValueError(f"Invalid TrackJSON document: {exc}")


def _validate_bounds(bounds: TrackJSONBounds) -> None:
    try:
        SpatialBounds(bounds.south, bounds.north, bounds.west, bounds.east)
    except ValueError as exc:
        raise ValueError(f"$.metadata.bounds: {exc}") from exc


def _validate_trackjson_semantics(document: TrackJSONDocument) -> None:
    """Validate canonical relationships; optional stats are only typed here."""
    ids = document.index.ids
    offsets = document.index.offsets
    times = document.data.times
    lats = document.data.lats
    lons = document.data.lons
    variables = document.data.variables
    n_tracks = len(ids)
    n_points = len(times)

    if any(isinstance(value, bool) for value in ids):
        raise ValueError("$.index.ids: Boolean IDs are not valid integers")
    if len(offsets) != n_tracks + 1:
        raise ValueError("$.index.offsets: length must equal len(ids) + 1")
    if not offsets or offsets[0] != 0:
        raise ValueError("$.index.offsets[0]: offsets must start at zero")
    if offsets[-1] != n_points:
        raise ValueError("$.index.offsets[-1]: final offset must equal point count")
    if len(set(ids)) != len(ids):
        raise ValueError("$.index.ids: track IDs must be unique")
    if n_tracks and any(right <= left for left, right in pairwise(offsets)):
        raise ValueError("$.index.offsets: offsets must be strictly increasing")
    if len(lats) != n_points:
        raise ValueError("$.data.lats: length must equal the time column length")
    if len(lons) != n_points:
        raise ValueError("$.data.lons: length must equal the time column length")
    if any(len(values) != n_points for values in variables.values()):
        raise ValueError("$.data.variables: every variable column must have length N")
    if any(not np.isfinite(value) for value in lats):
        raise ValueError("$.data.lats: coordinates must be finite")
    if any(not np.isfinite(value) for value in lons):
        raise ValueError("$.data.lons: coordinates must be finite")
    if any(
        value < -MAX_SAFE_JSON_INTEGER or value > MAX_SAFE_JSON_INTEGER
        for value in times
    ):
        raise ValueError(
            "$.data.times: values must be safe signed millisecond integers"
        )
    for name, values in variables.items():
        if any(value is not None and not np.isfinite(value) for value in values):
            raise ValueError(f"$.data.variables.{name}: values must be finite or null")
    for index in range(n_tracks):
        start = offsets[index]
        stop = offsets[index + 1]
        if any(right <= left for left, right in pairwise(times[start:stop])):
            raise ValueError(
                f"$.data.times[{start}:{stop}]: times must be strictly increasing"
            )

    metadata = document.metadata
    if metadata.time.units != CANONICAL_TIME_UNITS:
        raise ValueError("$.metadata.time.units: value must equal CANONICAL_TIME_UNITS")
    if metadata.time.calendar != PROLEPTIC_GREGORIAN:
        raise ValueError(
            "$.metadata.time.calendar: value must equal PROLEPTIC_GREGORIAN"
        )
    if set(variables) != set(metadata.units):
        raise ValueError("$.metadata.units: keys must exactly match data.variables")
    if metadata.primary_var not in variables:
        raise ValueError(
            "$.metadata.primary_var: variable is missing from data.variables"
        )
    if metadata.bounds is not msgspec.UNSET:
        _validate_bounds(metadata.bounds)


def _validate_stats(stats: TrackJSONStats, document: TrackJSONDocument) -> None:
    """Validate derived-stat alignment and relationships for explicit verification."""
    n_tracks = len(document.index.ids)
    for name in STAT_ARRAY_FIELDS:
        if len(getattr(stats, name)) != n_tracks:
            raise ValueError(f"$.stats.{name}: length must equal T")
    offsets = document.index.offsets
    times = document.data.times
    lats = document.data.lats
    lons = document.data.lons
    if stats.point_count != [right - left for left, right in pairwise(offsets)]:
        raise ValueError("$.stats.point_count: values must equal diff(index.offsets)")
    if stats.start_time != [times[offset] for offset in offsets[:-1]]:
        raise ValueError("$.stats.start_time: values must match first point times")
    if stats.end_time != [times[offset - 1] for offset in offsets[1:]]:
        raise ValueError("$.stats.end_time: values must match final point times")
    expected_start_lat = [lats[offset] for offset in offsets[:-1]]
    if not np.allclose(
        stats.start_lat,
        expected_start_lat,
        rtol=_STATS_RTOL,
        atol=_STATS_ATOL,
    ):
        raise ValueError("$.stats.start_lat: values must match first point latitudes")
    expected_end_lat = [lats[offset - 1] for offset in offsets[1:]]
    if not np.allclose(
        stats.end_lat,
        expected_end_lat,
        rtol=_STATS_RTOL,
        atol=_STATS_ATOL,
    ):
        raise ValueError("$.stats.end_lat: values must match final point latitudes")
    expected_start_lon = [lons[offset] for offset in offsets[:-1]]
    if not np.allclose(
        stats.start_lon,
        expected_start_lon,
        rtol=_STATS_RTOL,
        atol=_STATS_ATOL,
    ):
        raise ValueError("$.stats.start_lon: values must match first point longitudes")
    expected_end_lon = [lons[offset - 1] for offset in offsets[1:]]
    if not np.allclose(
        stats.end_lon,
        expected_end_lon,
        rtol=_STATS_RTOL,
        atol=_STATS_ATOL,
    ):
        raise ValueError("$.stats.end_lon: values must match final point longitudes")
    if any(
        south > north
        for south, north in zip(stats.south_lat, stats.north_lat, strict=True)
    ):
        raise ValueError("$.stats.south_lat: must not exceed north_lat")
    for index, values in enumerate(
        zip(
            stats.peak_time,
            stats.peak_lat,
            stats.peak_lon,
            stats.peak_value,
            strict=True,
        )
    ):
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError(
                f"$.stats.peak_time[{index}]: peak fields must be all present or "
                "all null"
            )
        peak_time = values[0]
        if peak_time is not None and not (
            stats.start_time[index] <= peak_time <= stats.end_time[index]
        ):
            raise ValueError(
                f"$.stats.peak_time[{index}]: time is outside track interval"
            )


def _decode(path: str | Path) -> TrackJSONDocument:
    try:
        document = _TRACKJSON_DECODER.decode(Path(path).read_bytes())
    except OSError as exc:
        raise ValueError(f"Unable to read TrackJSON file {path}: {exc}") from exc
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise _path_error(exc) from exc
    _validate_trackjson_semantics(document)
    return document


def _nullable_float_array(values: list[float | None]) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for index, value in enumerate(values):
        if value is not None:
            if not np.isfinite(value):
                raise ValueError("TrackJSON variable values must be finite or null")
            result[index] = value
    return result


def _time_view(values: list[int], name: str) -> NDArray[np.int64]:
    raw = np.asarray(values, dtype=np.int64)
    if np.any((raw < -MAX_SAFE_JSON_INTEGER) | (raw > MAX_SAFE_JSON_INTEGER)):
        raise ValueError(f"TrackJSON {name} must contain safe millisecond values")
    return raw


def _integer_list(values: object, name: str) -> list[int]:
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or raw.dtype.kind not in ("i", "u"):
        raise ValueError(f"{name} must contain integer values")
    if raw.size and (np.any(raw < INT64_MIN) or np.any(raw > _INT64_MAX)):
        raise ValueError(f"{name} values must fit signed int64")
    return [int(value) for value in raw.tolist()]


def _float_list(values: object, name: str) -> list[float]:
    raw = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(raw)):
        raise ValueError(f"{name} must contain finite floats")
    return [_clean_json_float(float(value)) for value in raw.tolist()]


def _nullable_float_list(values: object, name: str) -> list[float | None]:
    raw = np.asarray(values, dtype=np.float64)
    if np.any(np.isinf(raw)):
        raise ValueError(f"{name} must contain finite floats or NaN")
    return [
        None if np.isnan(value) else _clean_json_float(float(value))
        for value in raw.tolist()
    ]


def _time_list(values: object, name: str) -> list[int]:
    raw = np.asarray(values, dtype=np.int64)
    if np.any(
        (raw < -MAX_SAFE_JSON_INTEGER)
        | (raw > MAX_SAFE_JSON_INTEGER)
        | (raw == INT64_MIN)
    ):
        raise ValueError(f"{name} values must fit the safe integer range")
    return [int(value) for value in raw.tolist()]


def _nullable_time_list(values: object, name: str) -> list[int | None]:
    raw = np.asarray(values, dtype=np.int64)
    if np.any(
        (raw != INT64_MIN)
        & ((raw < -MAX_SAFE_JSON_INTEGER) | (raw > MAX_SAFE_JSON_INTEGER))
    ):
        raise ValueError(f"{name} values must fit the safe integer range")
    return [None if value == INT64_MIN else int(value) for value in raw.tolist()]


def _processing_to_wire(
    processing: tuple[ProcessingStep, ...],
) -> list[TrackJSONProcessingStep]:
    return [
        TrackJSONProcessingStep(
            operation=step.operation,
            enabled=step.enabled,
            parameters={
                name: cast(
                    str | int | float | bool | None,
                    _clean_processing_value(value),
                )
                for name, value in step.parameters.items()
            },
        )
        for step in processing
    ]


def _processing_from_wire(
    processing: list[TrackJSONProcessingStep],
) -> tuple[ProcessingStep, ...]:
    return tuple(
        ProcessingStep(
            operation=step.operation,
            enabled=step.enabled,
            parameters=step.parameters,
        )
        for step in processing
    )


def compute_trackjson_stats(tracks: Tracks) -> TrackJSONStats:
    """Compute the complete optional TrackJSON statistics cache."""
    point_count: list[int] = []
    start_time: list[int] = []
    end_time: list[int] = []
    duration_hours: list[float] = []
    start_lat: list[float] = []
    start_lon: list[float] = []
    end_lat: list[float] = []
    end_lon: list[float] = []
    south_lat: list[float] = []
    north_lat: list[float] = []
    west_lon: list[float] = []
    east_lon: list[float] = []
    antimeridian_wrap: list[bool] = []
    peak_time: list[int | None] = []
    peak_lat: list[float | None] = []
    peak_lon: list[float | None] = []
    peak_value: list[float | None] = []
    path_length_km: list[float] = []
    displacement_km: list[float] = []
    primary_values = tracks.variables[tracks.primary_var]

    for index in range(len(tracks)):
        start = int(tracks.offsets[index])
        stop = int(tracks.offsets[index + 1])
        times = tracks.times[start:stop]
        lats = tracks.lats[start:stop]
        lons = tracks.lons[start:stop]
        values = primary_values[start:stop]
        point_count.append(stop - start)
        start_time.append(int(times[0]))
        end_time.append(int(times[-1]))
        duration_hours.append((int(times[-1]) - int(times[0])) / 3_600_000.0)
        start_lat.append(float(lats[0]))
        start_lon.append(float(lons[0]))
        end_lat.append(float(lats[-1]))
        end_lon.append(float(lons[-1]))
        south_lat.append(float(np.min(lats)))
        north_lat.append(float(np.max(lats)))
        west, east, crossing = minimal_longitude_interval(lons)
        west_lon.append(west)
        east_lon.append(east)
        antimeridian_wrap.append(crossing)
        path_length_km.append(
            float(
                sum(
                    geod_dist_km(
                        float(lats[pos]),
                        float(lons[pos]),
                        float(lats[pos + 1]),
                        float(lons[pos + 1]),
                    )
                    for pos in range(len(lats) - 1)
                )
            )
        )
        displacement_km.append(
            float(geod_dist_km(lats[0], lons[0], lats[-1], lons[-1]))
        )
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite):
            relative = int(
                finite[
                    np.argmin(values[finite])
                    if tracks.mode == "min"
                    else np.argmax(values[finite])
                ]
            )
            peak_index = start + relative
            peak_time.append(int(tracks.times[peak_index]))
            peak_lat.append(float(tracks.lats[peak_index]))
            peak_lon.append(float(tracks.lons[peak_index]))
            peak_value.append(float(values[relative]))
        else:
            peak_time.append(None)
            peak_lat.append(None)
            peak_lon.append(None)
            peak_value.append(None)

    return TrackJSONStats(
        version=TRACKJSON_STATS_VERSION,
        point_count=point_count,
        start_time=start_time,
        end_time=end_time,
        duration_hours=_clean_float_values(duration_hours),
        start_lat=_clean_float_values(start_lat),
        start_lon=_clean_float_values(start_lon),
        end_lat=_clean_float_values(end_lat),
        end_lon=_clean_float_values(end_lon),
        south_lat=_clean_float_values(south_lat),
        north_lat=_clean_float_values(north_lat),
        west_lon=_clean_float_values(west_lon),
        east_lon=_clean_float_values(east_lon),
        antimeridian_wrap=antimeridian_wrap,
        peak_time=peak_time,
        peak_lat=_clean_optional_float_values(peak_lat),
        peak_lon=_clean_optional_float_values(peak_lon),
        peak_value=_clean_optional_float_values(peak_value),
        path_length_km=_clean_float_values(path_length_km),
        displacement_km=_clean_float_values(displacement_km),
    )


def _compare_stats(actual: TrackJSONStats, expected: TrackJSONStats) -> None:
    for name in (
        "version",
        *STAT_INT_FIELDS,
        *STAT_BOOL_FIELDS,
        *STAT_REQUIRED_TIME_FIELDS,
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(f"TrackJSON stats {name} failed verification")
    for name in STAT_OPTIONAL_TIME_FIELDS:
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(f"TrackJSON stats {name} failed verification")
    for name in STAT_REQUIRED_FLOAT_FIELDS + STAT_OPTIONAL_FLOAT_FIELDS:
        if not np.allclose(
            np.asarray(getattr(actual, name), dtype=np.float64),
            np.asarray(getattr(expected, name), dtype=np.float64),
            rtol=_STATS_RTOL,
            atol=_STATS_ATOL,
            equal_nan=True,
        ):
            raise ValueError(f"TrackJSON stats {name} failed verification")


def read_trackjson(
    path: str | Path,
    *,
    verify_stats: bool = False,
) -> Tracks:
    """Read canonical TrackJSON data; optional wire stats are never retained."""
    document = _decode(path)
    bounds = (
        SpatialBounds(
            document.metadata.bounds.south,
            document.metadata.bounds.north,
            document.metadata.bounds.west,
            document.metadata.bounds.east,
        )
        if document.metadata.bounds is not msgspec.UNSET
        else None
    )
    metadata = TracksMetadata(
        primary_var=document.metadata.primary_var,
        mode=document.metadata.mode,
        units=document.metadata.units,
        bounds=bounds,
        processing=_processing_from_wire(document.metadata.processing),
    )
    tracks = Tracks(
        ids=np.asarray(document.index.ids, dtype=np.int64),
        offsets=np.asarray(document.index.offsets, dtype=np.int64),
        times=_time_view(document.data.times, "times"),
        lats=np.asarray(document.data.lats, dtype=np.float64),
        lons=np.asarray(document.data.lons, dtype=np.float64),
        variables={
            name: _nullable_float_array(values)
            for name, values in document.data.variables.items()
        },
        metadata=metadata,
    )
    if verify_stats and document.stats is not msgspec.UNSET:
        _validate_stats(document.stats, document)
        _compare_stats(document.stats, compute_trackjson_stats(tracks))
    return tracks


def encode_trackjson(tracks: Tracks, *, include_stats: bool = False) -> bytes:
    """Encode canonical packed tracks, optionally adding a fresh wire stats cache."""
    bounds: TrackJSONBounds | msgspec.UnsetType = msgspec.UNSET
    if tracks.metadata.bounds is not None:
        declared = tracks.metadata.bounds
        bounds = TrackJSONBounds(
            south=_clean_json_float(declared.south),
            north=_clean_json_float(declared.north),
            west=_clean_json_float(declared.west),
            east=_clean_json_float(declared.east),
        )
    stats: TrackJSONStats | msgspec.UnsetType = msgspec.UNSET
    if include_stats:
        stats = compute_trackjson_stats(tracks)
    document = TrackJSONDocument(
        format=cast(Literal["TrackJSON/1.0"], TRACKJSON_FORMAT),
        metadata=TrackJSONMetadata(
            primary_var=tracks.primary_var,
            mode=tracks.mode,
            units=dict(tracks.units),
            time=TrackJSONTime(
                units=cast(
                    CanonicalTimeUnits,
                    CANONICAL_TIME_UNITS,
                ),
                calendar=PROLEPTIC_GREGORIAN,
            ),
            bounds=bounds,
            processing=_processing_to_wire(tracks.metadata.processing),
        ),
        index=TrackJSONIndex(
            ids=_integer_list(tracks.ids, "index.ids"),
            offsets=_integer_list(tracks.offsets, "index.offsets"),
        ),
        data=TrackJSONData(
            times=_time_list(tracks.times, "data.times"),
            lats=_float_list(tracks.lats, "data.lats"),
            lons=_float_list(tracks.lons, "data.lons"),
            variables={
                name: _nullable_float_list(values, f"data.variables.{name}")
                for name, values in tracks.variables.items()
            },
        ),
        stats=stats,
    )
    return _TRACKJSON_ENCODER.encode(document)


def write_trackjson(
    tracks: Tracks,
    path: str | Path,
    *,
    include_stats: bool = False,
) -> None:
    """Write TrackJSON bytes without an intermediate file."""
    Path(path).write_bytes(encode_trackjson(tracks, include_stats=include_stats))
