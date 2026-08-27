"""Canonical CF time handling for packed tracks.

The packed model uses one representation: signed int64 milliseconds since the
Unix epoch under the proleptic Gregorian calendar. Core calendar conversion uses
standard datetime and NumPy operations; optional ``cftime`` inputs are handled
lazily when available.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final, Literal, Protocol

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._cftime import (
    encode_cftime_objects,
    get_cftime_calendar,
    is_cftime_available,
    is_cftime_datetime,
    require_cftime,
)

type Calendar = Literal["proleptic_gregorian"]
PROLEPTIC_GREGORIAN: Final[Calendar] = "proleptic_gregorian"
_GREGORIAN_REFORM_DATE: Final[date] = date(1582, 10, 15)
type CanonicalTimeUnits = Literal["milliseconds since 1970-01-01 00:00:00"]
CANONICAL_TIME_UNITS: Final[CanonicalTimeUnits] = (
    "milliseconds since 1970-01-01 00:00:00"
)

_EPOCH: Final[datetime] = datetime(1970, 1, 1, 0, 0, 0)


class CftimeDateTime(Protocol):
    """Structural type for the optional ``cftime.datetime`` family."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    calendar: str


type TimePoint = datetime | np.datetime64 | int | CftimeDateTime
type TimeInput = TimePoint | str


@dataclass(slots=True)
class TimeRange:
    """Metadata used by detector orchestration to select an input interval."""

    start: TimePoint | None
    end: TimePoint | None
    step: np.timedelta64 | None = None


_SUPPORTED_SOURCE_CALENDARS = {
    "proleptic_gregorian",
    "standard",
    "gregorian",
}
_UNSUPPORTED_CALENDAR_MESSAGE = (
    "Unsupported CF calendar {!r}; PyStormTracker currently supports only "
    "proleptic_gregorian and standard/gregorian dates. Broader CF-calendar "
    "support is deferred."
)


def normalize_calendar(value: str | None) -> Calendar:
    """Normalize a supported source name to the sole packed calendar."""
    normalized = PROLEPTIC_GREGORIAN if value is None else value.strip().lower()
    if normalized in _SUPPORTED_SOURCE_CALENDARS:
        return PROLEPTIC_GREGORIAN
    raise ValueError(_UNSUPPORTED_CALENDAR_MESSAGE.format(value))


def _declared_calendar(value: str | None) -> str:
    normalized = PROLEPTIC_GREGORIAN if value is None else value.strip().lower()
    if normalized not in _SUPPORTED_SOURCE_CALENDARS:
        raise ValueError(_UNSUPPORTED_CALENDAR_MESSAGE.format(value))
    return normalized


def _check_standard_dates(values: Sequence[object]) -> None:
    for value in values:
        year = getattr(value, "year", None)
        month = getattr(value, "month", None)
        day = getattr(value, "day", None)
        if year is None or month is None or day is None:
            raise ValueError("CF time decoding did not produce calendar dates")
        if date(int(year), int(month), int(day)) < _GREGORIAN_REFORM_DATE:
            raise ValueError(
                "explicit standard/gregorian dates before "
                f"{_GREGORIAN_REFORM_DATE.isoformat()} cannot be converted because "
                "mixed Julian/Gregorian calendar conversion is not implemented"
            )


def _validate_int64(values: object, name: str = "times") -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or raw.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must contain integer millisecond values")
    if raw.dtype.kind == "f" and np.any(~np.isfinite(raw) | (raw != np.floor(raw))):
        raise ValueError(f"{name} must contain integer millisecond values")
    if raw.size and (
        np.any(raw < np.iinfo(np.int64).min) or np.any(raw > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must fit signed int64")
    return np.asarray(raw, dtype=np.int64)


def _validate_millisecond_precision(values: Sequence[object]) -> None:
    for index, value in enumerate(values):
        microsecond = getattr(value, "microsecond", 0)
        if int(microsecond) % 1000:
            raise ValueError(f"datetime at index {index} has sub-millisecond precision")
        if isinstance(value, datetime) and value.tzinfo is not None:
            raise ValueError("calendar datetimes must be timezone-naive")


def _datetime_to_ms(dt: datetime | date) -> int:
    if isinstance(dt, datetime):
        if dt.tzinfo is not None:
            raise ValueError("calendar datetimes must be timezone-naive")
        if dt.microsecond % 1000 != 0:
            raise ValueError("datetime has sub-millisecond precision")
        delta = dt - _EPOCH
        return (delta.days * 86400 + delta.seconds) * 1000 + dt.microsecond // 1000
    if isinstance(dt, date):
        full_dt = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        delta = full_dt - _EPOCH
        return (delta.days * 86400 + delta.seconds) * 1000
    raise ValueError(f"expected datetime or date, got {type(dt)}")


def _encode_datetime_sequence(
    values: Sequence[object],
    *,
    source_calendar: str,
) -> NDArray[np.int64]:
    if not values:
        return np.empty(0, dtype=np.int64)
    _validate_millisecond_precision(values)
    if source_calendar in ("standard", "gregorian"):
        _check_standard_dates(values)

    has_cftime = any(is_cftime_datetime(v) for v in values)
    if has_cftime:
        return encode_cftime_objects(
            values,
            canonical_time_units=CANONICAL_TIME_UNITS,
            proleptic_gregorian=PROLEPTIC_GREGORIAN,
        )

    result = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if isinstance(v, (datetime, date)):
            result[i] = _datetime_to_ms(v)
        else:
            raise TypeError(f"unsupported datetime object at index {i}: {type(v)}")
    return result


def encode_cf_datetimes(
    values: Sequence[datetime | object],
    *,
    calendar: str,
) -> NDArray[np.int64]:
    """Encode supported Python/cftime calendar values to packed milliseconds."""
    declared = _declared_calendar(calendar)
    return _encode_datetime_sequence(values, source_calendar=declared)


def _numpy_datetime_values(values: NDArray[np.datetime64]) -> NDArray[np.int64]:
    if np.any(np.isnat(values)):
        raise ValueError("times must not contain NaT")
    converted = values.astype("datetime64[ms]")
    if np.any(values != converted.astype(values.dtype)):
        raise ValueError("times must have millisecond precision")
    return _validate_int64(converted.view(np.int64))


def encode_time_values(values: object) -> NDArray[np.int64]:
    """Encode numeric, NumPy datetime, Python datetime, or cftime values."""
    raw = np.asarray(values)
    if raw.dtype.kind in ("b", "i", "u", "f"):
        return _validate_int64(raw)
    if raw.dtype.kind == "M":
        return _numpy_datetime_values(raw)
    if raw.dtype.kind in ("U", "S"):
        try:
            return _numpy_datetime_values(raw.astype("datetime64[ms]"))
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("times must contain valid datetime values") from exc
    if raw.dtype.kind == "O":
        values_tuple = tuple(raw.tolist())
        if all(
            isinstance(value, (datetime, date)) or is_cftime_datetime(value)
            for value in values_tuple
        ):
            calendars = {
                get_cftime_calendar(value, PROLEPTIC_GREGORIAN)
                for value in values_tuple
                if is_cftime_datetime(value)
            }
            if len(calendars) > 1:
                raise ValueError("time values must use one calendar")
            calendar = next(iter(calendars), PROLEPTIC_GREGORIAN)
            return encode_cf_datetimes(values_tuple, calendar=calendar)
        # Check if items look like cftime datetime without cftime being installed
        for item in values_tuple:
            if hasattr(item, "calendar") and not is_cftime_available():
                require_cftime()
    raise ValueError("times must contain integer milliseconds or supported datetimes")


_CF_UNITS_REGEX = re.compile(
    r"^(days|day|d|hours|hour|hr|hrs|h|minutes|minute|min|mins|m|seconds|second|sec|secs|s|milliseconds|millisecond|msec|ms)\s+since\s+(.+)$",
    re.IGNORECASE,
)

_UNIT_TO_MS = {
    "d": 86400000.0,
    "day": 86400000.0,
    "days": 86400000.0,
    "h": 3600000.0,
    "hr": 3600000.0,
    "hrs": 3600000.0,
    "hour": 3600000.0,
    "hours": 3600000.0,
    "m": 60000.0,
    "min": 60000.0,
    "mins": 60000.0,
    "minute": 60000.0,
    "minutes": 60000.0,
    "s": 1000.0,
    "sec": 1000.0,
    "secs": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "ms": 1.0,
    "msec": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
}


def encode_numeric_time_values(
    values: object,
    *,
    units: str,
    calendar: str | None = None,
) -> NDArray[np.int64]:
    """Decode source numeric CF values and re-encode canonical milliseconds."""
    declared = _declared_calendar(calendar)
    numeric = np.asarray(values)
    if numeric.dtype.kind not in ("i", "u", "f"):
        raise ValueError("CF time values must be numeric")
    if np.any(~np.isfinite(numeric.astype(np.float64))):
        raise ValueError("CF time values must be finite")

    match = _CF_UNITS_REGEX.match(units.strip())
    if match is not None:
        unit_name = match.group(1).lower()
        origin_str = match.group(2).strip()
        scale = _UNIT_TO_MS[unit_name]
        try:
            origin_dt = datetime.fromisoformat(origin_str.replace(" ", "T"))
        except ValueError:
            # Fall back to np.datetime64 or cftime if available
            origin_dt = None

        if origin_dt is not None:
            origin_ms = _datetime_to_ms(origin_dt)
            numeric_f = np.asarray(numeric, dtype=np.float64)
            offset_ms = numeric_f * scale
            rounded_offset = np.rint(offset_ms)
            if np.any(np.abs(offset_ms - rounded_offset) > 1e-3):
                raise ValueError("CF time values must resolve to integer milliseconds")
            result_ms = np.asarray(origin_ms + rounded_offset, dtype=np.int64)

            if declared in ("standard", "gregorian"):
                decoded_dts = decode_time_values(result_ms)
                _check_standard_dates(decoded_dts)
            return _validate_int64(result_ms)

    # If units format was not directly parseable by simple ISO parser, try lazy cftime
    if is_cftime_available():
        import cftime

        decoded = cftime.num2date(
            numeric,
            units,
            calendar=declared,
            only_use_cftime_datetimes=True,
            only_use_python_datetimes=False,
        )
        return _encode_datetime_sequence(tuple(decoded), source_calendar=declared)

    raise ValueError(f"cannot parse CF time units {units!r}")


def decode_time_values(
    values: Sequence[int] | NDArray[np.int64],
) -> tuple[datetime, ...]:
    """Decode canonical packed milliseconds to Python datetime objects."""
    safe_values = _validate_int64(values)
    return tuple(
        _EPOCH + timedelta(milliseconds=int(val)) for val in safe_values.tolist()
    )


def format_time(value: int) -> str:
    """Format one canonical packed time without implying UTC."""
    decoded = decode_time_values([value])[0]
    result = (
        f"{decoded.year:04d}-{decoded.month:02d}-{decoded.day:02d} "
        f"{decoded.hour:02d}:{decoded.minute:02d}:{decoded.second:02d}"
    )
    if decoded.microsecond:
        result += f".{decoded.microsecond:06d}".rstrip("0")
    return result


def infer_calendar(
    values: object,
    *,
    attrs: Mapping[str, object] | None = None,
    encoding: Mapping[str, object] | None = None,
) -> Calendar:
    """Infer the canonical calendar from source metadata and values."""
    metadata = dict(encoding or {})
    metadata.update(attrs or {})
    declared = metadata.get("calendar")
    if declared is None:
        raw = np.asarray(values)
        if raw.dtype.kind == "O":
            calendars = {
                get_cftime_calendar(value, PROLEPTIC_GREGORIAN)
                for value in raw.tolist()
                if is_cftime_datetime(value)
            }
            if len(calendars) > 1:
                raise ValueError("time coordinate contains multiple calendars")
            declared = next(iter(calendars), None)
    normalized_name = _declared_calendar(None if declared is None else str(declared))
    if normalized_name in ("standard", "gregorian"):
        raw_arr = np.asarray(values)
        if raw_arr.dtype.kind == "M":
            dts = decode_time_values(raw_arr.astype("datetime64[ms]").view(np.int64))
            _check_standard_dates(dts)
        elif raw_arr.dtype.kind == "O":
            _check_standard_dates(tuple(raw_arr.tolist()))
    return PROLEPTIC_GREGORIAN


def is_missing_time(value: object) -> bool:
    """Return whether a time-range endpoint is explicitly missing."""
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return value is None


def coerce_time_input(value: TimeInput | None) -> TimePoint | None:
    """Parse ordinary string boundaries for xarray selection."""
    if isinstance(value, str):
        return np.datetime64(value)
    return value


def select_time_range(
    data: xr.DataArray | xr.Dataset,
    *,
    start_time: TimeInput | None,
    end_time: TimeInput | None,
) -> xr.DataArray | xr.Dataset:
    """Select one inclusive range using the normalized time coordinate."""
    if start_time is None and end_time is None:
        return data
    time_name = next(
        (name for name in ("time", "valid_time") if name in data.coords), None
    )
    if time_name is None:
        raise ValueError("dataset has no supported time coordinate")
    return data.sel(
        {time_name: slice(coerce_time_input(start_time), coerce_time_input(end_time))}
    )
