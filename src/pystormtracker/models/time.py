"""Canonical CF time handling for packed tracks.

The packed model uses one representation: signed int64 milliseconds since the
Unix epoch under the proleptic Gregorian calendar. ``cftime`` performs all
source CF conversion; this module only applies the project's supported-calendar
policy and validates the integer representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal, TypeAlias, cast

import cftime
import numpy as np
import xarray as xr
from numpy.typing import NDArray

Calendar: TypeAlias = Literal["proleptic_gregorian"]
PROLEPTIC_GREGORIAN: Final[Calendar] = "proleptic_gregorian"
GREGORIAN_REFORM_DATE: Final[date] = date(1582, 10, 15)
CANONICAL_TIME_UNITS: Final[str] = "milliseconds since 1970-01-01 00:00:00"
CanonicalTimeUnits: TypeAlias = Literal["milliseconds since 1970-01-01 00:00:00"]
MAX_SAFE_JSON_INTEGER: Final[int] = 2**53 - 1
INT64_MIN: Final[int] = -(2**63)
INT64_MAX: Final[int] = 2**63 - 1

TimePoint: TypeAlias = datetime | cftime.datetime | np.datetime64 | int
TimeInput: TypeAlias = TimePoint | str


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
    "proleptic_gregorian and modern standard/gregorian dates. Broader CF-calendar "
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
        if date(int(year), int(month), int(day)) < GREGORIAN_REFORM_DATE:
            raise ValueError(
                "explicit standard/gregorian dates before "
                f"{GREGORIAN_REFORM_DATE.isoformat()} cannot be converted because "
                "mixed Julian/Gregorian calendar conversion is not implemented"
            )


def _validate_int64(values: object, name: str = "times") -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or raw.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must contain integer millisecond values")
    if raw.dtype.kind == "f" and np.any(~np.isfinite(raw) | (raw != np.floor(raw))):
        raise ValueError(f"{name} must contain integer millisecond values")
    if raw.size and (np.any(raw < INT64_MIN) or np.any(raw > INT64_MAX)):
        raise ValueError(f"{name} must fit signed int64")
    return np.asarray(raw, dtype=np.int64)


def _validate_millisecond_precision(values: Sequence[object]) -> None:
    for index, value in enumerate(values):
        microsecond = getattr(value, "microsecond", 0)
        if int(microsecond) % 1000:
            raise ValueError(f"datetime at index {index} has sub-millisecond precision")
        if isinstance(value, datetime) and value.tzinfo is not None:
            raise ValueError("calendar datetimes must be timezone-naive")


def _encode_cftime_values(
    values: Sequence[object],
    *,
    source_calendar: str,
) -> NDArray[np.int64]:
    if not values:
        return np.empty(0, dtype=np.int64)
    _validate_millisecond_precision(values)
    if source_calendar in ("standard", "gregorian"):
        _check_standard_dates(values)
    numeric = np.asarray(
        cftime.date2num(
            list(values),
            CANONICAL_TIME_UNITS,
            calendar=PROLEPTIC_GREGORIAN,
            longdouble=True,
        ),
        dtype=np.longdouble,
    )
    if np.any(~np.isfinite(numeric)):
        raise ValueError("CF time values must be finite")
    rounded = np.rint(numeric)
    if np.any(np.abs(numeric - rounded) > np.longdouble("1e-6")):
        raise ValueError("CF time values must resolve to integer milliseconds")
    result = _validate_int64(rounded)
    decoded = cftime.num2date(
        result.tolist(),
        CANONICAL_TIME_UNITS,
        calendar=PROLEPTIC_GREGORIAN,
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=False,
    )
    round_trip = np.asarray(
        cftime.date2num(
            decoded,
            CANONICAL_TIME_UNITS,
            calendar=PROLEPTIC_GREGORIAN,
            longdouble=True,
        ),
        dtype=np.longdouble,
    )
    if np.any(round_trip != result.astype(np.longdouble)):
        raise ValueError("CF time encoding failed its millisecond round trip")
    return result


def encode_cf_datetimes(
    values: Sequence[datetime | cftime.datetime],
    *,
    calendar: str,
) -> NDArray[np.int64]:
    """Encode supported Python/cftime calendar values to packed milliseconds."""
    declared = _declared_calendar(calendar)
    return _encode_cftime_values(values, source_calendar=declared)


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
            isinstance(value, (datetime, cftime.datetime)) for value in values_tuple
        ):
            calendars = {
                str(getattr(value, "calendar", PROLEPTIC_GREGORIAN)).lower()
                for value in values_tuple
                if isinstance(value, cftime.datetime)
            }
            if len(calendars) > 1:
                raise ValueError("time values must use one calendar")
            calendar = next(iter(calendars), PROLEPTIC_GREGORIAN)
            return encode_cf_datetimes(values_tuple, calendar=calendar)
    raise ValueError("times must contain integer milliseconds or supported datetimes")


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
    decoded = cftime.num2date(
        numeric,
        units,
        calendar=declared,
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=False,
    )
    decoded_values = tuple(decoded)
    return _encode_cftime_values(decoded_values, source_calendar=declared)


def decode_time_values(
    values: Sequence[int] | NDArray[np.int64],
) -> tuple[cftime.datetime, ...]:
    """Decode canonical packed milliseconds with cftime."""
    safe_values = _validate_int64(values)
    decoded = cftime.num2date(
        safe_values.tolist(),
        CANONICAL_TIME_UNITS,
        calendar=PROLEPTIC_GREGORIAN,
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=False,
    )
    return tuple(cast(cftime.datetime, value) for value in decoded)


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
    """Infer the canonical policy calendar from source metadata and values."""
    metadata = dict(encoding or {})
    metadata.update(attrs or {})
    declared = metadata.get("calendar")
    if declared is None:
        raw = np.asarray(values)
        if raw.dtype.kind == "O":
            calendars = {
                str(getattr(value, "calendar", PROLEPTIC_GREGORIAN)).lower()
                for value in raw.tolist()
                if isinstance(value, cftime.datetime)
            }
            if len(calendars) > 1:
                raise ValueError("time coordinate contains multiple calendars")
            declared = next(iter(calendars), None)
    normalized_name = _declared_calendar(None if declared is None else str(declared))
    if normalized_name in ("standard", "gregorian"):
        _check_standard_dates(tuple(np.asarray(values).tolist()))
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
