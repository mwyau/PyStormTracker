"""Optional cftime adapter for PyStormTracker.

Only lazily imported when user inputs or operations explicitly involve cftime objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any

import numpy as np
from numpy.typing import NDArray


def is_cftime_available() -> bool:
    """Return whether the optional cftime dependency is installed."""
    return find_spec("cftime") is not None


def require_cftime() -> None:
    """Raise actionable error if cftime is not installed."""
    if not is_cftime_available():
        raise ValueError(
            "cftime datetime input requires the 'netcdf4' optional dependency"
        )


def is_cftime_datetime(obj: object) -> bool:
    """Check if an object is a cftime.datetime instance without unconditional import."""
    mod = getattr(type(obj), "__module__", "")
    if not (mod == "cftime" or mod.startswith("cftime.")):
        return False
    if not is_cftime_available():
        return False
    import cftime

    return isinstance(obj, cftime.datetime)


def get_cftime_calendar(obj: object, default: str = "proleptic_gregorian") -> str:
    """Extract calendar name from a cftime datetime object."""
    return str(getattr(obj, "calendar", default)).lower()


def encode_cftime_objects(
    values: Sequence[object],
    *,
    canonical_time_units: str = "milliseconds since 1970-01-01 00:00:00",
    proleptic_gregorian: str = "proleptic_gregorian",
) -> NDArray[np.int64]:
    """Encode cftime datetime objects to canonical integer milliseconds."""
    require_cftime()
    import cftime

    if not values:
        return np.empty(0, dtype=np.int64)

    numeric = np.asarray(
        cftime.date2num(
            list(values),
            canonical_time_units,
            calendar=proleptic_gregorian,
            longdouble=True,
        ),
        dtype=np.longdouble,
    )
    if np.any(~np.isfinite(numeric)):
        raise ValueError("CF time values must be finite")
    rounded = np.rint(numeric)
    if np.any(np.abs(numeric - rounded) > np.longdouble("1e-6")):
        raise ValueError("CF time values must resolve to integer milliseconds")
    return np.asarray(rounded, dtype=np.int64)


def decode_to_cftime(
    values: Sequence[int] | NDArray[np.int64],
    *,
    canonical_time_units: str = "milliseconds since 1970-01-01 00:00:00",
    proleptic_gregorian: str = "proleptic_gregorian",
) -> tuple[Any, ...]:
    """Decode canonical integer milliseconds to cftime datetime objects."""
    require_cftime()
    import cftime

    safe_values = np.asarray(values, dtype=np.int64)
    decoded = cftime.num2date(
        safe_values.tolist(),
        canonical_time_units,
        calendar=proleptic_gregorian,
        only_use_cftime_datetimes=True,
        only_use_python_datetimes=False,
    )
    return tuple(decoded)
