"""Canonical units and automatic extrema-mode resolution."""

from typing import Literal, TypeAlias

import numpy as np
import xarray as xr

Mode: TypeAlias = Literal["min", "max"]
ModeOption: TypeAlias = Literal["auto", "min", "max"]

MODE_ALIASES: dict[str, Mode] = {
    "msl": "min",
    "slp": "min",
    "pnm": "min",
    "pres": "min",
    "pressure": "min",
    "vo": "max",
    "vort": "max",
    "vorticity": "max",
    "intensity1": "max",
}

CANONICAL_UNITS: dict[str, str] = {
    "msl": "Pa",
    "slp": "Pa",
    "pnm": "Pa",
    "pres": "Pa",
    "vo": "s^-1",
    "vort": "s^-1",
    "vorticity": "s^-1",
}


def canonical_unit_for(name: str) -> str | None:
    """Return a canonical unit for a recognized variable name, if available."""
    return CANONICAL_UNITS.get(name.lower())


def resolve_mode(variable_name: str, mode: ModeOption | None = "auto") -> Mode:
    """Resolve an explicit or automatic extrema mode for a variable name."""
    if mode in ("min", "max"):
        return mode
    if mode not in (None, "auto"):
        raise ValueError("mode must be 'auto', 'min', or 'max'")
    try:
        return MODE_ALIASES[variable_name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"cannot resolve automatic mode for variable {variable_name!r}; "
            "specify mode='min' or mode='max'"
        ) from exc


def _unit_text(value: object) -> str:
    return "" if value is None else " ".join(str(value).strip().lower().split())


def normalize_variable_units(
    data: xr.DataArray,
    *,
    variable_name: str,
    threshold: float | None,
) -> tuple[xr.DataArray, float | None, str]:
    """Normalize recognized source values and thresholds to declared units.

    Recognized pressure fields are stored in Pa and recognized vorticity fields
    in s^-1. Custom variables retain their declared unit, or use ``"1"`` when
    no unit is supplied.
    """
    canonical = canonical_unit_for(variable_name)
    source = _unit_text(data.attrs.get("units"))
    if canonical is None:
        return data, threshold, source or "1"

    if canonical == "Pa":
        factors = {
            "pa": 1.0,
            "hpa": 100.0,
            "mb": 100.0,
            "mbar": 100.0,
            "millibar": 100.0,
        }
    else:
        factors = {
            "s^-1": 1.0,
            "s-1": 1.0,
            "s**-1": 1.0,
            "1/s": 1.0,
            "10^-5 s^-1": 1.0e-5,
            "10**-5 s^-1": 1.0e-5,
        }
    if not source:
        factor = 1.0
    else:
        try:
            factor = factors[source]
        except KeyError as exc:
            raise ValueError(
                f"unsupported units {data.attrs.get('units')!r} for "
                f"recognized variable {variable_name!r}; expected {canonical!r} "
                "or a supported source unit"
            ) from exc
    if factor == 1.0 and source == canonical.lower():
        normalized = data
    else:
        normalized = data * factor
    attrs = dict(normalized.attrs)
    attrs["units"] = canonical
    normalized.attrs = attrs
    normalized_threshold = None if threshold is None else float(threshold) * factor
    if normalized_threshold is not None and not np.isfinite(normalized_threshold):
        raise ValueError("normalized detection threshold must be finite")
    return normalized, normalized_threshold, canonical
