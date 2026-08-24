from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .tracks import DetectionMode, ResolvedDetectionMode

_MODE_ALIASES: Final[dict[str, ResolvedDetectionMode]] = {
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

_CANONICAL_UNITS: Final[dict[str, str]] = {
    "msl": "Pa",
    "slp": "Pa",
    "pnm": "Pa",
    "pres": "Pa",
    "vo": "s^-1",
    "vort": "s^-1",
    "vorticity": "s^-1",
}

LOGGER = logging.getLogger(__name__)


def canonical_unit_for(name: str) -> str | None:
    """Return a canonical unit for a recognized variable name, if available."""
    return _CANONICAL_UNITS.get(name.lower())


def resolve_mode(
    variable: str,
    detection_mode: DetectionMode | None = "auto",
) -> ResolvedDetectionMode:
    """Resolve an explicit or automatic extrema mode for a variable name."""
    if detection_mode in ("min", "max"):
        return detection_mode
    if detection_mode not in (None, "auto"):
        raise ValueError("detection_mode must be 'auto', 'min', or 'max'")
    try:
        return _MODE_ALIASES[variable.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"cannot resolve automatic detection_mode for variable {variable!r}; "
            "specify detection_mode='min' or detection_mode='max'"
        ) from exc


def _unit_text(value: object) -> str:
    return "" if value is None else " ".join(str(value).strip().lower().split())


def normalize_variable_units(
    data: xr.DataArray,
    *,
    variable: str,
    intensity_threshold: float | None = None,
) -> tuple[xr.DataArray, float | None, str]:
    """Normalize recognized source values and thresholds to declared units.

    Recognized pressure fields are stored in Pa and recognized vorticity fields
    in s^-1. Custom variables retain their declared unit, or use ``"1"`` when
    no unit is supplied.
    """
    canonical = canonical_unit_for(variable)
    source = _unit_text(data.attrs.get("units"))
    if canonical is None:
        return data, intensity_threshold, source or "1"

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
                f"recognized variable {variable!r}; expected {canonical!r} "
                "or a supported source unit"
            ) from exc
    if factor == 1.0 and source == canonical.lower():
        normalized = data
    else:
        normalized = data * factor
    attrs = dict(normalized.attrs)
    attrs["units"] = canonical
    normalized.attrs = attrs
    normalized_threshold = (
        None if intensity_threshold is None else float(intensity_threshold) * factor
    )
    if normalized_threshold is not None and not np.isfinite(normalized_threshold):
        raise ValueError("normalized detection threshold must be finite")
    LOGGER.debug(
        "Normalized units variable=%s source=%r normalized=%r factor=%g threshold=%r",
        variable,
        source or "1",
        canonical,
        factor,
        normalized_threshold,
    )
    return normalized, normalized_threshold, canonical
