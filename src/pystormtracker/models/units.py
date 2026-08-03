"""Canonical internal units for tracked meteorological variables."""

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
