"""Centralized format inference and I/O facade for PyStormTracker trajectory files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ..models.tracks import Tracks
from .geojson import read_geojson, write_geojson
from .hodges import write_hodges
from .imilast import read_imilast, write_imilast
from .json import read_json, write_json

SupportedFormat = Literal["json", "trackjson", "geojson", "imilast", "hodges"]


def _format_from_extension(path: str | Path) -> SupportedFormat | None:
    """Return the format implied by a recognized file extension."""
    path_str = str(path).lower()
    if path_str.endswith(".trackjson"):
        return "json"
    if path_str.endswith(".geojson"):
        return "geojson"
    if path_str.endswith(".txt") or path_str.endswith(".dat"):
        return "imilast"
    if (
        path_str.endswith(".tdump")
        or path_str.endswith(".track")
        or path_str.endswith(".hodges")
    ):
        return "hodges"
    return None


def _detect_text_format(path: Path) -> SupportedFormat | None:
    """Identify supported text trajectory formats from their opening records."""
    try:
        with open(path, encoding="utf-8") as source:
            opening_lines = [source.readline().strip() for _ in range(3)]
    except OSError:
        return None
    first_line = opening_lines[0]
    if first_line.startswith("99 00,") or "CycloneNo" in first_line:
        return "imilast"
    if any(line.startswith("TRACK_NUM") for line in opening_lines):
        return "hodges"
    return None


def _detect_json_format(path: Path) -> SupportedFormat | None:
    """Identify TrackJSON or GeoJSON from required document-level fields."""
    try:
        with open(path, encoding="utf-8") as source:
            document: object = json.load(source)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("format") == "TrackJSON/1.0":
        return "json"
    if document.get("type") == "FeatureCollection":
        return "geojson"
    return None


def _detect_format(path: str | Path) -> SupportedFormat | None:
    """Detect a local track-file format without relying on its extension."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        with open(candidate, "rb") as source:
            opening = source.read(4_096).lstrip(b"\xef\xbb\xbf \t\r\n")
    except OSError:
        return None
    if opening.startswith((b"{", b"[")):
        return _detect_json_format(candidate)
    return _detect_text_format(candidate)


def infer_format(
    path: str | Path, default: SupportedFormat = "json"
) -> SupportedFormat:
    """
    Infer track file format based on file extension.

    Extensions:
        - .trackjson -> "json" (TrackJSON)
        - .geojson -> "geojson"
        - .txt, .dat -> "imilast"
        - .tdump, .track, .hodges -> "hodges"

    Existing ``.json`` files are inspected for either the TrackJSON ``format``
    field or the GeoJSON ``type`` field before a format is selected.
    """
    detected = _detect_format(path)
    if detected is not None:
        return detected
    return _format_from_extension(path) or default


def _resolve_format(
    path: str | Path, format: str | None, *, for_output: bool = False
) -> str:
    """Resolve an explicit format or a recognized filename extension."""
    if format is not None and format.lower() != "auto":
        return format.lower()
    detected = _detect_format(path)
    if detected is not None:
        return detected
    inferred = _format_from_extension(path)
    if inferred is not None:
        return inferred
    if for_output and str(path).lower().endswith(".json"):
        return "json"
    if Path(path).suffix:
        raise ValueError(
            f"Unsupported track file extension: '{Path(path).suffix.lower()}'."
        )
    return "json"


def load_tracks(path: str | Path, format: str | None = None) -> Tracks:
    """Load tracks from a file, auto-inferring format if omitted or set to 'auto'."""
    fmt = _resolve_format(path, format)

    if fmt in ("json", "trackjson"):
        return read_json(path)
    if fmt == "geojson":
        return read_geojson(path)
    if fmt == "imilast":
        return read_imilast(path)

    raise ValueError(
        "Unsupported input format "
        f"'{format}'. Supported formats: 'json', 'geojson', 'imilast'."
    )


def save_tracks(tracks: Tracks, path: str | Path, format: str | None = None) -> None:
    """Save tracks to a file, auto-inferring format if omitted or set to 'auto'."""
    fmt = _resolve_format(path, format, for_output=True)

    if fmt in ("json", "trackjson"):
        write_json(tracks, path)
    elif fmt == "geojson":
        write_geojson(tracks, path)
    elif fmt == "imilast":
        write_imilast(tracks, path)
    elif fmt == "hodges":
        write_hodges(tracks, path)
    else:
        raise ValueError(
            "Unsupported output format "
            f"'{format}'. Supported formats: 'json', 'geojson', 'imilast', 'hodges'."
        )
