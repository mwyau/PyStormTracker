"""Public routing and extension inference for supported trajectory formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

from ..models.tracks import DetectionMode, Tracks
from ..models.units import resolve_mode
from .imilast import read_imilast, write_imilast
from .track import TrackNumericTime, read_track, write_track
from .trackjson import read_trackjson, write_trackjson

type SupportedFormat = Literal["json", "track", "imilast"]
SUPPORTED_FORMATS: Final = ("json", "track", "imilast")

LOGGER = logging.getLogger(__name__)


def _extension_format(path: str | Path) -> SupportedFormat | None:
    suffix = Path(path).suffix.lower()
    if suffix in (".trackjson", ".json"):
        return "json"
    if suffix in (".track", ".tdump", ".hodges"):
        return "track"
    return None


def _detect_text_format(path: Path) -> SupportedFormat | None:
    try:
        with path.open(encoding="utf-8") as source:
            for _ in range(32):
                line = source.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("TRACK_NUM"):
                    return "track"
                if stripped.startswith("99 00") or "CycloneNo" in stripped:
                    return "imilast"
                if stripped[0].isdigit() and stripped.split()[0] in ("0", "0.0"):
                    continue
                break
    except OSError as exc:
        raise ValueError(f"Unable to inspect track format for {path}: {exc}") from exc
    return None


def _unsupported_format(name: str) -> ValueError:
    supported = ", ".join(SUPPORTED_FORMATS)
    return ValueError(f"unsupported track format {name!r}; use one of: {supported}")


def _resolve_format(
    path: str | Path,
    format_name: str | None,
    *,
    output: bool,
) -> SupportedFormat:
    if format_name is not None and format_name.lower() != "auto":
        normalized = format_name.lower()
        if normalized not in SUPPORTED_FORMATS:
            raise _unsupported_format(format_name)
        if normalized == "json":
            return "json"
        if normalized == "track":
            return "track"
        return "imilast"
    path_obj = Path(path)
    extension = _extension_format(path_obj)
    if extension is not None:
        return extension
    suffix = path_obj.suffix.lower()
    if output and suffix in (".txt", ".dat"):
        return "imilast"
    if not output and suffix in (".txt", ".dat"):
        detected = _detect_text_format(path_obj)
        if detected is not None:
            return detected
        raise ValueError(
            f"cannot identify text track format from {path}; specify an explicit format"
        )
    if output and not suffix:
        return "json"
    if output:
        raise ValueError(
            f"cannot infer output track format from suffix {suffix!r}; "
            "specify an explicit format"
        )
    raise ValueError(
        f"cannot infer input track format from suffix {suffix!r}; "
        "specify an explicit format"
    )


def infer_format(
    path: str | Path,
    *,
    format: str | None = None,
    output: bool = False,
) -> SupportedFormat:
    """Infer a supported format from an explicit option or path."""
    return _resolve_format(path, format, output=output)


def load_tracks(
    path: str | Path,
    format: str | None = None,
    *,
    primary_variable: str | None = None,
    mode: DetectionMode | None = "auto",
    track_numeric_time: TrackNumericTime = "reject",
    track_frame_times: object | None = None,
) -> Tracks:
    """Load one of the supported trajectory formats.

    Numeric point times in TRACK ASCII files are ambiguous by design. For a
    TRACK source output, select ``track_numeric_time='frame_index'`` and pass
    the exact source time coordinate as ``track_frame_times``. The default
    rejects numeric TRACK time tokens rather than guessing an epoch.
    """
    selected = _resolve_format(path, format, output=False)
    if selected == "imilast":
        return read_imilast(filename=path, primary_variable=primary_variable, mode=mode)
    if selected == "track":
        selected_var = primary_variable or "Intensity1"
        return read_track(
            path,
            primary_variable=selected_var,
            mode=resolve_mode(selected_var, mode),
            track_numeric_time=track_numeric_time,
            track_frame_times=track_frame_times,
        )
    return read_trackjson(path)


def save_tracks(
    tracks: Tracks, path: str | Path, format: SupportedFormat | None = None
) -> None:
    """Save one of the supported trajectory formats."""
    selected = _resolve_format(path, format, output=True)
    LOGGER.info(
        "Writing tracks: path=%s format=%s tracks=%d points=%d",
        path,
        selected,
        len(tracks),
        int(tracks.times.size),
    )
    if selected == "imilast":
        write_imilast(tracks, path)
    elif selected == "track":
        write_track(tracks, path)
    else:
        write_trackjson(tracks, path)


__all__ = [
    "SUPPORTED_FORMATS",
    "SupportedFormat",
    "infer_format",
    "load_tracks",
    "save_tracks",
]
