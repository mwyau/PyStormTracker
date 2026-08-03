"""Public routing and extension inference for supported trajectory formats."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from ..models.tracks import Tracks
from ..models.units import ModeOption, resolve_mode
from .hodges import read_hodges, write_hodges
from .imilast import read_imilast, write_imilast
from .trackjson import read_trackjson, write_trackjson

SupportedFormat: TypeAlias = Literal["trackjson", "imilast", "hodges"]
SUPPORTED_FORMATS: tuple[SupportedFormat, ...] = ("trackjson", "imilast", "hodges")


def _extension_format(path: str | Path) -> SupportedFormat | None:
    suffix = Path(path).suffix.lower()
    if suffix in (".trackjson", ".json"):
        return "trackjson"
    if suffix in (".hodges", ".track", ".tdump"):
        return "hodges"
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
                    return "hodges"
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
        return normalized
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
        return "trackjson"
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
    primary_var: str | None = None,
    mode: ModeOption | None = "auto",
) -> Tracks:
    """Load one of the supported trajectory formats."""
    selected = _resolve_format(path, format, output=False)
    if selected == "imilast":
        return read_imilast(primary_var=primary_var, mode=mode, filename=path)
    if selected == "hodges":
        selected_var = primary_var or "Intensity1"
        return read_hodges(
            path,
            primary_var=selected_var,
            mode=resolve_mode(selected_var, mode),
        )
    return read_trackjson(path)


def save_tracks(
    tracks: Tracks, path: str | Path, format: SupportedFormat | None = None
) -> None:
    """Save one of the supported trajectory formats."""
    selected = _resolve_format(path, format, output=True)
    if selected == "imilast":
        write_imilast(tracks, path)
    elif selected == "hodges":
        write_hodges(tracks, path)
    else:
        write_trackjson(tracks, path)
