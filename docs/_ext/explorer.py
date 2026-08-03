"""Install the TrackJSON explorer demonstration for Sphinx."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pystormtracker

if TYPE_CHECKING:
    from sphinx.application import Sphinx


_TRACKJSON_FIXTURE = (
    Path(__file__).parents[2]
    / "tests"
    / "data"
    / "tracks"
    / "era5_msl_2025-2026_djf_2.5x2.5_hodges.trackjson"
)


def _copy_explorer_assets(destination: Path) -> None:
    """Copy source-only browser assets into Sphinx's generated static tree."""
    package_file = pystormtracker.__file__
    if package_file is None:
        raise RuntimeError("Cannot locate PyStormTracker explorer source assets.")
    source_directory = Path(package_file).parent / "templates"
    for asset_name in ("explorer.html", "explorer.css", "explorer.js"):
        shutil.copyfile(source_directory / asset_name, destination / asset_name)


def build_explorer(app: Sphinx) -> None:
    """Install explorer assets and demo data only in an HTML build output."""
    if app.builder.format != "html":
        return
    destination = Path(app.outdir) / "_static" / "explorer"
    destination.mkdir(parents=True, exist_ok=True)
    _copy_explorer_assets(destination)
    shutil.copyfile(_TRACKJSON_FIXTURE, destination / "tracks.trackjson")


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the generated explorer build hook."""
    app.connect("builder-inited", build_explorer)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
