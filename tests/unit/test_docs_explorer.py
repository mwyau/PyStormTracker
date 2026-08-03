from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import jsonschema


class ExplorerExtension(Protocol):
    _TRACKJSON_FIXTURE: Path

    def build_explorer(self, app: object) -> None: ...


def _explorer_extension() -> ExplorerExtension:
    path = Path(__file__).parents[2] / "docs" / "_ext" / "explorer.py"
    specification = importlib.util.spec_from_file_location("docs_explorer", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Unable to load the documentation explorer extension.")
    module: ModuleType = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return cast(ExplorerExtension, module)


def test_docs_fixture_is_trackjson_and_has_expected_segments(
    tmp_path: Path,
) -> None:
    extension = _explorer_extension()
    fixture = extension._TRACKJSON_FIXTURE
    document = json.loads(fixture.read_text(encoding="utf-8"))

    assert document["format"] == "TrackJSON/1.0"
    assert sum(track["end"] - track["start"] for track in document["tracks"]) == 20_099
    schema_path = Path(__file__).parents[2] / "schema" / "trackjson.schema.json"
    jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    ).validate(document)

    assert fixture.name == "era5_msl_2025-2026_djf_2.5x2.5_hodges.trackjson"


def test_sphinx_hook_generates_linked_assets_only_in_output(tmp_path: Path) -> None:
    extension = _explorer_extension()
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html"),
        outdir=str(tmp_path),
    )
    extension.build_explorer(app)

    explorer = tmp_path / "_static" / "explorer"
    assert (explorer / "explorer.html").is_file()
    assert (explorer / "explorer.css").is_file()
    assert (explorer / "explorer.js").is_file()
    assert (explorer / "tracks.trackjson").is_file()
    assert (
        (explorer / "tracks.trackjson").read_bytes()
        == extension._TRACKJSON_FIXTURE.read_bytes()
    )
    html = (explorer / "explorer.html").read_text(encoding="utf-8")
    javascript = (explorer / "explorer.js").read_text(encoding="utf-8")
    assert 'href="explorer.css"' in html
    assert 'src="explorer.js"' in html
    assert 'const DATA_URL = "tracks.trackjson"' in javascript
    assert "new deck.LineLayer" in javascript
    assert "new deck.ScatterplotLayer" in javascript
    assert "DataFilterExtension" in javascript
    assert "filterRange" in javascript
    assert "backgroundColor: rgba(object.color)" in javascript
    assert 'if (mode === "min") position = 1 - position' in javascript
