from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from pystormtracker.convert import generate_html, main
from pystormtracker.io.trackjson import read_trackjson
from pystormtracker.models.tracks import Tracks, TracksMetadata, _TracksBuilder


def dummy_tracks() -> Tracks:
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.add_track(
        1,
        [1577836800000, 1577840400000],
        [0.0, 1.0],
        [0.0, 1.0],
        {"msl": [100000.0, 99000.0]},
    )
    return builder.finish()


def test_generate_html_writes_placeholder_without_data_script(tmp_path: Path) -> None:
    outfile = tmp_path / "explorer.html"
    with pytest.warns(UserWarning, match="static placeholder"):
        generate_html(outfile)
    content = outfile.read_text(encoding="utf-8")
    assert "being redesigned" in content
    assert "trackjson-data" not in content
    assert "Plotly" not in content
    assert not (tmp_path / "explorer.tracks.js").exists()


def test_convert_uses_extension_defaults_for_json(tmp_path: Path) -> None:
    source = tmp_path / "source.trackjson"
    output = tmp_path / "result.json"
    from pystormtracker.io.trackjson import write_trackjson

    write_trackjson(dummy_tracks(), source)
    main(
        Namespace(
            input=str(source),
            output=str(output),
            in_format="auto",
            out_format="auto",
            variable=None,
            unit=None,
            detection_mode="auto",
        )
    )
    assert read_trackjson(output) == dummy_tracks()
