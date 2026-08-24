from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pytest

from pystormtracker.convert import main, setup_parser
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


def test_convert_parser_rejects_removed_html_output() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_parser(subparsers)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "convert",
                "-i",
                "input.trackjson",
                "-o",
                "output.html",
                "-F",
                "html",
            ]
        )

    assert exc_info.value.code == 2
