from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pystormtracker import convert
from pystormtracker.io.format import infer_format, load_tracks
from pystormtracker.io.trackjson import read_trackjson, write_trackjson
from pystormtracker.models.tracks import Tracks, TracksMetadata, _TracksBuilder


def _source_tracks() -> Tracks:
    builder = _TracksBuilder(TracksMetadata("msl", "min", {"msl": "Pa"}))
    builder.add_track(10, ["2020-01-01T00:00"], [1.0], [2.0], {"msl": [3.0]})
    return builder.finish()


def test_extension_inference_and_text_header_detection(tmp_path: Path) -> None:
    assert infer_format("tracks.json") == "trackjson"
    assert infer_format("tracks.trackjson") == "trackjson"
    assert infer_format("tracks.hodges") == "hodges"
    assert infer_format("tracks.track") == "hodges"
    assert infer_format("tracks.tdump") == "hodges"
    assert infer_format("tracks.txt", output=True) == "imilast"
    assert infer_format("tracks.dat", output=True) == "imilast"
    assert infer_format("tracks", output=True) == "trackjson"

    hodges = tmp_path / "input.txt"
    hodges.write_text("TRACK_NUM 0 ADD_FLD 0 0 &\n", encoding="utf-8")
    assert infer_format(hodges) == "hodges"
    imilast = tmp_path / "imilast.txt"
    imilast.write_text("99 00,CycloneNo,StepNo\n", encoding="utf-8")
    assert infer_format(imilast) == "imilast"
    unknown = tmp_path / "unknown.txt"
    unknown.write_text("not a supported header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot identify text track format"):
        infer_format(unknown)


def test_text_header_detection_reads_imilast_before_extension_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text(
        "99 00,CycloneNo,StepNo,DateI10,Year,Month,Day,Time,LongE,LatN,MSL\n"
        "90 7 2\n"
        "00 7 1 2025010100 2025 01 01 00 10.0 20.0 1000.0\n"
        "00 7 2 2025010106 2025 01 01 06 11.0 21.0 990.0\n",
        encoding="utf-8",
    )
    tracks = load_tracks(source)
    assert tracks.primary_var == "MSL"
    assert tracks.mode == "min"
    assert tracks.units == {"MSL": "Pa"}
    assert tracks.variables["MSL"].tolist() == [100000.0, 99000.0]


def test_hodges_input_and_json_output_are_inferred_from_extensions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hodges"
    source.write_text(
        "TRACK_NUM 1 ADD_FLD 0 0 &\n"
        "TRACK_ID 3\n"
        "POINT_NUM 2\n"
        "2025010100 10.0 20.0 4.0\n"
        "2025010106 11.0 21.0 5.0\n",
        encoding="utf-8",
    )
    tracks = load_tracks(source)
    assert tracks.primary_var == "Intensity1"
    assert tracks.mode == "max"
    output = tmp_path / "result.json"
    tracks.write(output)
    loaded = read_trackjson(output)
    assert loaded == tracks


def test_unknown_suffixes_require_explicit_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot infer input track format"):
        infer_format(tmp_path / "input.unknown")
    with pytest.raises(ValueError, match="cannot infer output track format"):
        infer_format(tmp_path / "output.unknown", output=True)
    assert infer_format(tmp_path / "input.unknown", format="trackjson") == "trackjson"
    assert (
        infer_format(tmp_path / "output.unknown", format="trackjson", output=True)
        == "trackjson"
    )


def test_convert_defaults_infer_json_and_no_suffix_outputs(tmp_path: Path) -> None:
    source = _source_tracks()
    input_json = tmp_path / "input.json"
    write_trackjson(source, input_json, include_stats=False)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    convert.setup_parser(subparsers)
    args = parser.parse_args(
        ["convert", "-i", str(input_json), "-o", str(tmp_path / "out.trackjson")]
    )
    assert args.in_format == "auto"
    assert args.out_format == "auto"
    assert args.mode == "auto"
    convert.main(args)
    trackjson_output = tmp_path / "out.trackjson"
    assert read_trackjson(trackjson_output) == source

    args = parser.parse_args(
        ["convert", "-i", str(trackjson_output), "-o", str(tmp_path / "out.json")]
    )
    convert.main(args)
    assert read_trackjson(tmp_path / "out.json") == source

    no_suffix = tmp_path / "out"
    args = parser.parse_args(["convert", "-i", str(input_json), "-o", str(no_suffix)])
    convert.main(args)
    assert read_trackjson(no_suffix) == source


def test_convert_can_rename_intensity_variable_and_resolve_mode(
    tmp_path: Path,
) -> None:
    builder = _TracksBuilder(TracksMetadata("Intensity1", "max", {"Intensity1": "1"}))
    builder.add_track(
        3,
        ["2020-01-01T00:00"],
        [10.0],
        [20.0],
        {"Intensity1": [1.0]},
    )
    source = tmp_path / "source.trackjson"
    write_trackjson(builder.finish(), source, include_stats=False)

    output = tmp_path / "renamed.json"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    convert.setup_parser(subparsers)
    args = parser.parse_args(
        [
            "convert",
            "-i",
            str(source),
            "-o",
            str(output),
            "--var",
            "msl",
        ]
    )
    convert.main(args)

    renamed = read_trackjson(output)
    assert renamed.primary_var == "msl"
    assert renamed.mode == "min"
    assert renamed.units == {"msl": "Pa"}
    assert set(renamed.variables) == {"msl"}
