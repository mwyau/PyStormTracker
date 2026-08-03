from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from pystormtracker.io.format import infer_format, load_tracks, save_tracks
from pystormtracker.io.geojson import read_geojson, write_geojson
from pystormtracker.io.json import read_json, write_json
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


def _tracks() -> Tracks:
    tracks = Tracks(track_type="msl")
    tracks.add_track(
        [
            Center(
                time=np.datetime64("2020-01-01T00:00"),
                lat=40.0,
                lon=-10.0,
                vars={"msl": 101_000.0, "vo": 1.0e-4},
            ),
            Center(
                time=np.datetime64("2020-01-01T06:00"),
                lat=41.0,
                lon=-9.0,
                vars={"msl": 99_000.0, "vo": 2.0e-4},
            ),
        ]
    )
    tracks.add_track(
        [
            Center(
                time=np.datetime64("2020-01-02T00:00"),
                lat=42.0,
                lon=-8.0,
                vars={"msl": 100_000.0, "vo": 1.5e-4},
            )
        ]
    )
    return tracks


def test_trackjson_matches_draft_2020_12_schema(tmp_path: Path) -> None:
    output = tmp_path / "tracks.json"
    write_json(_tracks(), output)
    document = json.loads(output.read_text(encoding="utf-8"))
    schema_path = Path(__file__).parents[3] / "schema/trackjson.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(document)

    assert schema["$id"].endswith("/schema/trackjson.schema.json")
    assert document["format"] == "TrackJSON/1.0"
    assert document["metadata"]["primary_var"] == "msl"
    assert "data" in document
    assert document["metadata"]["units"] == {"msl": "Pa", "vo": "s^-1"}
    assert document["tracks"][0]["peak_value"] == 99_000.0


def test_trackjson_round_trips_all_variables(tmp_path: Path) -> None:
    output = tmp_path / "tracks.json"
    original = _tracks()
    write_json(original, output)

    loaded = read_json(output)

    assert loaded.track_type == "msl"
    np.testing.assert_array_equal(loaded.track_ids, original.track_ids)
    np.testing.assert_array_equal(loaded.times, original.times)
    np.testing.assert_allclose(loaded.lats, original.lats)
    np.testing.assert_allclose(loaded.lons, original.lons)
    for name, values in original.vars.items():
        np.testing.assert_allclose(loaded.vars[name], values)


def test_trackjson_preserves_explicit_tracking_mode(tmp_path: Path) -> None:
    output = tmp_path / "tracks.trackjson"
    original = _tracks()
    original.mode = "max"

    write_json(original, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    loaded = read_json(output)

    assert document["metadata"]["mode"] == "max"
    assert document["tracks"][0]["peak_value"] == 101_000.0
    assert loaded.mode == "max"


def test_empty_trackjson_preserves_primary_variable_and_variables(
    tmp_path: Path,
) -> None:
    output = tmp_path / "empty.trackjson"
    original = Tracks(
        track_ids=np.empty(0, dtype=np.int64),
        times=np.empty(0, dtype="datetime64[s]"),
        lats=np.empty(0, dtype=np.float64),
        lons=np.empty(0, dtype=np.float64),
        vars_dict={
            "msl": np.empty(0, dtype=np.float64),
            "vo": np.empty(0, dtype=np.float64),
        },
        track_type="msl",
    )
    write_json(original, output)

    loaded = read_json(output)

    assert len(loaded) == 0
    assert loaded.track_type == "msl"
    assert set(loaded.vars) == {"msl", "vo"}
    assert all(values.size == 0 for values in loaded.vars.values())


def test_geojson_round_trips_all_variables(tmp_path: Path) -> None:
    output = tmp_path / "tracks.geojson"
    original = _tracks()
    write_geojson(original, output)

    loaded = read_geojson(output)

    assert loaded.track_type == "msl"
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["features"][1]["geometry"]["type"] == "Point"
    np.testing.assert_array_equal(loaded.track_ids, original.track_ids)
    for name, values in original.vars.items():
        np.testing.assert_allclose(loaded.vars[name], values)


def test_geojson_splits_antimeridian_crossing_tracks(tmp_path: Path) -> None:
    output = tmp_path / "crossing.geojson"
    original = Tracks(track_type="msl")
    original.add_track(
        [
            Center(
                time=np.datetime64("2020-01-01T00:00"),
                lat=10.0,
                lon=170.0,
                vars={"msl": 101_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T06:00"),
                lat=11.0,
                lon=179.0,
                vars={"msl": 100_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T12:00"),
                lat=12.0,
                lon=-179.0,
                vars={"msl": 99_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T18:00"),
                lat=13.0,
                lon=-170.0,
                vars={"msl": 98_000.0},
            ),
        ]
    )

    write_geojson(original, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    geometry = document["features"][0]["geometry"]
    loaded = read_geojson(output)

    assert geometry["type"] == "MultiLineString"
    assert all(len(segment) >= 2 for segment in geometry["coordinates"])
    assert all(
        abs(segment[index][0] - segment[index - 1][0]) <= 180.0
        for segment in geometry["coordinates"]
        for index in range(1, len(segment))
    )
    np.testing.assert_array_equal(loaded.lons, original.lons)
    np.testing.assert_array_equal(loaded.lats, original.lats)


def test_geojson_uses_geometry_collection_for_two_point_crossing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "two-point-crossing.geojson"
    original = Tracks(track_type="msl")
    original.add_track(
        [
            Center(
                time=np.datetime64("2020-01-01T00:00"),
                lat=10.0,
                lon=179.0,
                vars={"msl": 101_000.0},
            ),
            Center(
                time=np.datetime64("2020-01-01T06:00"),
                lat=11.0,
                lon=-179.0,
                vars={"msl": 100_000.0},
            ),
        ]
    )

    write_geojson(original, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    loaded = read_geojson(output)

    assert document["features"][0]["geometry"]["type"] == "GeometryCollection"
    assert [
        geometry["type"]
        for geometry in document["features"][0]["geometry"]["geometries"]
    ] == ["Point", "Point"]
    np.testing.assert_array_equal(loaded.lons, original.lons)


def test_format_facade_detects_track_content_without_recognized_extension(
    tmp_path: Path,
) -> None:
    output = tmp_path / "tracks.payload"
    save_tracks(_tracks(), output, format="json")
    trackjson_with_json_extension = tmp_path / "tracks.track-data.json"
    save_tracks(_tracks(), trackjson_with_json_extension, format="json")
    geojson_with_json_extension = tmp_path / "tracks.json"
    save_tracks(_tracks(), geojson_with_json_extension, format="geojson")

    loaded = load_tracks(output)

    assert loaded.track_type == "msl"
    assert infer_format(trackjson_with_json_extension) == "json"
    assert load_tracks(trackjson_with_json_extension).track_type == "msl"
    assert infer_format(geojson_with_json_extension) == "geojson"
    assert load_tracks(geojson_with_json_extension).track_type == "msl"


def test_format_facade_infers_and_routes(tmp_path: Path) -> None:
    output = tmp_path / "tracks.geojson"
    save_tracks(_tracks(), output)

    assert infer_format("tracks.trackjson") == "json"
    assert infer_format("tracks.geojson") == "geojson"
    assert infer_format("tracks.dat") == "imilast"
    assert infer_format("tracks.tdump") == "hodges"
    assert load_tracks(output).track_type == "msl"


def test_save_tracks_uses_extension_when_overwriting_existing_file(
    tmp_path: Path,
) -> None:
    geojson_output = tmp_path / "tracks.geojson"
    json_output = tmp_path / "tracks.json"
    save_tracks(_tracks(), geojson_output, format="json")
    save_tracks(_tracks(), json_output, format="geojson")

    save_tracks(_tracks(), geojson_output)
    save_tracks(_tracks(), json_output)

    assert (
        json.loads(geojson_output.read_text(encoding="utf-8"))["type"]
        == "FeatureCollection"
    )
    assert (
        json.loads(json_output.read_text(encoding="utf-8"))["format"] == "TrackJSON/1.0"
    )
