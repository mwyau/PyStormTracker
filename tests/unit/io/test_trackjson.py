from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from pystormtracker.io.format import infer_format, load_tracks, save_tracks
from pystormtracker.io.geojson import read_geojson, write_geojson
from pystormtracker.io.imilast import read_imilast
from pystormtracker.io.json import read_json, write_json
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks

TRACK_FIXTURE_DIR = Path(__file__).parents[3] / "tests/data/tracks"
IMILAST_FIXTURE = (
    TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges_imilast.txt"
)
TRACKJSON_FIXTURE = (
    TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.trackjson"
)
GEOJSON_FIXTURE = TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.geojson"


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
    assert document["metadata"]["units"] == {"msl": "Pa", "vo": "s^-1"}
    assert document["tracks"][0]["peak_value"] == 99_000.0


def test_trackjson_fixture_matches_schema_and_loads() -> None:
    document = json.loads(TRACKJSON_FIXTURE.read_text(encoding="utf-8"))
    schema_path = Path(__file__).parents[3] / "schema/trackjson.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator(schema).validate(document)
    source = read_imilast(IMILAST_FIXTURE)
    loaded = read_json(TRACKJSON_FIXTURE)

    assert loaded.track_type == "msl"
    np.testing.assert_array_equal(loaded.track_ids, source.track_ids)
    np.testing.assert_array_equal(loaded.times, source.times)
    np.testing.assert_allclose(loaded.lats, source.lats)
    np.testing.assert_allclose(loaded.lons, source.lons)
    np.testing.assert_allclose(loaded.vars["msl"], source.vars["MSL"])


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


def test_geojson_round_trips_all_variables(tmp_path: Path) -> None:
    output = tmp_path / "tracks.geojson"
    original = _tracks()
    write_geojson(original, output)

    loaded = read_geojson(output)

    assert loaded.track_type == "msl"
    np.testing.assert_array_equal(loaded.track_ids, original.track_ids)
    for name, values in original.vars.items():
        np.testing.assert_allclose(loaded.vars[name], values)


def test_geojson_fixture_matches_trackjson_fixture() -> None:
    trackjson_tracks = read_json(TRACKJSON_FIXTURE)
    geojson_tracks = read_geojson(GEOJSON_FIXTURE)

    np.testing.assert_array_equal(geojson_tracks.track_ids, trackjson_tracks.track_ids)
    np.testing.assert_array_equal(geojson_tracks.times, trackjson_tracks.times)
    np.testing.assert_allclose(geojson_tracks.lats, trackjson_tracks.lats)
    np.testing.assert_allclose(geojson_tracks.lons, trackjson_tracks.lons)
    for name, values in trackjson_tracks.vars.items():
        np.testing.assert_allclose(geojson_tracks.vars[name], values)


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
    assert infer_format(IMILAST_FIXTURE) == "imilast"
    assert (
        infer_format(TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.txt")
        == "hodges"
    )


def test_format_facade_infers_and_routes(tmp_path: Path) -> None:
    output = tmp_path / "tracks.geojson"
    save_tracks(_tracks(), output)

    assert infer_format("tracks.trackjson") == "json"
    assert infer_format("tracks.geojson") == "geojson"
    assert infer_format("tracks.dat") == "imilast"
    assert infer_format("tracks.tdump") == "hodges"
    assert load_tracks(output).track_type == "msl"
