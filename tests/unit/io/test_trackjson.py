from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jsonschema
import msgspec
import numpy as np
import pytest

from pystormtracker.io.format import load_tracks, save_tracks
from pystormtracker.io.trackjson import (
    STAT_ARRAY_FIELDS,
    TrackJSONDocument,
    TrackJSONStats,
    compute_trackjson_stats,
    encode_trackjson,
    read_trackjson,
    write_trackjson,
)
from pystormtracker.models.geo import SpatialBounds
from pystormtracker.models.tracks import (
    ProcessingStep,
    Tracks,
    TracksBuilder,
    TracksMetadata,
)
from pystormtracker.schemas.generate import generate_trackjson_schema
from pystormtracker.time import CANONICAL_TIME_UNITS

JSONObject = dict[str, object]


def make_tracks(*, bounds: SpatialBounds | None = None) -> Tracks:
    builder = TracksBuilder(
        TracksMetadata(
            "msl",
            "min",
            {"msl": "Pa", "label": "1"},
            bounds,
            (ProcessingStep("spectral_filter", True, {"lmin": 5, "lmax": 42}),),
        )
    )
    builder.add_track(
        10,
        [
            np.datetime64("2020-01-01T00:00:00.123"),
            np.datetime64("2020-01-01T06:00:00.456"),
            np.datetime64("2020-01-01T12:00:00.789"),
        ],
        [10.0, 11.0, 12.0],
        [175.0, 179.0, -178.0],
        {"msl": [101000.0, np.nan, 99000.0], "label": [1.0, 2.0, 3.0]},
    )
    builder.add_track(
        35,
        [np.datetime64("2020-01-02T00:00:00.999")],
        [-20.0],
        [-181.0],
        {"msl": [100000.0], "label": [4.0]},
    )
    return builder.finish()


def read_payload(path: Path) -> JSONObject:
    return cast(JSONObject, json.loads(path.read_text(encoding="utf-8")))


def write_payload(path: Path, payload: JSONObject) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def object_member(payload: JSONObject, key: str) -> JSONObject:
    value = payload[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} is not a JSON object")
    return cast(JSONObject, value)


def list_member(payload: JSONObject, key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} is not a JSON array")
    return cast(list[object], value)


def test_default_writer_omits_stats_and_explicit_writer_adds_wire_stats(
    tmp_path: Path,
) -> None:
    source = make_tracks(bounds=SpatialBounds(0.0, 90.0, 120.0, -100.0))
    default_path = tmp_path / "default.trackjson"
    stats_path = tmp_path / "stats.trackjson"
    write_trackjson(source, default_path)
    write_trackjson(source, stats_path, include_stats=True)

    default_payload = read_payload(default_path)
    stats_payload = read_payload(stats_path)
    assert "stats" not in default_payload
    assert "stats" in stats_payload
    processing = object_member(default_payload, "metadata")["processing"]
    assert isinstance(processing, list)
    assert processing


def test_round_trip_preserves_canonical_data_metadata_and_discards_stats(
    tmp_path: Path,
) -> None:
    source = make_tracks(bounds=SpatialBounds(0.0, 90.0, 120.0, -100.0))
    path = tmp_path / "tracks.trackjson"
    path.write_bytes(encode_trackjson(source, include_stats=True))

    loaded_default = read_trackjson(path)
    loaded_verified = read_trackjson(path, verify_stats=True)
    assert loaded_default == source
    assert loaded_verified == source
    assert not hasattr(loaded_default, "stats")
    assert loaded_default.metadata.bounds == source.metadata.bounds
    assert loaded_default.metadata.processing == source.metadata.processing
    np.testing.assert_array_equal(loaded_default.offsets, [0, 3, 4])
    np.testing.assert_array_equal(loaded_default.times[:1], [1577836800123])
    assert loaded_default.lons.tolist() == [175.0, 179.0, -178.0, 179.0]
    assert np.isnan(loaded_default.variables["msl"][1])


def test_stats_are_wire_only_and_align_with_ids() -> None:
    stats = compute_trackjson_stats(make_tracks())
    assert isinstance(stats, TrackJSONStats)
    assert stats.point_count == [3, 1]
    assert stats.start_time == [1577836800123, 1577923200999]
    assert stats.end_time == [1577880000789, 1577923200999]
    assert stats.duration_hours == pytest.approx([12.000185, 0.0])
    assert stats.path_length_km[1] == 0.0
    assert stats.displacement_km[1] == 0.0
    assert stats.antimeridian_wrap == [True, False]
    assert all(len(getattr(stats, name)) == 2 for name in STAT_ARRAY_FIELDS)


def test_stats_field_classification_matches_wire_struct() -> None:
    wire_fields = {
        field.name
        for field in msgspec.structs.fields(TrackJSONStats)
        if field.name != "version"
    }
    assert set(STAT_ARRAY_FIELDS) == wire_fields
    assert len(STAT_ARRAY_FIELDS) == 19


def test_nan_variables_and_missing_peak_are_encoded_as_null() -> None:
    builder = TracksBuilder(TracksMetadata("custom", "max", {"custom": "1"}))
    builder.add_track(
        1,
        [1577836800000, 1577840400000],
        [0.0, 1.0],
        [0.0, 1.0],
        {"custom": [np.nan, np.nan]},
    )
    document = msgspec.json.decode(
        encode_trackjson(builder.finish(), include_stats=True), type=TrackJSONDocument
    )
    assert document.data.variables["custom"] == [None, None]
    assert document.stats is not msgspec.UNSET
    assert document.stats.peak_time == [None]
    assert document.stats.peak_lat == [None]
    assert document.stats.peak_lon == [None]
    assert document.stats.peak_value == [None]


def test_typed_document_and_packaged_schema_validate() -> None:
    raw = encode_trackjson(make_tracks(), include_stats=True)
    document = msgspec.json.Decoder(TrackJSONDocument).decode(raw)
    payload = cast(JSONObject, json.loads(raw))
    schema = generate_trackjson_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(payload)
    assert document.metadata.time.units == CANONICAL_TIME_UNITS
    assert document.index.offsets == [0, 3, 4]


def test_empty_document_uses_complete_offset_buffer_and_omits_optional_members(
    tmp_path: Path,
) -> None:
    source = Tracks.empty(TracksMetadata("msl", "min", {"msl": "Pa"}))
    path = tmp_path / "empty.json"
    write_trackjson(source, path)
    payload = read_payload(path)
    assert payload["index"] == {"ids": [], "offsets": [0]}
    assert "bounds" not in object_member(payload, "metadata")
    assert "processing" not in object_member(payload, "metadata")
    assert "stats" not in payload
    assert read_trackjson(path).variables["msl"].size == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda payload: payload.update(format="TrackJSON/0.1"), "Invalid TrackJSON"),
        (lambda payload: payload.update(tracks={}), "Invalid TrackJSON"),
        (
            lambda payload: object_member(payload, "index").update(offsets=[0, 1]),
            "offsets",
        ),
        (
            lambda payload: object_member(payload, "index").update(ids=[10, 10]),
            "track IDs",
        ),
        (
            lambda payload: object_member(payload, "data").update(lats=[1.0]),
            "lats",
        ),
        (
            lambda payload: object_member(payload, "metadata").pop("time"),
            "Invalid TrackJSON",
        ),
        (
            lambda payload: object_member(payload, "metadata").update(bounds=None),
            "Invalid TrackJSON",
        ),
    ],
)
def test_invalid_documents_report_the_relevant_invariant(
    tmp_path: Path,
    mutation: Callable[[JSONObject], object],
    expected: str,
) -> None:
    source = tmp_path / "source.trackjson"
    source.write_bytes(encode_trackjson(make_tracks()))
    payload = read_payload(source)
    mutation(payload)
    invalid = tmp_path / "invalid.trackjson"
    write_payload(invalid, payload)
    with pytest.raises(ValueError, match=expected):
        read_trackjson(invalid)


@pytest.mark.parametrize("value", [True, 1.5, "10"])
def test_ids_require_integer_json_values(tmp_path: Path, value: object) -> None:
    source = tmp_path / "source.trackjson"
    source.write_bytes(encode_trackjson(make_tracks()))
    payload = read_payload(source)
    ids = list_member(object_member(payload, "index"), "ids")
    ids[0] = value
    invalid = tmp_path / "invalid.trackjson"
    write_payload(invalid, payload)
    with pytest.raises(ValueError, match="ids"):
        read_trackjson(invalid)


def test_stale_stats_are_ignored_by_default_and_rejected_when_verified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.trackjson"
    source.write_bytes(encode_trackjson(make_tracks(), include_stats=True))
    payload = read_payload(source)
    path_lengths = list_member(object_member(payload, "stats"), "path_length_km")
    current = path_lengths[0]
    assert isinstance(current, (int, float))
    assert not isinstance(current, bool)
    path_lengths[0] = float(current) + 1.0
    stale = tmp_path / "stale.trackjson"
    write_payload(stale, payload)
    assert read_trackjson(stale) == make_tracks()
    with pytest.raises(ValueError, match="stats path_length_km failed verification"):
        read_trackjson(stale, verify_stats=True)


def test_invalid_stats_lengths_fail_only_during_explicit_verification(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.trackjson"
    source.write_bytes(encode_trackjson(make_tracks(), include_stats=True))
    payload = read_payload(source)
    object_member(payload, "stats")["point_count"] = [3]
    stale = tmp_path / "invalid-length.trackjson"
    write_payload(stale, payload)
    assert read_trackjson(stale) == make_tracks()
    with pytest.raises(ValueError, match=r"\$\.stats\.point_count"):
        read_trackjson(stale, verify_stats=True)


def test_format_routing_for_json_and_trackjson(tmp_path: Path) -> None:
    source = make_tracks()
    trackjson_path = tmp_path / "tracks.trackjson"
    json_path = tmp_path / "tracks.json"
    save_tracks(source, trackjson_path)
    save_tracks(source, json_path)
    assert load_tracks(trackjson_path) == source
    assert load_tracks(json_path) == source


def test_processing_round_trip(tmp_path: Path) -> None:
    source = make_tracks()
    path = tmp_path / "proleptic.trackjson"
    path.write_bytes(encode_trackjson(source))
    loaded = read_trackjson(path)
    assert loaded.metadata.processing == source.metadata.processing
