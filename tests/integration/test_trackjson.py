"""Integration checks for the compact TrackJSON test data.

Regenerate the committed JSON test tracks manually with the exact production
command documented in ``docs/trackjson.md``. CI must not regenerate it.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import jsonschema
import msgspec
import numpy as np
import pytest

from pystormtracker.io.hodges import read_hodges
from pystormtracker.io.imilast import read_imilast
from pystormtracker.io.trackjson import TrackJSONDocument, read_trackjson
from pystormtracker.time import CANONICAL_TIME_UNITS, PROLEPTIC_GREGORIAN

TEST_DATA_DIR = Path(__file__).parents[1] / "data" / "tracks"
IMILAST_TEST_DATA = TEST_DATA_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges_imilast.txt"
HODGES_TEST_DATA = TEST_DATA_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.txt"
TRACKJSON_TEST_DATA = TEST_DATA_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.trackjson"


@pytest.mark.integration
def test_committed_trackjson_test_data_match_schema_and_source() -> None:
    raw = TRACKJSON_TEST_DATA.read_bytes()
    assert b"\n" not in raw.rstrip(b"\n")
    document = msgspec.json.Decoder(TrackJSONDocument).decode(raw)
    payload = json.loads(raw)
    schema = json.loads(
        files("pystormtracker.schemas").joinpath("trackjson.schema.json").read_bytes()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(payload)

    assert set(payload) == {"format", "metadata", "index", "data", "stats"}
    assert document.format == "TrackJSON/1.0"
    assert document.metadata.time.units == CANONICAL_TIME_UNITS
    assert document.metadata.time.calendar == PROLEPTIC_GREGORIAN
    assert document.index.offsets[-1] == 23033

    source = read_imilast(IMILAST_TEST_DATA)
    loaded = read_trackjson(TRACKJSON_TEST_DATA)
    verified = read_trackjson(TRACKJSON_TEST_DATA, verify_stats=True)

    assert loaded.metadata == source.metadata
    assert verified.metadata == source.metadata
    assert len(loaded) == 2934
    assert len(loaded.times) == 23033
    np.testing.assert_array_equal(loaded.ids, source.ids)
    np.testing.assert_array_equal(loaded.offsets, source.offsets)
    np.testing.assert_array_equal(loaded.times, source.times)
    np.testing.assert_allclose(loaded.lats, source.lats)
    np.testing.assert_allclose(loaded.lons, source.lons)
    np.testing.assert_allclose(
        loaded.variables[loaded.primary_var], source.variables[source.primary_var]
    )

    np.testing.assert_array_equal(loaded.ids[:5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        loaded.times[:5],
        np.array(
            [
                1764547200000,
                1764568800000,
                1764590400000,
                1764612000000,
                1764633600000,
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_allclose(loaded.lats[:5], [77.46, 77.69, 77.74, 77.77, 77.83])
    np.testing.assert_allclose(
        loaded.lons[:5], [142.74, 143.34, 144.26, 144.34, 145.43]
    )
    assert np.all((loaded.lons >= -180.0) & (loaded.lons < 180.0))
    assert loaded.times[0] == 1764547200000


@pytest.mark.integration
def test_hodges_test_data_are_readable_and_use_signed_longitudes() -> None:
    loaded = read_hodges(HODGES_TEST_DATA)
    assert len(loaded) == 2934
    assert loaded.offsets[-1] == 23033
    assert loaded.primary_var == "Intensity1"
    assert np.all((loaded.lons >= -180.0) & (loaded.lons < 180.0))
