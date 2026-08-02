"""Integration tests using checked-in track-format fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from pystormtracker.compare import _load_tracks
from pystormtracker.io.imilast import read_imilast
from pystormtracker.io.json import read_json
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from pystormtracker.models.tracks import Tracks

TRACK_FIXTURE_DIR = Path(__file__).parents[1] / "data/tracks"
IMILAST_FIXTURE = (
    TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges_imilast.txt"
)
TRACKJSON_FIXTURE = (
    TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.trackjson"
)
GEOJSON_FIXTURE = TRACK_FIXTURE_DIR / "era5_msl_2025-2026_djf_2.5x2.5_hodges.geojson"


def _first_tracks(tracks: Tracks, count: int) -> Tracks:
    """Return a bounded subset while retaining complete real trajectories."""
    selected_ids = tracks.unique_track_ids[:count]
    mask = np.isin(tracks.track_ids, selected_ids)
    return Tracks(
        track_ids=tracks.track_ids[mask],
        times=tracks.times[mask],
        lats=tracks.lats[mask],
        lons=tracks.lons[mask],
        vars_dict={name: values[mask] for name, values in tracks.vars.items()},
        track_type=tracks.track_type,
    )


@pytest.mark.integration
def test_real_trackjson_fixture_matches_schema_and_imilast_source() -> None:
    """TrackJSON fixture is schema-valid and preserves the converted source."""
    document = json.loads(TRACKJSON_FIXTURE.read_text(encoding="utf-8"))
    schema_path = Path(__file__).parents[2] / "schema/trackjson.schema.json"
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


@pytest.mark.integration
def test_compare_api_loads_real_trackjson_and_geojson_fixtures() -> None:
    """Format-aware compare loading produces identical matched trajectories."""
    reference = _first_tracks(_load_tracks(str(TRACKJSON_FIXTURE)), count=8)
    candidate = _first_tracks(_load_tracks(str(GEOJSON_FIXTURE)), count=8)

    result = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(var="msl"),
    )

    assert result.reference_count == 8
    assert result.candidate_count == 8
    assert result.match_count == 8
    assert result.reference_coverage == 1.0
    assert result.candidate_coverage == 1.0
    assert all(match.mean_separation_km < 1.0e-3 for match in result.matches)
