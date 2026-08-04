from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import numpy as np
import pytest

from pystormtracker.compare import _load_tracks, main
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from pystormtracker.models.tracks import Tracks, TracksBuilder, TracksMetadata


def make_tracks(
    records: list[tuple[list[float], list[float], list[np.datetime64], list[float]]],
    *,
    variable: str = "intensity",
    mode: str = "max",
) -> Tracks:
    unit = {
        "msl": "Pa",
        "vo": "s^-1",
        "vort": "s^-1",
    }.get(variable, "1")
    builder = TracksBuilder(
        TracksMetadata(variable, mode, {variable: unit})  # type: ignore[arg-type]
    )
    for track_id, (lats, lons, times, values) in enumerate(records, start=1):
        builder.add_track(track_id, times, lats, lons, {variable: values})
    return builder.finish()


@pytest.fixture
def times() -> list[np.datetime64]:
    return [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T06:00"),
        np.datetime64("2020-01-01T12:00"),
    ]


def test_compare_tracks_selects_closest_candidate_per_reference(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([50.0, 51.0, 52.0], [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([10.0, 10.0, 10.0], [100.0, 101.0, 102.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks(
        [
            ([50.2, 51.2, 52.2], [0.2, 1.2, 2.2], times, [0.0] * 3),
            ([10.2, 10.2, 10.2], [100.2, 101.2, 102.2], times, [0.0] * 3),
        ]
    )

    result = compare_tracks(reference, candidate)

    assert result.match_count == 2
    assert [(match.reference_id, match.candidate_id) for match in result.matches] == [
        (1, 1),
        (2, 2),
    ]
    assert result.reference_coverage == 1.0
    assert result.candidate_coverage == 1.0


def test_compare_tracks_allows_candidate_reuse(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([0.1] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks([([0.05] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3)])

    result = compare_tracks(reference, candidate)

    assert result.match_count == 2
    assert [match.candidate_id for match in result.matches] == [1, 1]
    assert result.candidate_coverage == 1.0


def test_compare_tracks_requires_equal_overlap_section_lengths() -> None:
    reference_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T06:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    candidate_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    reference = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], reference_times, [0.0] * 3)])
    candidate = make_tracks([([0.0] * 2, [0.0, 2.0], candidate_times, [0.0] * 2)])

    with pytest.raises(ValueError, match="equal point counts"):
        compare_tracks(reference, candidate)


def test_compare_tracks_reports_path_and_intensity_metrics(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [([0.0] * 3, [0.0, 1.0, 2.0], times, [1.0e-4, 2.0e-4, 3.0e-4])],
        variable="vo",
    )
    candidate = make_tracks(
        [([0.0] * 3, [0.1, 1.1, 2.1], times, [1.1e-4, 2.1e-4, 3.1e-4])],
        variable="vo",
    )

    match = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(var="vo"),
    ).matches[0]

    assert match.reference.duration_hours == 12.0
    assert match.reference.path_length_km == pytest.approx(222.39, rel=1e-3)
    assert match.reference.mean_speed_kmh == pytest.approx(18.53, rel=1e-3)
    assert match.intensity_difference is not None
    assert match.intensity_difference.bias == pytest.approx(1.0e-5)
    assert match.mean_separation_km == pytest.approx(11.12, rel=1e-2)


def test_compare_tracks_uses_explicit_metadata_mode(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [([0.0] * 3, [0.0, 1.0, 2.0], times, [10.0, 50.0, 20.0])],
        variable="custom",
    )
    candidate = make_tracks(
        [([0.0] * 3, [0.1, 1.1, 2.1], times, [12.0, 52.0, 22.0])],
        variable="custom",
    )

    result_min = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(var="custom", mode="min"),
    )
    result_max = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(var="custom", mode="max"),
    )

    assert result_min.matches[0].reference.peak_intensity == 10.0
    assert result_max.matches[0].reference.peak_intensity == 50.0


def test_compare_tracks_handles_dateline(times: list[np.datetime64]) -> None:
    reference = make_tracks([([30.0] * 3, [179.8, 179.9, 180.0], times, [0.0] * 3)])
    candidate = make_tracks([([30.0] * 3, [-179.8, -179.9, -180.0], times, [0.0] * 3)])

    result = compare_tracks(reference, candidate)

    assert result.match_count == 1
    assert result.matches[0].mean_separation_deg < 1.0


def test_compare_tracks_rejects_missing_requested_variable(
    times: list[np.datetime64],
) -> None:
    values = [np.nan] * 3
    reference = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, values)])
    candidate = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, values)])

    with pytest.raises(ValueError, match="intensity variable 'vo'"):
        compare_tracks(reference, candidate, config=TrackComparisonConfig(var="vo"))


def test_comparison_to_dict_is_json_serializable(times: list[np.datetime64]) -> None:
    tracks = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3)])
    payload = compare_tracks(tracks, tracks).to_dict()
    assert json.loads(json.dumps(payload))["match_count"] == 1


def test_load_tracks_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="cannot infer input track format"):
        _load_tracks("tracks.csv")


def test_compare_json_stdout_is_machine_readable(
    times: list[np.datetime64], capsys: pytest.CaptureFixture[str]
) -> None:
    tracks = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3)])
    args = argparse.Namespace(
        reference="reference.trackjson",
        candidate="candidate.trackjson",
        max_mean_separation=2.0,
        min_overlap=0.6,
        var=None,
        mode="max",
        report=None,
        matched_candidate_output=None,
        json=True,
    )
    with patch("pystormtracker.compare._load_tracks", return_value=tracks):
        main(args)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["match_count"] == 1
    assert "Loading reference" in captured.err
