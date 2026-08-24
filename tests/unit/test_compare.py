from __future__ import annotations

import argparse
import json
import logging
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from pystormtracker.compare import _load_tracks, main
from pystormtracker.metrics.compare import (
    MatchingMethod,
    TrackComparison,
    TrackComparisonConfig,
    _CandidatePair,
    compare_tracks,
)
from pystormtracker.models.tracks import (
    ResolvedDetectionMode as Mode,
)
from pystormtracker.models.tracks import (
    Tracks,
    TracksMetadata,
    _TracksBuilder,
)


def make_tracks(
    records: list[tuple[list[float], list[float], list[np.datetime64], list[float]]],
    *,
    variable: str = "intensity",
    mode: Mode = "max",
) -> Tracks:
    unit = {
        "msl": "Pa",
        "vo": "s^-1",
        "vort": "s^-1",
    }.get(variable, "1")
    builder = _TracksBuilder(TracksMetadata(variable, mode, {variable: unit}))
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


def test_compare_tracks_aligns_only_exact_common_timestamps() -> None:
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

    result = compare_tracks(reference, candidate)

    assert result.match_count == 1
    assert result.matches[0].overlap_count == 2
    assert result.matches[0].topology_identical is False


def test_compare_tracks_topology_requires_exact_timestamp_sequence() -> None:
    reference_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T06:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    candidate_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T03:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    reference = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], reference_times, [0.0] * 3)])
    candidate = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], candidate_times, [0.0] * 3)])

    result = compare_tracks(reference, candidate)

    assert result.matches[0].same_time_range is True
    assert result.matches[0].same_point_count is True
    assert result.matches[0].topology_identical is False


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
        config=TrackComparisonConfig(variable="vo"),
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
        config=TrackComparisonConfig(variable="custom", mode="min"),
    )
    result_max = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(variable="custom", mode="max"),
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
        compare_tracks(
            reference, candidate, config=TrackComparisonConfig(variable="vo")
        )


@pytest.mark.parametrize("legacy_name", ["mutual", "assignment"])
def test_comparison_rejects_legacy_matching_aliases(legacy_name: str) -> None:
    with pytest.raises(ValueError, match="mutual_nearest.*global_assignment"):
        TrackComparisonConfig(matching=cast(MatchingMethod, legacy_name))


def test_comparison_to_dict_is_json_serializable(times: list[np.datetime64]) -> None:
    tracks = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3)])
    payload = compare_tracks(tracks, tracks).to_dict()
    assert json.loads(json.dumps(payload))["match_count"] == 1


def test_load_tracks_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="cannot infer input track format"):
        _load_tracks("tracks.csv")


def test_compare_json_stdout_is_machine_readable(
    times: list[np.datetime64],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="pystormtracker.compare")
    tracks = make_tracks([([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3)])
    args = argparse.Namespace(
        reference="reference.trackjson",
        candidate="candidate.trackjson",
        matching="nearest",
        max_mean_separation=2.0,
        min_overlap=0.6,
        variable=None,
        detection_mode="max",
        report=None,
        matched_candidate_output=None,
        json=True,
    )
    with patch("pystormtracker.compare._load_tracks", return_value=tracks):
        main(args)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["match_count"] == 1
    assert "Loading reference" in caplog.text


def test_compare_tracks_nearest_candidate_reuse_metrics(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([0.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([0.1] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks(
        [
            ([0.05] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([30.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )

    result = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(matching="nearest"),
    )

    assert result.match_count == 2
    assert result.reference_count == 2
    assert result.candidate_count == 2
    assert result.unique_candidate_count == 1
    assert result.reused_candidate_count == 1
    assert result.reused_candidate_assignments == 1
    assert result.reference_coverage == 1.0
    assert result.candidate_coverage == 0.5
    assert result.unmatched_reference_ids == ()
    assert result.unmatched_candidate_ids == (2,)


def test_compare_tracks_mutual_nearest_matching_semantics(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([10.00] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([10.09] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([30.00] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks(
        [
            ([10.05] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([10.20] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([30.05] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )

    result = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(matching="mutual_nearest"),
    )

    assert result.match_count == 2
    assert [(m.reference_id, m.candidate_id) for m in result.matches] == [
        (2, 1),
        (3, 3),
    ]
    assert result.agreement == pytest.approx(2.0 / 3.0)
    assert result.unmatched_reference_ids == (1,)
    assert result.unmatched_candidate_ids == (2,)


def test_compare_tracks_assignment_globally_optimal_cardinality(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([10.00] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([10.08] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks(
        [
            ([10.05] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([10.20] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )

    config = TrackComparisonConfig(
        matching="global_assignment",
        max_mean_separation_deg=0.15,
    )
    result = compare_tracks(reference, candidate, config=config)

    assert result.match_count == 2
    assert [(m.reference_id, m.candidate_id) for m in result.matches] == [
        (1, 1),
        (2, 2),
    ]
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.topology_identical_count == 2


def test_compare_tracks_assignment_deterministic_tie_breaking(
    times: list[np.datetime64],
) -> None:
    reference = make_tracks(
        [
            ([10.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([20.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )
    candidate = make_tracks(
        [
            ([10.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
            ([20.0] * 3, [0.0, 1.0, 2.0], times, [0.0] * 3),
        ]
    )

    res1 = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(matching="global_assignment"),
    )
    res2 = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(matching="global_assignment"),
    )

    assert [(m.reference_id, m.candidate_id) for m in res1.matches] == [
        (1, 1),
        (2, 2),
    ]
    assert [(m.reference_id, m.candidate_id) for m in res2.matches] == [
        (1, 1),
        (2, 2),
    ]


def _assignment_test_tracks() -> tuple[Tracks, Tracks]:
    times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T06:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    records = [
        ([0.0] * 3, [0.0] * 3, times, [0.0] * 3),
        ([1.0] * 3, [0.0] * 3, times, [0.0] * 3),
    ]
    return make_tracks(records), make_tracks(records)


def _compare_with_synthetic_assignment_pairs(
    monkeypatch: pytest.MonkeyPatch,
    pair_values: dict[tuple[int, int], tuple[float, float]],
) -> TrackComparison:
    import importlib

    compare_module = importlib.import_module("pystormtracker.metrics.compare")

    def synthetic_pair(
        _reference: object,
        _candidate: object,
        reference_index: int,
        candidate_index: int,
        _config: TrackComparisonConfig,
    ) -> _CandidatePair | None:
        values = pair_values.get((reference_index, candidate_index))
        if values is None:
            return None
        overlap, separation_deg = values
        separation_km = np.asarray(
            [separation_deg * np.pi / 180.0 * 6371.0], dtype=np.float64
        )
        return _CandidatePair(
            reference_index,
            candidate_index,
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            overlap,
            separation_km,
        )

    monkeypatch.setattr(compare_module, "_candidate_pair", synthetic_pair)
    reference, candidate = _assignment_test_tracks()
    return compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(
            matching="global_assignment",
            max_mean_separation_deg=20.0,
            min_overlap_fraction=0.0,
        ),
    )


def test_global_assignment_max_cardinality_beats_better_separation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compare_with_synthetic_assignment_pairs(
        monkeypatch,
        {
            (0, 0): (1.0, 0.0),
            (0, 1): (1.0, 19.0),
            (1, 0): (1.0, 19.0),
        },
    )

    assert result.match_count == 2
    assert [(match.reference_id, match.candidate_id) for match in result.matches] == [
        (1, 2),
        (2, 1),
    ]
    assert result.reused_candidate_count == 0


def test_global_assignment_total_overlap_beats_separation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compare_with_synthetic_assignment_pairs(
        monkeypatch,
        {
            (0, 0): (0.501, 10.0),
            (0, 1): (0.500, 0.0),
            (1, 0): (0.500, 0.0),
            (1, 1): (0.501, 10.0),
        },
    )

    assert [(match.reference_id, match.candidate_id) for match in result.matches] == [
        (1, 1),
        (2, 2),
    ]


def test_global_assignment_separation_is_last_scientific_criterion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compare_with_synthetic_assignment_pairs(
        monkeypatch,
        {
            (0, 0): (0.500, 10.0),
            (0, 1): (0.500, 0.0),
            (1, 0): (0.500, 0.0),
            (1, 1): (0.500, 10.0),
        },
    )

    assert [(match.reference_id, match.candidate_id) for match in result.matches] == [
        (1, 2),
        (2, 1),
    ]


def test_global_assignment_exact_ties_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair_values = {
        (0, 0): (0.500, 1.0),
        (0, 1): (0.500, 1.0),
        (1, 0): (0.500, 1.0),
        (1, 1): (0.500, 1.0),
    }
    first = _compare_with_synthetic_assignment_pairs(monkeypatch, pair_values)
    second = _compare_with_synthetic_assignment_pairs(monkeypatch, pair_values)

    assert [(match.reference_id, match.candidate_id) for match in first.matches] == [
        (match.reference_id, match.candidate_id) for match in second.matches
    ]
    assert first.reused_candidate_count == 0


def test_global_assignment_solves_disconnected_components_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compare_with_synthetic_assignment_pairs(
        monkeypatch,
        {
            (0, 0): (0.2, 5.0),
            (1, 1): (0.8, 1.0),
        },
    )

    assert [(match.reference_id, match.candidate_id) for match in result.matches] == [
        (1, 1),
        (2, 2),
    ]
