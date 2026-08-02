from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import numpy as np
import pytest

from pystormtracker.compare import _load_tracks, main
from pystormtracker.metrics.compare import TrackComparisonConfig, compare_tracks
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


def add_track(
    tracks: Tracks,
    lats: list[float],
    lons: list[float],
    times: list[np.datetime64],
    intensities: list[float] | None = None,
) -> None:
    """Append a small trajectory with an optional vorticity variable."""
    centers = [
        Center(
            time=time,
            lat=lat,
            lon=lon,
            vars={} if intensities is None else {"vo": intensity},
        )
        for lat, lon, time, intensity in zip(
            lats,
            lons,
            times,
            intensities if intensities is not None else [0.0] * len(times),
            strict=True,
        )
    ]
    tracks.add_track(centers)


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
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [50.0, 51.0, 52.0], [0.0, 1.0, 2.0], times)
    add_track(reference, [10.0, 10.0, 10.0], [100.0, 101.0, 102.0], times)
    add_track(candidate, [50.2, 51.2, 52.2], [0.2, 1.2, 2.2], times)
    add_track(candidate, [10.2, 10.2, 10.2], [100.2, 101.2, 102.2], times)

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
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    add_track(reference, [0.1, 0.1, 0.1], [0.0, 1.0, 2.0], times)
    add_track(candidate, [0.05, 0.05, 0.05], [0.0, 1.0, 2.0], times)

    result = compare_tracks(reference, candidate)

    assert result.match_count == 2
    assert [match.candidate_id for match in result.matches] == [1, 1]
    assert result.candidate_coverage == 1.0


def test_compare_tracks_requires_equal_overlap_section_lengths() -> None:
    reference = Tracks()
    candidate = Tracks()
    reference_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T06:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    candidate_times = [
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-01T12:00"),
    ]
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], reference_times)
    add_track(candidate, [0.0, 0.0], [0.0, 2.0], candidate_times)

    with pytest.raises(ValueError, match="equal point counts"):
        compare_tracks(reference, candidate)


def test_compare_tracks_breaks_distance_ties_by_candidate_file_order(
    times: list[np.datetime64],
) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    add_track(candidate, [0.1, 0.1, 0.1], [0.0, 1.0, 2.0], times)
    add_track(candidate, [-0.1, -0.1, -0.1], [0.0, 1.0, 2.0], times)

    result = compare_tracks(reference, candidate)

    assert result.matches[0].candidate_id == 1
    assert result.matches[0].eligible_candidate_count == 2


def test_compare_tracks_reports_lifecycle_path_and_intensity_metrics(
    times: list[np.datetime64],
) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(
        reference,
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        times,
        [1.0e-4, 2.0e-4, 3.0e-4],
    )
    add_track(
        candidate,
        [0.0, 0.0, 0.0],
        [0.1, 1.1, 2.1],
        times,
        [1.1e-4, 2.1e-4, 3.1e-4],
    )

    result = compare_tracks(
        reference,
        candidate,
        config=TrackComparisonConfig(intensity_var="vo"),
    )
    match = result.matches[0]

    assert match.reference.duration_hours == 12.0
    assert match.reference.path_length_km == pytest.approx(222.39, rel=1e-3)
    assert match.reference.mean_speed_kmh == pytest.approx(18.53, rel=1e-3)
    assert match.intensity_difference is not None
    assert match.intensity_difference.bias == pytest.approx(1.0e-5)
    assert match.intensity_difference.mae == pytest.approx(1.0e-5)
    assert match.intensity_difference.rmse == pytest.approx(1.0e-5)
    assert match.intensity_difference.correlation == pytest.approx(1.0)
    assert match.mean_separation_km == pytest.approx(11.12, rel=1e-2)


def test_compare_tracks_handles_dateline(times: list[np.datetime64]) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [30.0, 30.0, 30.0], [179.8, 179.9, 180.0], times)
    add_track(candidate, [30.0, 30.0, 30.0], [-179.8, -179.9, -180.0], times)

    result = compare_tracks(reference, candidate)

    assert result.match_count == 1
    assert result.matches[0].mean_separation_deg < 1.0


def test_compare_tracks_rejects_duplicate_times(times: list[np.datetime64]) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    add_track(
        candidate,
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [times[0], times[0], times[2]],
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        compare_tracks(reference, candidate)


def test_compare_tracks_rejects_missing_intensity_variable(
    times: list[np.datetime64],
) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    add_track(candidate, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)

    with pytest.raises(ValueError, match="intensity variable 'vo'"):
        compare_tracks(
            reference, candidate, config=TrackComparisonConfig(intensity_var="vo")
        )


@pytest.mark.parametrize(
    ("max_mean_separation_deg", "min_overlap_fraction"),
    [(0.0, 0.6), (-1.0, 0.6), (2.0, -0.1), (2.0, 1.1)],
)
def test_config_rejects_invalid_parameters(
    max_mean_separation_deg: float, min_overlap_fraction: float
) -> None:
    with pytest.raises(ValueError, match="must be"):
        TrackComparisonConfig(
            max_mean_separation_deg=max_mean_separation_deg,
            min_overlap_fraction=min_overlap_fraction,
        )


def test_comparison_to_dict_is_json_serializable(times: list[np.datetime64]) -> None:
    reference = Tracks()
    candidate = Tracks()
    add_track(reference, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    add_track(candidate, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)

    payload = compare_tracks(reference, candidate).to_dict()

    assert json.loads(json.dumps(payload))["match_count"] == 1


def test_load_tracks_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported track file extension"):
        _load_tracks("tracks.csv")


def test_compare_json_stdout_is_machine_readable(
    times: list[np.datetime64], capsys: pytest.CaptureFixture[str]
) -> None:
    tracks = Tracks()
    add_track(tracks, [0.0, 0.0, 0.0], [0.0, 1.0, 2.0], times)
    args = argparse.Namespace(
        reference="reference.json",
        candidate="candidate.json",
        max_mean_separation=2.0,
        min_overlap=0.6,
        intensity_var=None,
        report=None,
        matched_candidate_output=None,
        json=True,
    )
    with patch("pystormtracker.compare._load_tracks", return_value=tracks):
        main(args)

    captured = capsys.readouterr()
    assert json.loads(captured.out)["match_count"] == 1
    assert "Loading reference" in captured.err
