from __future__ import annotations

import numpy as np
import pytest

from pystormtracker.metrics.compare import match_tracks
from pystormtracker.models.center import Center
from pystormtracker.models.tracks import Tracks


def create_dummy_track(
    tracks: Tracks, lats: list[float], lons: list[float], times: list[np.datetime64]
) -> None:
    centers = []
    for lat, lon, time in zip(lats, lons, times, strict=False):
        centers.append(Center(time=time, lat=lat, lon=lon, vars={}))
    tracks.add_track(centers)


@pytest.fixture
def tracks_ref() -> Tracks:
    tracks = Tracks()
    # Track 1: Northern Hemisphere move
    create_dummy_track(
        tracks,
        [50.0, 51.0, 52.0],
        [0.0, 1.0, 2.0],
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T06:00"),
            np.datetime64("2020-01-01T12:00"),
        ],
    )
    # Track 2: Tropics
    create_dummy_track(
        tracks,
        [10.0, 10.0, 10.0],
        [100.0, 101.0, 102.0],
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T06:00"),
            np.datetime64("2020-01-01T12:00"),
        ],
    )
    return tracks


def test_match_perfect(tracks_ref: Tracks) -> None:
    # Perfect match
    matches = match_tracks(tracks_ref, tracks_ref)
    assert len(matches) == 2
    assert matches[1] == 1
    assert matches[2] == 2


def test_match_slight_drift(tracks_ref: Tracks) -> None:
    tracks_comp = Tracks()
    # Comparison track 1: drifted by 0.5 degrees (~55km)
    create_dummy_track(
        tracks_comp,
        [50.5, 51.5, 52.5],
        [0.5, 1.5, 2.5],
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T06:00"),
            np.datetime64("2020-01-01T12:00"),
        ],
    )

    matches = match_tracks(tracks_ref, tracks_comp, max_dist_km=100.0)
    assert matches[1] == 1


def test_match_no_overlap_time(tracks_ref: Tracks) -> None:
    tracks_comp = Tracks()
    # Same coords, different day
    create_dummy_track(
        tracks_comp,
        [50.0, 51.0, 52.0],
        [0.0, 1.0, 2.0],
        [
            np.datetime64("2020-02-01T00:00"),
            np.datetime64("2020-02-01T06:00"),
            np.datetime64("2020-02-01T12:00"),
        ],
    )

    matches = match_tracks(tracks_ref, tracks_comp)
    assert len(matches) == 0


def test_match_insufficient_overlap_fraction(tracks_ref: Tracks) -> None:
    tracks_comp = Tracks()
    # Track is much longer, but only 1 point overlaps
    create_dummy_track(
        tracks_comp,
        [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T06:00"),
            np.datetime64("2020-01-01T12:00"),
            np.datetime64("2020-01-01T18:00"),
            np.datetime64("2020-01-02T00:00"),
            np.datetime64("2020-01-02T06:00"),
            np.datetime64("2020-01-02T12:00"),
            np.datetime64("2020-01-02T18:00"),
        ],
    )

    # Overlap is 3 points. Total lengths: 3 and 8.
    # Ratio = (2 * 3) / (3 + 8) = 6/11 = 0.54

    matches = match_tracks(tracks_ref, tracks_comp, min_overlap_fraction=0.6)
    assert len(matches) == 0

    matches = match_tracks(tracks_ref, tracks_comp, min_overlap_fraction=0.5)
    assert matches[1] == 1


def test_match_too_far(tracks_ref: Tracks) -> None:
    tracks_comp = Tracks()
    # Shares time, but 1000km away
    create_dummy_track(
        tracks_comp,
        [60.0, 61.0, 62.0],
        [0.0, 1.0, 2.0],
        [
            np.datetime64("2020-01-01T00:00"),
            np.datetime64("2020-01-01T06:00"),
            np.datetime64("2020-01-01T12:00"),
        ],
    )

    matches = match_tracks(tracks_ref, tracks_comp, max_dist_km=200.0)
    assert len(matches) == 0
