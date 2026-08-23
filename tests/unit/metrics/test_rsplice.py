from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from pystormtracker.hodges.rsplice import RspliceDistanceMode, filter_rsplice
from pystormtracker.models.geo import DEG_TO_RAD, geod_dist
from pystormtracker.models.tracks import Tracks, TracksMetadata


def _tracks(
    lengths: tuple[int, ...],
    *,
    endpoint_latitudes: tuple[float, ...] | None = None,
    times: tuple[np.ndarray, ...] | None = None,
) -> Tracks:
    total = sum(lengths)
    offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
    )
    lats = np.zeros(total, dtype=np.float64)
    lons = np.zeros(total, dtype=np.float64)
    packed_times: list[np.ndarray] = []
    for index, length in enumerate(lengths):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        end_latitude = (
            endpoint_latitudes[index] if endpoint_latitudes is not None else 0.0
        )
        lats[start:stop] = np.linspace(0.0, end_latitude, length)
        if times is None:
            packed_times.append(
                np.arange(length, dtype=np.int64) * 6 * 60 * 60 * 1000
                + np.datetime64("2024-01-01T00:00:00", "ms").astype(np.int64)
            )
        else:
            packed_times.append(times[index])
    return Tracks(
        ids=np.arange(1, len(lengths) + 1, dtype=np.int64),
        offsets=offsets,
        times=np.concatenate(packed_times),
        lats=lats,
        lons=lons,
        variables={"intensity": np.ones(total, dtype=np.float64)},
        metadata=TracksMetadata("intensity", "min", {"intensity": "1"}),
    )


def test_rsplice_uses_inclusive_point_and_endpoint_bounds() -> None:
    tracks = _tracks((7, 8), endpoint_latitudes=(20.0, 10.0))
    threshold = float(geod_dist(0.0, 0.0, 10.0, 0.0) / DEG_TO_RAD)

    result = filter_rsplice(tracks, min_points=8, distance_degrees=threshold)

    assert result.ids.tolist() == [2]


def test_rsplice_endpoint_differs_from_cumulative_travel() -> None:
    tracks = _tracks((8,), endpoint_latitudes=(0.0,))
    # Replace the stationary latitude sequence with a 5-degree out-and-back.
    lats = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 0.0])
    tracks = Tracks(
        ids=tracks.ids,
        offsets=tracks.offsets,
        times=tracks.times,
        lats=lats,
        lons=tracks.lons,
        variables=tracks.variables,
        metadata=tracks.metadata,
    )

    endpoint = filter_rsplice(tracks, distance_degrees=1.0)
    travel = filter_rsplice(tracks, distance_degrees=9.0, distance_mode="travel")

    assert len(endpoint) == 0
    assert len(travel) == 1


def test_rsplice_counts_integral_missing_cadence_gaps() -> None:
    base = np.datetime64("2024-01-01T00:00:00", "ms").astype(np.int64)
    times = np.array(
        [base, base + 6 * 60 * 60 * 1000, base + 18 * 60 * 60 * 1000],
        dtype=np.int64,
    )
    tracks = _tracks((3,), endpoint_latitudes=(20.0,), times=(times,))

    result = filter_rsplice(
        tracks,
        min_points=4,
        distance_degrees=1.0,
        expected_cadence=np.timedelta64(6, "h"),
    )

    assert result.ids.tolist() == [1]


def test_rsplice_rejects_nonintegral_cadence_gaps() -> None:
    base = np.datetime64("2024-01-01T00:00:00", "ms").astype(np.int64)
    times = np.array([base, base + 7 * 60 * 60 * 1000], dtype=np.int64)
    tracks = _tracks((2,), endpoint_latitudes=(20.0,), times=(times,))

    with pytest.raises(ValueError, match="integral multiples"):
        filter_rsplice(
            tracks,
            min_points=1,
            distance_degrees=1.0,
            expected_cadence=np.timedelta64(6, "h"),
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda tracks: filter_rsplice(tracks, min_points=-1), "min_points"),
        (
            lambda tracks: filter_rsplice(tracks, min_points=4, max_points=3),
            "max_points",
        ),
        (
            lambda tracks: filter_rsplice(tracks, distance_degrees=-1.0),
            "distance_degrees",
        ),
        (
            lambda tracks: filter_rsplice(
                tracks,
                distance_mode=cast(RspliceDistanceMode, "invalid"),
            ),
            "distance_mode",
        ),
    ],
)
def test_rsplice_validates_configuration(
    call: Callable[[Tracks], Tracks], message: str
) -> None:
    tracks = _tracks((8,), endpoint_latitudes=(20.0,))

    with pytest.raises(ValueError, match=message):
        call(tracks)
