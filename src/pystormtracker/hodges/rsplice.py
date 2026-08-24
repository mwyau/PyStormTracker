from __future__ import annotations

from typing import Literal

import numpy as np

from ..models.geo import DEG_TO_RAD, geod_dist
from ..models.tracks import Tracks

type RspliceDistanceMode = Literal["endpoint", "travel"]


def filter_rsplice(
    tracks: Tracks,
    *,
    min_points: int = 8,
    max_points: int | None = None,
    distance_degrees: float = 10.0,
    distance_mode: RspliceDistanceMode = "endpoint",
    expected_cadence: np.timedelta64 | None = None,
) -> Tracks:
    """Apply TRACK 1.5.4's RSPLICE lifetime and displacement filters.

    TRACK first retains tracks whose observed-point count plus known missing
    input frames lies in the inclusive ``[min_points, max_points]`` interval,
    then retains tracks whose geodesic displacement is at least the inclusive
    ``distance_degrees`` threshold. The Kevin workflow uses eight points,
    no finite upper bound, and endpoint displacement at ten degrees.

    ``expected_cadence`` is required when packed track times contain gaps that
    should count as source ``nfm`` frames. Gaps must be integral multiples of
    that cadence, matching TRACK's explicit missing-frame rule. With no
    cadence, only observed points are counted. ``distance_mode="travel"``
    selects TRACK's alternative cumulative segment-distance option for
    diagnostic boundary tests; the Kevin configuration uses ``"endpoint"``.

    The source comparison is strict only below the thresholds: a track at
    exactly either boundary is retained.
    """
    if min_points < 0:
        raise ValueError("min_points must be nonnegative")
    if max_points is not None and max_points < min_points:
        raise ValueError("max_points must be at least min_points")
    if not np.isfinite(distance_degrees) or distance_degrees < 0.0:
        raise ValueError("distance_degrees must be finite and nonnegative")
    if distance_mode not in ("endpoint", "travel"):
        raise ValueError("distance_mode must be 'endpoint' or 'travel'")

    cadence_ms: int | None = None
    if expected_cadence is not None:
        cadence_value = float(expected_cadence / np.timedelta64(1, "ms"))
        if (
            not np.isfinite(cadence_value)
            or cadence_value <= 0.0
            or cadence_value != np.floor(cadence_value)
        ):
            raise ValueError("expected_cadence must be a positive whole number of ms")
        cadence_ms = int(cadence_value)

    threshold_radians = float(distance_degrees) * DEG_TO_RAD
    keep = np.zeros(len(tracks), dtype=np.bool_)
    for index, track in enumerate(tracks):
        point_count = len(track)
        if cadence_ms is not None:
            deltas = np.diff(track.times)
            if deltas.size:
                if np.any(deltas % cadence_ms != 0):
                    raise ValueError(
                        "track time gaps must be integral multiples of expected_cadence"
                    )
                point_count += int(np.sum(deltas // cadence_ms - 1))
        if point_count < min_points or (
            max_points is not None and point_count > max_points
        ):
            continue

        if len(track) < 2:
            displacement = 0.0
        elif distance_mode == "endpoint":
            displacement = geod_dist(
                float(track.lats[0]),
                float(track.lons[0]),
                float(track.lats[-1]),
                float(track.lons[-1]),
            )
        else:
            displacement = 0.0
            for point_index in range(len(track) - 1):
                displacement += geod_dist(
                    float(track.lats[point_index]),
                    float(track.lons[point_index]),
                    float(track.lats[point_index + 1]),
                    float(track.lons[point_index + 1]),
                )
        keep[index] = displacement >= threshold_radians
    return tracks.filter(keep)
