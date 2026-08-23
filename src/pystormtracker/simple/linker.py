from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.geo import geod_dist_km
from ..models.time import encode_time_values
from ..models.tracker import CenterFrame
from ..models.tracks import _TracksBuilder
from .constants import MAX_LINK_DISTANCE_KM


@nb.njit(cache=True, nogil=True)
def _great_circle_distance_matrix(
    lats1: NDArray[np.float64],
    lons1: NDArray[np.float64],
    lats2: NDArray[np.float64],
    lons2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return clamped great-circle distances in km using unit-vector dot products."""
    distances = np.empty((lats1.size, lats2.size), dtype=np.float64)
    for i in range(lats1.size):
        for j in range(lats2.size):
            distances[i, j] = geod_dist_km(lats1[i], lons1[i], lats2[j], lons2[j])
    return distances


@nb.njit(cache=True, nogil=True)
def _match_nearest_neighbors(
    tail_lats: NDArray[np.float64],
    tail_lons: NDArray[np.float64],
    new_lats: NDArray[np.float64],
    new_lons: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.int64]:
    """Perform mutual-nearest matching between active tails and new centers.

    The closest-feature association within 500 km is the Simple Tracker
    concept described by Yau and Chang (2020).  Mutual-nearest rounds,
    strict distance-boundary handling, index tie ordering, and native sample
    cadence are current PST/historical implementation details rather than
    claims about the paper's exact algorithm.
    """
    n_tails = tail_lats.size
    n_centers = new_lats.size
    matched_indices = np.full(n_centers, -1, dtype=np.int64)

    if n_tails == 0 or n_centers == 0:
        return matched_indices

    distances = _great_circle_distance_matrix(tail_lats, tail_lons, new_lats, new_lons)

    while True:
        has_match = False
        for c in range(n_centers):
            if matched_indices[c] != -1:
                continue

            # Find closest tail to center c (tie-break lowest tail index)
            best_tail = -1
            min_col_dist = np.inf
            for r in range(n_tails):
                d = distances[r, c]
                if d < min_col_dist:
                    min_col_dist = d
                    best_tail = r

            if best_tail != -1 and min_col_dist < threshold:
                # Verify center c is closest to best_tail (tie-break lowest index)
                best_center = -1
                min_row_dist = np.inf
                for j in range(n_centers):
                    d = distances[best_tail, j]
                    if d < min_row_dist:
                        min_row_dist = d
                        best_center = j

                if best_center == c:
                    matched_indices[c] = best_tail
                    for r in range(n_tails):
                        distances[r, c] = np.inf
                    for j in range(n_centers):
                        distances[best_tail, j] = np.inf
                    has_match = True

        if not has_match:
            break

    return matched_indices


class SimpleLinker:
    """Heuristic nearest-neighbor linker operating on a mutable builder.

    The high-level 500-km consecutive-period association has Yau and Chang
    (2020) lineage.  The exact matching rounds and deterministic ordering are
    PyStormTracker implementation behavior.
    """

    def __init__(self, threshold: float = MAX_LINK_DISTANCE_KM) -> None:
        self.threshold = threshold
        self._tail_ids: set[int] = set()
        self._last_time: int | None = None
        self._step: int | None = None

    def _new_track(
        self,
        builder: _TracksBuilder,
        time: int,
        lat: float,
        lon: float,
        variables: dict[str, float],
    ) -> int:
        track_id = builder.new_track()
        builder.append(track_id, time, lat, lon, variables)
        return track_id

    def append(self, builder: _TracksBuilder, step_data: CenterFrame) -> None:
        """Link one time step into the builder without mutating finalized tracks."""
        raw_time, new_lats, new_lons, values = step_data
        time_val = int(encode_time_values([raw_time])[0])
        num_centers = len(new_lats)
        if num_centers == 0:
            self._tail_ids.clear()
            return

        # Deterministic spatial priority sorting for reproducible matching
        sort_idx = np.lexsort((new_lons, new_lats))
        new_lats = new_lats[sort_idx]
        new_lons = new_lons[sort_idx]
        values = values[sort_idx]

        # Reset active track tails if time step exceeds expected interval
        if (
            self._last_time is not None
            and self._step is not None
            and time_val - self._last_time > self._step
        ):
            self._tail_ids.clear()

        if not self._tail_ids:
            new_tail_ids: set[int] = set()
            for point_index in range(num_centers):
                track_id = self._new_track(
                    builder,
                    time_val,
                    float(new_lats[point_index]),
                    float(new_lons[point_index]),
                    {builder.metadata.primary_variable: float(values[point_index])},
                )
                new_tail_ids.add(track_id)
            self._tail_ids = new_tail_ids
            self._record_time(time_val)
            return

        # Deterministic sorting of existing active track tails
        tail_ids = sorted(
            self._tail_ids,
            key=lambda track_id: self._last_point(builder, track_id),
        )
        tail_lats = np.asarray(
            [self._last_point(builder, track_id)[0] for track_id in tail_ids],
            dtype=np.float64,
        )
        tail_lons = np.asarray(
            [self._last_point(builder, track_id)[1] for track_id in tail_ids],
            dtype=np.float64,
        )

        matched_indices = _match_nearest_neighbors(
            tail_lats,
            tail_lons,
            new_lats,
            new_lons,
            self.threshold,
        )

        next_tail_ids: set[int] = set()
        for center_index in range(num_centers):
            variables = {builder.metadata.primary_variable: float(values[center_index])}
            tail_index = int(matched_indices[center_index])
            if tail_index == -1:
                track_id = self._new_track(
                    builder,
                    time_val,
                    float(new_lats[center_index]),
                    float(new_lons[center_index]),
                    variables,
                )
            else:
                track_id = tail_ids[tail_index]
                builder.append(
                    track_id,
                    time_val,
                    float(new_lats[center_index]),
                    float(new_lons[center_index]),
                    variables,
                )
            next_tail_ids.add(track_id)
        self._tail_ids = next_tail_ids
        self._record_time(time_val)

    def _last_point(
        self, builder: _TracksBuilder, track_id: int
    ) -> tuple[float, float]:
        return builder.last_point(track_id)

    def _record_time(self, time_val: int) -> None:
        if self._last_time is not None and self._step is None:
            self._step = time_val - self._last_time
        self._last_time = time_val
