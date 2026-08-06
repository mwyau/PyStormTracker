from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.geo import geod_dist_km
from ..models.time import encode_time_values
from ..models.tracker import RawDetectionStep
from ..models.tracks import _TracksBuilder


@nb.njit(cache=True, nogil=True)
def great_circle_distance_matrix(
    lats1: NDArray[np.float64],
    lons1: NDArray[np.float64],
    lats2: NDArray[np.float64],
    lons2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return clamped great-circle distances using unit-vector dot products."""
    distances = np.empty((lats1.size, lats2.size), dtype=np.float64)
    for i in range(lats1.size):
        for j in range(lats2.size):
            distances[i, j] = geod_dist_km(lats1[i], lons1[i], lats2[j], lons2[j])
    return distances


class SimpleLinker:
    """Heuristic nearest-neighbor linker operating on a mutable builder."""

    def __init__(self, threshold: float = 500.0) -> None:
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

    def append(self, builder: _TracksBuilder, step_data: RawDetectionStep) -> None:
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
                    {builder.metadata.primary_var: float(values[point_index])},
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
        distances = great_circle_distance_matrix(
            tail_lats, tail_lons, new_lats, new_lons
        )
        matched_indices = np.full(num_centers, -1, dtype=np.int64)

        # Global greedy matching with mutual-closest constraint
        while True:
            has_match = False
            for center_index in range(num_centers):
                if matched_indices[center_index] != -1:
                    continue
                column = distances[:, center_index]
                if np.any(column < self.threshold):
                    tail_index = int(np.argmin(column))
                    if np.argmin(distances[tail_index, :]) == center_index:
                        matched_indices[center_index] = tail_index
                        distances[:, center_index] = np.inf
                        distances[tail_index, :] = np.inf
                        has_match = True
            if not has_match:
                break

        next_tail_ids: set[int] = set()
        for center_index in range(num_centers):
            variables = {builder.metadata.primary_var: float(values[center_index])}
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
