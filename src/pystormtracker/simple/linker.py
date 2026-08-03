from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.geo import geod_dist_km
from ..models.tracker import RawDetectionStep
from ..models.tracks import TrackHandle, TracksBuilder


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
        self._head_ids: set[int] = set()
        self._handles: dict[int, TrackHandle] = {}
        self._last_time: np.datetime64 | None = None
        self._step: np.timedelta64 | None = None

    def _new_track(
        self,
        builder: TracksBuilder,
        time: np.datetime64,
        lat: float,
        lon: float,
        variables: dict[str, float],
    ) -> int:
        handle = builder.new_track()
        handle.append(time, lat, lon, variables)
        self._handles[handle.track_id] = handle
        self._head_ids.add(handle.track_id)
        return handle.track_id

    def append(self, builder: TracksBuilder, step_data: RawDetectionStep) -> None:
        """Link one time step into the builder without mutating finalized tracks."""
        time_val, new_lats, new_lons, vars_dict = step_data
        num_centers = len(new_lats)
        if num_centers == 0:
            self._tail_ids.clear()
            return

        sort_idx = np.lexsort((new_lons, new_lats))
        new_lats = new_lats[sort_idx]
        new_lons = new_lons[sort_idx]
        vars_dict = {name: values[sort_idx] for name, values in vars_dict.items()}

        if (
            self._last_time is not None
            and self._step is not None
            and time_val - self._last_time > self._step
        ):
            self._tail_ids.clear()

        if not self._tail_ids:
            for point_index in range(num_centers):
                self._new_track(
                    builder,
                    time_val,
                    float(new_lats[point_index]),
                    float(new_lons[point_index]),
                    {
                        name: float(values[point_index])
                        for name, values in vars_dict.items()
                    },
                )
            self._tail_ids = set(self._handles).difference(
                set(self._handles).difference(self._head_ids)
            )
            self._record_time(time_val)
            return

        tail_ids = sorted(
            self._tail_ids,
            key=lambda track_id: self._last_point(track_id),
        )
        tail_lats = np.asarray(
            [self._last_point(track_id)[0] for track_id in tail_ids], dtype=np.float64
        )
        tail_lons = np.asarray(
            [self._last_point(track_id)[1] for track_id in tail_ids], dtype=np.float64
        )
        distances = great_circle_distance_matrix(
            tail_lats, tail_lons, new_lats, new_lons
        )
        matched_indices = np.full(num_centers, -1, dtype=np.int64)
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

        new_tail_ids: set[int] = set()
        for center_index in range(num_centers):
            variables = {
                name: float(values[center_index]) for name, values in vars_dict.items()
            }
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
                self._handles[track_id].append(
                    time_val,
                    float(new_lats[center_index]),
                    float(new_lons[center_index]),
                    variables,
                )
            new_tail_ids.add(track_id)
        self._tail_ids = new_tail_ids
        self._record_time(time_val)

    def _last_point(self, track_id: int) -> tuple[float, float]:
        return self._handles[track_id].last_point

    def _record_time(self, time_val: np.datetime64) -> None:
        if self._last_time is not None and self._step is None:
            self._step = time_val - self._last_time
        self._last_time = time_val
