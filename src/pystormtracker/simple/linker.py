from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..models.center import Center
from ..models.tracker import RawDetectionStep
from ..models.tracks import TimeRange, Tracks


def haversine_matrix(
    lats1: NDArray[np.float64],
    lons1: NDArray[np.float64],
    lats2: NDArray[np.float64],
    lons2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized Haversine distance calculation returning a distance matrix in km."""
    R = 6367.0
    DEGTORAD = np.pi / 180.0

    lats1_rad = lats1 * DEGTORAD
    lats2_rad = lats2 * DEGTORAD

    dlat = lats2_rad[None, :] - lats1_rad[:, None]
    dlon = (lons2 * DEGTORAD)[None, :] - (lons1 * DEGTORAD)[:, None]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lats1_rad)[:, None]
        * np.cos(lats2_rad)[None, :]
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return np.asarray(R * c)


class SimpleLinker:
    """
    Heuristic nearest-neighbor linker for cyclone trajectories.
    Uses spatial priority sorting and vectorized distance matrices for performance.
    """

    def __init__(self, threshold: float = 500.0) -> None:
        self.threshold = threshold

    def append(self, tracks: Tracks, step_data: RawDetectionStep) -> None:
        """
        Links a single time step of detections to existing track tails.
        """
        time_val, new_lats, new_lons, vars_dict = step_data

        num_centers = len(new_lats)
        if num_centers == 0:
            # If no centers, all previous tails die
            tracks._tail_ids = set()
            return

        # Deterministic sorting of input centers (Spatial Priority)
        # This ensures greedy matches are reproducible.
        sort_idx = np.lexsort((new_lons, new_lats))
        new_lats = new_lats[sort_idx]
        new_lons = new_lons[sort_idx]
        vars_dict = {k: v[sort_idx] for k, v in vars_dict.items()}

        current_time = time_val

        if not tracks._tail_ids:
            # First ever centers in this object
            new_tail_ids = set()
            for i in range(num_centers):
                c_vars = {k: float(v[i]) for k, v in vars_dict.items()}
                c = Center(time_val, float(new_lats[i]), float(new_lons[i]), c_vars)
                t = tracks.add_track([c])
                new_tail_ids.add(t.track_id)
                tracks._head_ids.add(t.track_id)

            tracks._tail_ids = new_tail_ids
            # Update boundaries if not set
            if tracks.time_range is None:
                tracks.time_range = TimeRange(start=current_time, end=current_time)
            return

        # Temporal gap check
        if (
            tracks.time_range
            and tracks.time_range.end is not None
            and tracks.time_range.step is not None
            and current_time - tracks.time_range.step > tracks.time_range.end
        ):
            # All previous tails die due to gap
            new_tail_ids = set()
            for i in range(num_centers):
                c_vars = {k: float(v[i]) for k, v in vars_dict.items()}
                c = Center(time_val, float(new_lats[i]), float(new_lons[i]), c_vars)
                t = tracks.add_track([c])
                new_tail_ids.add(t.track_id)
                tracks._head_ids.add(t.track_id)
            tracks._tail_ids = new_tail_ids
            tracks.time_range.end = current_time
            return

        # Deterministic sorting of existing tails
        tail_tracks = sorted(tracks.tail, key=lambda t: (t[-1].lat, t[-1].lon))
        tail_lats = np.array([t[-1].lat for t in tail_tracks])
        tail_lons = np.array([t[-1].lon for t in tail_tracks])

        dist_matrix = haversine_matrix(tail_lats, tail_lons, new_lats, new_lons)
        matched_indices = np.full(num_centers, -1, dtype=np.int64)

        # Global greedy matching with mutual-closest constraint
        while True:
            has_match = False
            for ic in range(num_centers):
                if matched_indices[ic] == -1:
                    # Find closest tail for this center
                    col = dist_matrix[:, ic]
                    if np.any(col < self.threshold):
                        it_match = int(np.argmin(col))
                        # Check if this center is also the closest for that tail
                        if np.argmin(dist_matrix[it_match, :]) == ic:
                            matched_indices[ic] = it_match
                            dist_matrix[:, ic] = np.inf
                            dist_matrix[it_match, :] = np.inf
                            has_match = True
            if not has_match:
                break

        # Prepare for bulk update
        append_tids = []
        append_times = []
        append_lats = []
        append_lons = []
        append_vars: dict[str, list[float]] = {k: [] for k in vars_dict}

        new_tail_ids = set()

        # Handle matches
        for ic in range(num_centers):
            it_match = matched_indices[ic]
            if it_match != -1:
                t = tail_tracks[it_match]
                tid = t.track_id
                append_tids.append(tid)
                append_times.append(time_val)
                append_lats.append(new_lats[ic])
                append_lons.append(new_lons[ic])
                for k, v in vars_dict.items():
                    append_vars[k].append(v[ic])
                new_tail_ids.add(tid)
            else:
                # Create new track
                c_vars = {k: float(v[ic]) for k, v in vars_dict.items()}
                c = Center(time_val, float(new_lats[ic]), float(new_lons[ic]), c_vars)
                t = tracks.add_track([c])
                tracks._head_ids.add(t.track_id)
                new_tail_ids.add(t.track_id)

        if append_tids:
            tracks.bulk_append(
                np.array(append_tids, dtype=np.int64),
                np.array(append_times, dtype="datetime64[s]"),
                np.array(append_lats, dtype=np.float64),
                np.array(append_lons, dtype=np.float64),
                {k: np.array(v, dtype=np.float64) for k, v in append_vars.items()},
            )

        # Update tails: ONLY tracks that received a center at THIS time step
        tracks._tail_ids = new_tail_ids

        # Bookkeeping for TimeRange
        if tracks.time_range:
            if (
                tracks.time_range.step is None
                and current_time != tracks.time_range.start
            ):
                tracks.time_range.step = current_time - tracks.time_range.start
            # Only extend end if it's forward in time
            if current_time > tracks.time_range.end:
                tracks.time_range.end = current_time
