from __future__ import annotations

import numpy as np

from ..models.center import Center
from ..models.tracks import TimeRange, Track, Tracks
from .detector import RawDetectionStep


def haversine_matrix(
    lats1: np.ndarray, lons1: np.ndarray, lats2: np.ndarray, lons2: np.ndarray
) -> np.ndarray:
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
    def __init__(self, threshold: float = 500.0) -> None:
        self.threshold = threshold

    def append(self, tracks: Tracks, step_data: RawDetectionStep) -> None:
        time_val, new_lats, new_lons, vars_dict = step_data

        num_centers = len(new_lats)
        if num_centers == 0:
            tracks.tail = []
            return

        current_time = time_val

        if not tracks.tail:
            # First ever centers
            new_tail = []
            for i in range(num_centers):
                c_vars = {k: float(v[i]) for k, v in vars_dict.items()}
                c = Center(time_val, float(new_lats[i]), float(new_lons[i]), c_vars)
                t = tracks.add_track([c])
                new_tail.append(t)
                tracks._head_ids.add(t.track_id)
                tracks._tail_ids.add(t.track_id)
            if tracks.time_range is None:
                tracks.time_range = TimeRange(start=current_time, end=current_time)
            return

        if (
            tracks.time_range
            and tracks.time_range.end is not None
            and tracks.time_range.step is not None
            and current_time - tracks.time_range.step > tracks.time_range.end
        ):
            # Gap in time detected, do not link
            for i in range(num_centers):
                c_vars = {k: float(v[i]) for k, v in vars_dict.items()}
                c = Center(time_val, float(new_lats[i]), float(new_lons[i]), c_vars)
                t = tracks.add_track([c])
                tracks._head_ids.add(t.track_id)
                tracks._tail_ids.add(t.track_id)
            tracks.time_range.end = current_time
            return

        tail_tracks = tracks.tail
        tail_lats = np.array([t[-1].lat for t in tail_tracks])
        tail_lons = np.array([t[-1].lon for t in tail_tracks])

        dist_matrix = haversine_matrix(tail_lats, tail_lons, new_lats, new_lons)
        matched_indices = [-1] * num_centers

        while True:
            has_match = False
            for ic in range(num_centers):
                if matched_indices[ic] == -1 and np.any(
                    dist_matrix[:, ic] < self.threshold
                ):
                    it_match = int(np.argmin(dist_matrix[:, ic]))
                    if dist_matrix[it_match, ic] >= self.threshold:
                        continue

                    if np.argmin(dist_matrix[it_match, :]) == ic:
                        matched_indices[ic] = it_match
                        dist_matrix[:, ic] = np.inf
                        dist_matrix[it_match, :] = np.inf
                        has_match = True
            if not has_match:
                break

        new_tail_ids = set()
        for ic in range(num_centers):
            it_match = matched_indices[ic]
            c_vars = {k: float(v[ic]) for k, v in vars_dict.items()}
            c = Center(time_val, float(new_lats[ic]), float(new_lons[ic]), c_vars)

            if it_match != -1:
                t = tail_tracks[it_match]
                t.append(c)
                new_tail_ids.add(t.track_id)
            else:
                t = tracks.add_track([c])
                tracks._head_ids.add(t.track_id)
                new_tail_ids.add(t.track_id)

        tracks._tail_ids = new_tail_ids

        if tracks.time_range is None:
            tracks.time_range = TimeRange(start=current_time, end=current_time)
        else:
            if tracks.time_range.step is None:
                tracks.time_range.step = current_time - tracks.time_range.start
            tracks.time_range.end = current_time

    def extend_track(self, tracks1: Tracks, tracks2: Tracks) -> None:
        if len(tracks2) == 0:
            return

        if len(tracks1) == 0:
            tracks1.track_ids = tracks2.track_ids.copy()
            tracks1.times = tracks2.times.copy()
            tracks1.lats = tracks2.lats.copy()
            tracks1.lons = tracks2.lons.copy()
            tracks1.vars = {k: v.copy() for k, v in tracks2.vars.items()}
            tracks1._head_ids = tracks2._head_ids.copy()
            tracks1._tail_ids = tracks2._tail_ids.copy()
            tracks1.time_range = tracks2.time_range
            return

        t2_heads = tracks2.head
        if not t2_heads:
            return

        new_centers = [t[0] for t in t2_heads]
        new_lats = np.array([c.lat for c in new_centers])
        new_lons = np.array([c.lon for c in new_centers])

        t1_tails = tracks1.tail
        if not t1_tails:
            for tid in tracks2.unique_track_ids:
                tracks1.append(Track(tid, tracks2))
            tracks1._tail_ids = tracks2._tail_ids.copy()
            if tracks1.time_range and tracks2.time_range:
                tracks1.time_range.end = tracks2.time_range.end
            return

        tail_lats = np.array([t[-1].lat for t in t1_tails])
        tail_lons = np.array([t[-1].lon for t in t1_tails])

        dist_matrix = haversine_matrix(tail_lats, tail_lons, new_lats, new_lons)

        # Invalidate matches that don't align in time or have a temporal gap
        gap_exists = False
        t1_tr = tracks1.time_range
        t2_tr = tracks2.time_range
        if (
            t1_tr
            and t2_tr
            and t1_tr.step
            and t2_tr.start - t1_tr.step > t1_tr.end
        ):
            gap_exists = True

        expected_start_time = tracks2.time_range.start if tracks2.time_range else None

        for ic, c in enumerate(new_centers):
            if gap_exists or c.time != expected_start_time:
                dist_matrix[:, ic] = np.inf

        matched_indices = [-1] * len(new_centers)

        while True:
            has_match = False
            for ic in range(len(new_centers)):
                if matched_indices[ic] == -1 and np.any(
                    dist_matrix[:, ic] < self.threshold
                ):
                    it_match = int(np.argmin(dist_matrix[:, ic]))
                    if dist_matrix[it_match, ic] >= self.threshold:
                        continue
                    if np.argmin(dist_matrix[it_match, :]) == ic:
                        matched_indices[ic] = it_match
                        dist_matrix[:, ic] = np.inf
                        dist_matrix[it_match, :] = np.inf
                        has_match = True
            if not has_match:
                break

        matched_t1_tails = set()
        new_tails = set()
        t2_tail_ids = tracks2._tail_ids

        for ic in range(len(new_centers)):
            it_match = matched_indices[ic]
            t2_track = t2_heads[ic]

            if it_match != -1:
                t1_track = t1_tails[it_match]
                t1_track.extend(t2_track)
                matched_t1_tails.add(t1_track.track_id)
                if t2_track.track_id in t2_tail_ids:
                    new_tails.add(t1_track.track_id)
            else:
                tracks1.append(t2_track)
                tracks1._head_ids.add(t2_track.track_id)
                if t2_track.track_id in t2_tail_ids:
                    new_tails.add(t2_track.track_id)

        # Unmatched t1 tails remain tails
        for tid in tracks1._tail_ids:
            if tid not in matched_t1_tails:
                new_tails.add(tid)

        tracks1._tail_ids = new_tails

        if tracks1.time_range and tracks2.time_range:
            tracks1.time_range.end = tracks2.time_range.end
