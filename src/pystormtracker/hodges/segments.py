from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..models.tracks import Tracks, TracksMetadata, _TracksBuilder

DEFAULT_SEGMENT_FRAMES: Final[int] = 62


@dataclass(frozen=True, slots=True)
class TrackingSegment:
    """
    A temporal work partition for tracking execution.

    Attributes
    ----------
    index : int
        Sequential segment identifier (0, 1, 2, ...).
    start : int
        0-based start index into source time dimension (inclusive).
    stop : int
        0-based stop index into source time dimension (exclusive).
    splice_index : int
        0-based local index in this segment where the overlap boundary frame lies
        (0 for the first segment or un-overlapped segments, 2 for 3-frame overlap).
    """

    index: int
    start: int
    stop: int
    splice_index: int = 0

    @property
    def length(self) -> int:
        """Number of time steps in this segment."""
        return self.stop - self.start


def plan_tracking_segments(
    total_steps: int,
    segment_frames: int | None,
    *,
    overlap: int = 0,
) -> list[TrackingSegment]:
    """
    Plan temporal work segments across the available time steps.

    Parameters
    ----------
    total_steps : int
        Total number of available time steps.
    segment_frames : int | None
        Target segment frame count. If None or >= total_steps, returns a single
        monolithic segment.
    overlap : int
        Number of steps overlapping with the preceding segment. For Hodges / TRACK,
        this is 2 (yielding a 3-frame overlap window: t-2, t-1, t).

    Returns
    -------
    list[TrackingSegment]
        Ordered sequence of work segments covering [0, total_steps).
    """
    if total_steps <= 0:
        return []

    if segment_frames is None or segment_frames >= total_steps:
        return [TrackingSegment(index=0, start=0, stop=total_steps, splice_index=0)]

    if segment_frames <= 0:
        raise ValueError(f"segment_frames must be positive, got {segment_frames}")
    if overlap < 0:
        raise ValueError(f"overlap must be nonnegative, got {overlap}")

    segments: list[TrackingSegment] = []
    segment_idx = 0

    if overlap == 0:
        start = 0
        while start < total_steps:
            stop = min(start + segment_frames, total_steps)
            segments.append(
                TrackingSegment(
                    index=segment_idx,
                    start=start,
                    stop=stop,
                    splice_index=0,
                )
            )
            segment_idx += 1
            start = stop
        return segments

    # Overlapped segmenting
    first_stop = min(segment_frames, total_steps)
    segments.append(
        TrackingSegment(index=segment_idx, start=0, stop=first_stop, splice_index=0)
    )
    segment_idx += 1
    prev_last_frame = first_stop - 1

    advance = max(1, segment_frames - 1)

    while prev_last_frame + 1 < total_steps:
        start = max(0, prev_last_frame - overlap)
        splice_index = prev_last_frame - start
        next_last_frame = min(prev_last_frame + advance, total_steps - 1)
        stop = next_last_frame + 1
        if stop <= start or prev_last_frame == next_last_frame:
            break
        segments.append(
            TrackingSegment(
                index=segment_idx,
                start=start,
                stop=stop,
                splice_index=splice_index,
            )
        )
        segment_idx += 1
        prev_last_frame = next_last_frame

    return segments


def _points_match(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    tolerance_degrees: float = 1.0e-5,
) -> bool:
    """Check if two geographic coordinates match within tolerance."""
    dlat = abs(lat1 - lat2)
    dlon = abs((lon1 - lon2 + 180.0) % 360.0 - 180.0)
    return bool(dlat <= tolerance_degrees and dlon <= tolerance_degrees)


class _ActiveTrack:
    __slots__ = (
        "eofs",
        "lats",
        "lons",
        "times",
        "variables",
    )

    def __init__(
        self,
        times: NDArray[np.int64],
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
        variables: dict[str, NDArray[np.float64]],
        eofs: bool,
    ) -> None:
        self.times = times
        self.lats = lats
        self.lons = lons
        self.variables = variables
        self.eofs = eofs


def merge_segments(
    segment_tracks: Sequence[Tracks],
    segment_plan: Sequence[TrackingSegment],
    *,
    tolerance_degrees: float = 1.0e-5,
) -> Tracks:
    """Merge ordered raw segment tracks across overlapping boundary frames.

    Parameters
    ----------
    segment_tracks : Sequence[Tracks]
        Ordered sequence of raw Tracks objects produced by per-segment MGE linking.
    segment_plan : Sequence[TrackingSegment]
        Ordered sequence of TrackingSegment definitions for `segment_tracks`.
    tolerance_degrees : float
        Maximum angular distance in degrees for matching track endpoints.

    Returns
    -------
    Tracks
        Single unified Tracks object containing complete merged trajectories.
    """
    if not segment_tracks:
        raise ValueError("segment_tracks must not be empty")

    metadata: TracksMetadata = segment_tracks[0].metadata

    if len(segment_tracks) == 1 or len(segment_plan) <= 1:
        # Single segment: return copy with sequential IDs
        seg0 = segment_tracks[0]
        if len(seg0) == 0:
            return Tracks.empty(metadata)
        builder = _TracksBuilder(metadata)
        for new_id, tr in enumerate(seg0, start=1):
            builder.new_track(new_id)
            builder.extend(new_id, tr.times, tr.lats, tr.lons, tr.variables)
        return builder.finish()

    accumulated: list[_ActiveTrack] = []

    # Initialize with Segment 0
    seg0 = segment_tracks[0]
    if len(seg0) > 0 and len(seg0.times) > 0:
        max_time_seg0 = int(np.max(seg0.times))
        for tr in seg0:
            if len(tr) == 0:
                continue
            reaches_end = bool(tr.times[-1] == max_time_seg0)
            accumulated.append(
                _ActiveTrack(
                    times=tr.times.copy(),
                    lats=tr.lats.copy(),
                    lons=tr.lons.copy(),
                    variables={k: v.copy() for k, v in tr.variables.items()},
                    eofs=reaches_end,
                )
            )

    # Iterate through subsequent segments
    for seg_idx in range(1, len(segment_tracks)):
        curr_segment_tracks = segment_tracks[seg_idx]
        curr_plan = segment_plan[seg_idx]

        if len(curr_segment_tracks) == 0 or len(curr_segment_tracks.times) == 0:
            for acc_tr in accumulated:
                acc_tr.eofs = False
            continue

        all_segment_times = np.unique(curr_segment_tracks.times)
        if len(all_segment_times) == 0:
            for acc_tr in accumulated:
                acc_tr.eofs = False
            continue

        splice_idx = curr_plan.splice_index
        if splice_idx < len(all_segment_times):
            t_boundary = int(all_segment_times[splice_idx])
        else:
            t_boundary = int(all_segment_times[0])

        last_time_curr_segment = int(all_segment_times[-1])

        active_candidates: list[tuple[int, _ActiveTrack]] = [
            (idx, acc_cand)
            for idx, acc_cand in enumerate(accumulated)
            if acc_cand.eofs
            and len(acc_cand.times) > 0
            and acc_cand.times[-1] == t_boundary
        ]

        matched_active_indices: set[int] = set()

        for cand_track in curr_segment_tracks:
            if len(cand_track) == 0:
                continue

            b_indices = np.where(cand_track.times == t_boundary)[0]

            matched = False
            if len(b_indices) > 0:
                b_idx = int(b_indices[0])
                cand_lat = float(cand_track.lats[b_idx])
                cand_lon = float(cand_track.lons[b_idx])

                for act_idx, act_tr in active_candidates:
                    if act_idx in matched_active_indices:
                        continue
                    act_lat = float(act_tr.lats[-1])
                    act_lon = float(act_tr.lons[-1])

                    if _points_match(
                        cand_lat,
                        cand_lon,
                        act_lat,
                        act_lon,
                        tolerance_degrees=tolerance_degrees,
                    ):
                        post_b_idx = b_idx + 1
                        if post_b_idx < len(cand_track):
                            act_tr.times = np.concatenate(
                                [act_tr.times, cand_track.times[post_b_idx:]]
                            )
                            act_tr.lats = np.concatenate(
                                [act_tr.lats, cand_track.lats[post_b_idx:]]
                            )
                            act_tr.lons = np.concatenate(
                                [act_tr.lons, cand_track.lons[post_b_idx:]]
                            )
                            for k in act_tr.variables:
                                if k in cand_track.variables:
                                    act_tr.variables[k] = np.concatenate(
                                        [
                                            act_tr.variables[k],
                                            cand_track.variables[k][post_b_idx:],
                                        ]
                                    )
                                else:
                                    act_tr.variables[k] = np.concatenate(
                                        [
                                            act_tr.variables[k],
                                            np.full(
                                                len(cand_track) - post_b_idx,
                                                np.nan,
                                            ),
                                        ]
                                    )
                        reaches_end = bool(
                            cand_track.times[-1] == last_time_curr_segment
                        )
                        act_tr.eofs = reaches_end
                        matched_active_indices.add(act_idx)
                        matched = True
                        break

            if not matched:
                valid_idx = np.where(cand_track.times >= t_boundary)[0]
                if len(valid_idx) > 0:
                    start_k = int(valid_idx[0])
                    reaches_end = bool(cand_track.times[-1] == last_time_curr_segment)
                    accumulated.append(
                        _ActiveTrack(
                            times=cand_track.times[start_k:].copy(),
                            lats=cand_track.lats[start_k:].copy(),
                            lons=cand_track.lons[start_k:].copy(),
                            variables={
                                k: v[start_k:].copy()
                                for k, v in cand_track.variables.items()
                            },
                            eofs=reaches_end,
                        )
                    )

        for act_idx, act_tr in active_candidates:
            if act_idx not in matched_active_indices:
                act_tr.eofs = False

    if not accumulated:
        return Tracks.empty(metadata)

    builder = _TracksBuilder(metadata)
    new_id = 1
    for final_tr in accumulated:
        if len(final_tr.times) > 0:
            builder.new_track(new_id)
            builder.extend(
                new_id,
                final_tr.times,
                final_tr.lats,
                final_tr.lons,
                final_tr.variables,
            )
            new_id += 1

    return builder.finish()
