from __future__ import annotations

import numpy as np

from ..models.geo import geod_dist_km
from ..models.tracks import Tracks


def match_tracks(
    tracks_ref: Tracks,
    tracks_comp: Tracks,
    max_dist_km: float = 440.0,
    min_overlap_fraction: float = 0.1,
) -> dict[int, int]:
    """
    Matches tracks from a comparison set to a reference set based on spatial
    proximity and temporal overlap, mirroring the algorithm in TRACK's trdist_eps.c.

    Args:
        tracks_ref: The reference Tracks object.
        tracks_comp: The comparison Tracks object.
        max_dist_km: Maximum mean geodetic distance (in km) allowed between tracks.
                     Default is 440.0 km (~4 degrees).
        min_overlap_fraction: Minimum overlap ratio required.
                              Ratio = (2.0 * overlap_len) / (len(A) + len(B)).
                              Default is 0.1 (10%).

    Returns:
        A dictionary mapping comparison track IDs to reference track IDs.
    """
    matches: dict[int, int] = {}

    # 1. Index reference tracks by time for faster initial lookup
    ref_by_time: dict[np.datetime64, list[int]] = {}
    for tid in tracks_ref.unique_track_ids:
        idx = np.where(tracks_ref.track_ids == tid)[0]
        times = tracks_ref.times[idx]
        for t in times:
            if t not in ref_by_time:
                ref_by_time[t] = []
            ref_by_time[t].append(tid)

    # 2. Iterate over comparison tracks
    for tid_comp in tracks_comp.unique_track_ids:
        idx_comp = np.where(tracks_comp.track_ids == tid_comp)[0]
        times_comp = tracks_comp.times[idx_comp]
        lats_comp = tracks_comp.lats[idx_comp]
        lons_comp = tracks_comp.lons[idx_comp]

        best_ref_id = -1
        min_mean_dist = float("inf")

        # Find potential candidates in ref that share any time step
        candidates = set()
        for t in times_comp:
            if t in ref_by_time:
                candidates.update(ref_by_time[t])

        for tid_ref in candidates:
            idx_ref = np.where(tracks_ref.track_ids == tid_ref)[0]
            times_ref = tracks_ref.times[idx_ref]

            # Calculate overlap length by finding concurrent time steps.
            mask_comp = np.isin(times_comp, times_ref)
            mask_ref = np.isin(times_ref, times_comp)

            overlap_len = np.sum(mask_comp)
            if overlap_len == 0:
                continue

            # Overlap Fraction check: Ensures the lifecycle matches reasonably well.
            # Mirroring TRACK's (2*ol)/(len1 + len2) formula.
            ratio = (2.0 * overlap_len) / (len(times_comp) + len(times_ref))
            if ratio < min_overlap_fraction:
                continue

            # Spatial Proximity check: Mean distance during the overlapping period.
            ol_lats_comp = lats_comp[mask_comp]
            ol_lons_comp = (lons_comp[mask_comp] + 180) % 360 - 180

            ol_lats_ref = tracks_ref.lats[idx_ref][mask_ref]
            ol_lons_ref = (tracks_ref.lons[idx_ref][mask_ref] + 180) % 360 - 180

            # Vectorized distance calculation across all overlapping points.
            dist_func = np.vectorize(geod_dist_km)
            dists = dist_func(ol_lats_comp, ol_lons_comp, ol_lats_ref, ol_lons_ref)
            mean_dist = float(np.mean(dists))

            # Select the best match (minimum mean separation distance).
            if mean_dist <= max_dist_km and mean_dist < min_mean_dist:
                min_mean_dist = mean_dist
                best_ref_id = tid_ref

        if best_ref_id != -1:
            matches[int(tid_comp)] = int(best_ref_id)

    return matches
