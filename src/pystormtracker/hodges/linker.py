from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..models.center import Center
from ..models.tracker import RawDetectionStep
from ..models.tracks import Tracks
from ..utils.geo import geod_dist
from .kernels import (
    _initial_break_pass,
    _mge_iteration,
    get_regional_dmax,
)


class HodgesLinker:
    """
    Implements the Modified Greedy Exchange (MGE) tracking algorithm.

    The MGE algorithm (Hodges 1999) optimizes trajectories by iteratively
    swapping feature points between tracks to minimize a global smoothness cost.
    It supports adaptive search radii and smoothness constraints.
    """

    def __init__(
        self,
        w1: float = 0.2,
        w2: float = 0.8,
        dmax: float = 5.0,
        phimax: float = 0.5,
        n_iterations: int = 10,
        max_missing: int = 0,
        zones: NDArray[np.float64] | None = None,
        adapt_thresholds: NDArray[np.float64] | None = None,
        adapt_values: NDArray[np.float64] | None = None,
    ) -> None:
        """
        Initialize the MGE linker.

        Args:
            w1, w2: Weights for the cost function.
            dmax: Default maximum displacement (degrees).
            phimax: Penalty for phantom points in the cost function.
            n_iterations: Maximum number of MGE passes.
            max_missing: Maximum consecutive phantom points allowed.
            zones: Regional dmax definitions.
            adapt_thresholds, adapt_values: Piecewise linear adaptive smoothness.
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.max_missing = max_missing

        # Ensure Numba-compatible contiguous arrays
        if zones is None:
            self.zones = np.zeros((0, 5), dtype=np.float64)
        else:
            self.zones = np.ascontiguousarray(zones, dtype=np.float64)

        if adapt_thresholds is None:
            self.adapt_thresholds = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_thresholds = np.ascontiguousarray(
                adapt_thresholds, dtype=np.float64
            )

        if adapt_values is None:
            self.adapt_values = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_values = np.ascontiguousarray(adapt_values, dtype=np.float64)

    def link(self, detections: list[RawDetectionStep]) -> Tracks:
        """
        Links raw detections into trajectories using MGE optimization.

        Args:
            detections: List of (time, lats, lons, values) for each frame.

        Returns:
            A Tracks object containing the optimized trajectories.
        """
        n_frames = len(detections)
        if n_frames < 2:
            return Tracks()

        # 1. Flatten features and store offsets for mapping to track matrix
        all_lats: list[float] = []
        all_lons: list[float] = []
        all_vals: list[float] = []
        step_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        varname = "intensity"
        for i, (_t, lats, lons, data) in enumerate(detections):
            all_lats.extend(lats)
            all_lons.extend(lons)
            if i == 0:
                varname = next(iter(data.keys()))
            all_vals.extend(data[varname])
            step_offsets[i + 1] = step_offsets[i] + len(lats)

        features_lat = np.array(all_lats, dtype=np.float64)
        features_lon = np.array(all_lons, dtype=np.float64)
        features_val = np.array(all_vals, dtype=np.float64)

        # 2. Initial Linking (Greedy Nearest Neighbor)
        # Seed tracks with points from the first frame
        n_init = step_offsets[1]
        track_matrix = np.full((n_init, n_frames), -1, dtype=np.int64)
        for i in range(n_init):
            track_matrix[i, 0] = i

        current_n_tracks = n_init
        deg_to_rad = np.pi / 180.0

        for k in range(n_frames - 1):
            features_kp1 = np.arange(step_offsets[k + 1], step_offsets[k + 2])
            used_kp1 = np.zeros(len(features_kp1), dtype=bool)

            # Match existing track tails to features in the next frame
            for t_idx in range(current_n_tracks):
                idx_k = track_matrix[t_idx, k]
                if idx_k == -1:
                    continue

                best_dist = 1e30
                best_feat = -1
                for f_idx, f_global in enumerate(features_kp1):
                    if used_kp1[f_idx]:
                        continue

                    # Determine effective dmax based on zones
                    dmax_eff = 0.5 * (
                        get_regional_dmax(
                            features_lat[idx_k],
                            features_lon[idx_k],
                            self.zones,
                            self.dmax,
                        )
                        + get_regional_dmax(
                            features_lat[f_global],
                            features_lon[f_global],
                            self.zones,
                            self.dmax,
                        )
                    )

                    dist = geod_dist(
                        features_lat[idx_k],
                        features_lon[idx_k],
                        features_lat[f_global],
                        features_lon[f_global],
                    )
                    if dist < dmax_eff * deg_to_rad and dist < best_dist:
                        best_dist, best_feat = dist, f_idx

                if best_feat != -1:
                    track_matrix[t_idx, k + 1] = features_kp1[best_feat]
                    used_kp1[best_feat] = True

            # Unlinked features start new tracks
            unlinked_indices = []
            for i in range(len(features_kp1)):
                if not used_kp1[i]:
                    unlinked_indices.append(features_kp1[i])

            if unlinked_indices:
                new_rows = np.full(
                    (len(unlinked_indices), n_frames), -1, dtype=np.int64
                )
                for i, f_global in enumerate(unlinked_indices):
                    new_rows[i, k + 1] = f_global
                track_matrix = np.vstack((track_matrix, new_rows))
                current_n_tracks += len(unlinked_indices)

        # 3. Initial Smoothness Breaking Pass
        # Breaks tracks that violate adaptive smoothness right after linking
        track_matrix = _initial_break_pass(
            track_matrix,
            features_lat,
            features_lon,
            self.w1,
            self.w2,
            self.phimax,
            self.adapt_thresholds,
            self.adapt_values,
        )

        # 4. MGE Optimization (Iterate until convergence)
        for _ in range(self.n_iterations):
            changed = False
            # Forward Pass: optimize from start to end
            for k in range(1, n_frames - 1):
                while True:
                    best_i, _ = _mge_iteration(
                        track_matrix,
                        features_lat,
                        features_lon,
                        k,
                        True,
                        self.w1,
                        self.w2,
                        self.dmax,
                        self.phimax,
                        self.zones,
                        self.adapt_thresholds,
                        self.adapt_values,
                        self.max_missing,
                    )
                    if best_i != -1:
                        changed = True
                    else:
                        break
            # Backward Pass: optimize from end to start
            for k in range(n_frames - 2, 0, -1):
                while True:
                    best_i, _ = _mge_iteration(
                        track_matrix,
                        features_lat,
                        features_lon,
                        k,
                        False,
                        self.w1,
                        self.w2,
                        self.dmax,
                        self.phimax,
                        self.zones,
                        self.adapt_thresholds,
                        self.adapt_values,
                        self.max_missing,
                    )
                    if best_i != -1:
                        changed = True
                    else:
                        break
            if not changed:
                break

        # 5. Convert track_matrix back to PyStormTracker's Tracks model
        tracks = Tracks()
        times = [d[0] for d in detections]
        for t_idx in range(track_matrix.shape[0]):
            centers: list[Center] = []
            consecutive_missing = 0
            for k in range(n_frames):
                f_idx = track_matrix[t_idx, k]
                if f_idx != -1:
                    # Enforce max_missing if tracks were merged during MGE
                    if consecutive_missing > self.max_missing and centers:
                        tracks.add_track(centers)
                        centers = []
                    centers.append(
                        Center(
                            time=times[k],
                            lat=features_lat[f_idx],
                            lon=features_lon[f_idx],
                            vars={varname: float(features_val[f_idx])},
                        )
                    )
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
            if centers:
                tracks.add_track(centers)
        return tracks
