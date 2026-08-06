from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..models.constants import DEGTORAD
from ..models.geo import SpatialBounds, geod_dist
from ..models.time import encode_time_values
from ..models.tracker import RawDetectionStep
from ..models.tracks import ProcessingStep, Tracks, TracksMetadata, _TracksBuilder
from ..models.units import ResolvedDetectionMode, canonical_unit_for
from . import constants
from .kernels import (
    _break_track,
    _initial_break_pass,
    _mge_iteration,
    get_regional_dmax,
)

_MGE_SAFETY_LIMIT = 10


class HodgesLinker:
    """
    Implements the Modified Greedy Exchange (MGE) tracking algorithm.

    The MGE algorithm (Hodges 1999) optimizes trajectories by iteratively
    swapping feature points between tracks to minimize a global smoothness cost.
    It supports adaptive search radii and smoothness constraints.
    """

    def __init__(
        self,
        w1: float = constants.W1_DEFAULT,
        w2: float = constants.W2_DEFAULT,
        dmax: float = constants.DMAX_DEFAULT,
        phimax: float = constants.PHIMAX_DEFAULT,
        max_missing_steps: int = constants.MISSING_DEFAULT,
        dmax_zones: NDArray[np.float64] = constants.TRACK_ZONES,
        adaptive_smoothness: NDArray[np.float64] = constants.ADAPT_PARAMS,
    ) -> None:
        """
        Initialize the MGE linker.

        Args:
            w1, w2: Weights for the cost function.
            dmax: Default maximum displacement (degrees).
            phimax: Penalty for phantom points in the cost function.
            max_missing_steps: Maximum consecutive phantom points allowed.
            dmax_zones: Regional dmax definitions.
            adaptive_smoothness: Piecewise linear adaptive smoothness parameters (2xN).
        """
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.max_missing_steps = max_missing_steps
        self.dmax_zones = dmax_zones
        self.adaptive_smoothness = adaptive_smoothness

    def link(
        self,
        detections: list[RawDetectionStep],
        *,
        primary_var: str,
        mode: ResolvedDetectionMode,
        bounds: SpatialBounds | None = None,
        unit: str | None = None,
        processing: tuple[ProcessingStep, ...] = (),
    ) -> Tracks:
        """
        Links raw detections into trajectories using MGE optimization.

        Args:
            detections: List of (time, lats, lons, values) for each frame.

        Returns:
            A Tracks object containing the optimized trajectories.
        """
        n_frames = len(detections)
        if n_frames == 0:
            return Tracks.empty(
                TracksMetadata(
                    primary_var,
                    mode,
                    {primary_var: unit or canonical_unit_for(primary_var) or "1"},
                    bounds,
                    processing,
                )
            )

        # 1. Flatten features and store offsets for mapping to track matrix
        all_lats: list[float] = []
        all_lons: list[float] = []
        all_vals: list[float] = []
        step_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        for i, (_t, lats, lons, values) in enumerate(detections):
            all_lats.extend(lats)
            all_lons.extend(lons)
            all_vals.extend(values)
            step_offsets[i + 1] = step_offsets[i] + len(lats)

        features_lat = np.array(all_lats, dtype=np.float64)
        features_lon = np.array(all_lons, dtype=np.float64)
        features_val = np.array(all_vals, dtype=np.float64)
        n_features = len(features_lat)

        if n_features == 0:
            return Tracks.empty(
                TracksMetadata(
                    primary_var,
                    mode,
                    {primary_var: unit or canonical_unit_for(primary_var) or "1"},
                    bounds,
                    processing,
                )
            )

        # 2. Nearest-Neighbor Initialization Pass (Greedy Exchange setup)
        first_frame_count = step_offsets[1]
        track_matrix = np.full((first_frame_count, n_frames), -1, dtype=np.int64)
        for f_idx in range(first_frame_count):
            track_matrix[f_idx, 0] = f_idx

        current_n_tracks = first_frame_count

        for k in range(n_frames - 1):
            f_start_kp1 = step_offsets[k + 1]
            f_end_kp1 = step_offsets[k + 2]

            features_kp1 = np.arange(f_start_kp1, f_end_kp1, dtype=np.int64)
            used_kp1 = np.zeros(len(features_kp1), dtype=bool)

            for t_idx in range(current_n_tracks):
                idx_k = track_matrix[t_idx, k]
                if idx_k == -1:
                    continue

                dmax_eff = get_regional_dmax(
                    features_lat[idx_k],
                    features_lon[idx_k],
                    self.dmax_zones,
                    self.dmax,
                )

                best_dist = float("inf")
                best_feat = -1

                for f_idx, f_global in enumerate(features_kp1):
                    if used_kp1[f_idx]:
                        continue
                    dist = geod_dist(
                        features_lat[idx_k],
                        features_lon[idx_k],
                        features_lat[f_global],
                        features_lon[f_global],
                    )
                    if dist < dmax_eff * DEGTORAD and dist < best_dist:
                        best_dist, best_feat = dist, f_idx

                if best_feat != -1:
                    track_matrix[t_idx, k + 1] = features_kp1[best_feat]
                    used_kp1[best_feat] = True

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
        track_matrix = _initial_break_pass(
            track_matrix,
            features_lat,
            features_lon,
            self.w1,
            self.w2,
            self.phimax,
            self.adaptive_smoothness,
        )

        # 4. MGE Optimization (Iterate until convergence)
        for _pass in range(_MGE_SAFETY_LIMIT):
            changed = False
            # Forward Pass: one best swap per frame
            for k in range(1, n_frames - 1):
                best_i, best_j = _mge_iteration(
                    track_matrix,
                    features_lat,
                    features_lon,
                    k,
                    True,
                    self.w1,
                    self.w2,
                    self.dmax,
                    self.phimax,
                    self.dmax_zones,
                    self.adaptive_smoothness,
                    self.max_missing_steps,
                )
                if best_i != -1:
                    p_i = track_matrix[best_i, k + 1]
                    p_j = track_matrix[best_j, k + 1]
                    track_matrix[best_i, k + 1] = p_j
                    track_matrix[best_j, k + 1] = p_i
                    changed = True

                    if k + 2 < n_frames:
                        track_matrix = _break_track(
                            track_matrix,
                            best_i,
                            k + 1,
                            features_lat,
                            features_lon,
                            self.dmax_zones,
                            self.dmax,
                            True,
                        )
                        track_matrix = _break_track(
                            track_matrix,
                            best_j,
                            k + 1,
                            features_lat,
                            features_lon,
                            self.dmax_zones,
                            self.dmax,
                            True,
                        )

            # Backward Pass
            for k in range(n_frames - 2, 0, -1):
                best_i, best_j = _mge_iteration(
                    track_matrix,
                    features_lat,
                    features_lon,
                    k,
                    False,
                    self.w1,
                    self.w2,
                    self.dmax,
                    self.phimax,
                    self.dmax_zones,
                    self.adaptive_smoothness,
                    self.max_missing_steps,
                )
                if best_i != -1:
                    p_i = track_matrix[best_i, k - 1]
                    p_j = track_matrix[best_j, k - 1]
                    track_matrix[best_i, k - 1] = p_j
                    track_matrix[best_j, k - 1] = p_i
                    changed = True

                    if k - 2 >= 0:
                        track_matrix = _break_track(
                            track_matrix,
                            best_i,
                            k - 1,
                            features_lat,
                            features_lon,
                            self.dmax_zones,
                            self.dmax,
                            False,
                        )
                        track_matrix = _break_track(
                            track_matrix,
                            best_j,
                            k - 1,
                            features_lat,
                            features_lon,
                            self.dmax_zones,
                            self.dmax,
                            False,
                        )

            if not changed:
                break
        else:
            raise RuntimeError(
                f"MGE optimization failed to converge within "
                f"{_MGE_SAFETY_LIMIT} passes."
            )

        # 5. Build final Tracks output
        units = {primary_var: unit or canonical_unit_for(primary_var) or "1"}
        builder = _TracksBuilder(
            TracksMetadata(primary_var, mode, units, bounds, processing)
        )

        times_packed = encode_time_values([step[0] for step in detections])

        for row in range(track_matrix.shape[0]):
            feature_indices = track_matrix[row, :]
            valid_mask = feature_indices != -1

            if not np.any(valid_mask):
                continue

            track_times = times_packed[valid_mask]
            valid_feats = feature_indices[valid_mask]

            track_lats = features_lat[valid_feats]
            track_lons = features_lon[valid_feats]
            track_vals = features_val[valid_feats]

            builder.add_track(
                row + 1,
                track_times,
                track_lats,
                track_lons,
                {primary_var: track_vals},
            )

        return builder.finish()
