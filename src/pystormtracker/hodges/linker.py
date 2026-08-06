from __future__ import annotations

from typing import Literal

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

Direction = Literal["forward", "backward"]


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
        max_iterations: int = constants.MAX_ITERATIONS_DEFAULT,
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
            max_iterations: Maximum number of outer forward/backward rounds.
                Must be positive. Reaching this limit is a normal termination
                path, not an error.
            max_missing_steps: Maximum consecutive phantom points allowed.
            dmax_zones: Regional dmax definitions.
            adaptive_smoothness: Piecewise linear adaptive smoothness parameters (2xN).
        """
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.max_iterations = max_iterations
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

        # 4. Directional MGE Optimization
        # Forward and backward passes alternate based on whether exchanges
        # occurred. This follows the directional control flow described by
        # Hodges 1999 and represented in TRACK 1.5.2.
        track_matrix = self._run_directional_mge(
            track_matrix,
            features_lat,
            features_lon,
            n_frames,
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

    def _run_directional_mge(
        self,
        track_matrix: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
    ) -> NDArray[np.int64]:
        """Run TRACK-style bounded forward/backward MGE optimization.

        Each active direction runs repeatedly until a complete sweep performs no
        exchange. Successful exchanges reactivate the opposite direction.

        ``max_iterations`` limits the number of outer directional rounds. The
        final permitted round is forward-only, matching ``mge_tracks.c`` where
        backward MGE runs only while ``tot_count < tot_term``.

        Reaching the outer iteration bound is a normal termination condition.
        """
        forward_active = True
        backward_active = True

        for outer_iteration in range(self.max_iterations):
            if not (forward_active or backward_active):
                break

            if forward_active:
                track_matrix, forward_changed = self._run_mge_direction_until_stable(
                    track_matrix,
                    features_lat,
                    features_lon,
                    n_frames,
                    direction="forward",
                )

                # The forward direction has converged for the current state.
                forward_active = False

                # TRACK's fel_mge() reactivates backward processing when at least
                # one forward exchange occurred.
                if forward_changed:
                    backward_active = True

            # TRACK skips bel_mge() during the final permitted outer round.
            if outer_iteration == self.max_iterations - 1:
                break

            if backward_active:
                track_matrix, backward_changed = self._run_mge_direction_until_stable(
                    track_matrix,
                    features_lat,
                    features_lon,
                    n_frames,
                    direction="backward",
                )

                # The backward direction has converged for the current state.
                backward_active = False

                # TRACK's bel_mge() reactivates forward processing when at least
                # one backward exchange occurred.
                if backward_changed:
                    forward_active = True

        return track_matrix

    def _run_mge_direction_until_stable(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
        *,
        direction: Direction,
    ) -> tuple[NDArray[np.int64], bool]:
        """Repeat one directional MGE sweep until it makes no exchange.

        TRACK implements this repetition recursively in ``fel_mge.c`` and
        ``bel_mge.c``. The Python implementation uses an iterative loop to avoid
        recursive stack growth.

        Returns:
            The updated track matrix and whether this directional stage performed
            at least one exchange.
        """
        changed_any = False

        while True:
            if direction == "forward":
                tracks, sweep_changed = self._run_forward_mge_iteration(
                    tracks,
                    features_lat,
                    features_lon,
                    n_frames,
                )
            else:
                tracks, sweep_changed = self._run_backward_mge_iteration(
                    tracks,
                    features_lat,
                    features_lon,
                    n_frames,
                )

            if not sweep_changed:
                return tracks, changed_any

            changed_any = True

    def _run_forward_mge_iteration(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
    ) -> tuple[NDArray[np.int64], bool]:
        """Run a single forward-direction MGE iteration.

        Iterates k from 1 to n_frames-2, attempting one best swap per frame
        in the forward direction (optimizing point k+1).

        Args:
            tracks: Track matrix [n_tracks, n_frames].
            features_lat: Flat array of all feature latitudes.
            features_lon: Flat array of all feature longitudes.
            n_frames: Number of time frames.

        Returns:
            Updated track matrix and whether any swap occurred.
        """
        changed = False

        for k in range(1, n_frames - 1):
            best_i, best_j = _mge_iteration(
                tracks,
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
                p_i = tracks[best_i, k + 1]
                p_j = tracks[best_j, k + 1]
                tracks[best_i, k + 1] = p_j
                tracks[best_j, k + 1] = p_i
                changed = True

                if k + 2 < n_frames:
                    tracks = _break_track(
                        tracks,
                        best_i,
                        k + 1,
                        features_lat,
                        features_lon,
                        self.dmax_zones,
                        self.dmax,
                        True,
                    )
                    tracks = _break_track(
                        tracks,
                        best_j,
                        k + 1,
                        features_lat,
                        features_lon,
                        self.dmax_zones,
                        self.dmax,
                        True,
                    )

        return tracks, changed

    def _run_backward_mge_iteration(
        self,
        tracks: NDArray[np.int64],
        features_lat: NDArray[np.float64],
        features_lon: NDArray[np.float64],
        n_frames: int,
    ) -> tuple[NDArray[np.int64], bool]:
        """Run a single backward-direction MGE iteration.

        Iterates k from n_frames-2 down to 1, attempting one best swap per
        frame in the backward direction (optimizing point k-1).

        Args:
            tracks: Track matrix [n_tracks, n_frames].
            features_lat: Flat array of all feature latitudes.
            features_lon: Flat array of all feature longitudes.
            n_frames: Number of time frames.

        Returns:
            Updated track matrix and whether any swap occurred.
        """
        changed = False

        for k in range(n_frames - 2, 0, -1):
            best_i, best_j = _mge_iteration(
                tracks,
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
                p_i = tracks[best_i, k - 1]
                p_j = tracks[best_j, k - 1]
                tracks[best_i, k - 1] = p_j
                tracks[best_j, k - 1] = p_i
                changed = True

                if k - 2 >= 0:
                    tracks = _break_track(
                        tracks,
                        best_i,
                        k - 1,
                        features_lat,
                        features_lon,
                        self.dmax_zones,
                        self.dmax,
                        False,
                    )
                    tracks = _break_track(
                        tracks,
                        best_j,
                        k - 1,
                        features_lat,
                        features_lon,
                        self.dmax_zones,
                        self.dmax,
                        False,
                    )

        return tracks, changed
