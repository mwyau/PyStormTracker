from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.center import Center
from ..models.tracks import Centers, TimeRange, Tracks
from .detector import RawDetectionStep
from .kernels import geod_dev, geod_dist


@nb.njit(cache=True)
def _get_cost(
    tracks: NDArray[np.int64],
    k: int,
    track_idx: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    step_offsets: NDArray[np.int64],
    w1: float,
    w2: float,
    phimax: float,
) -> float:
    """Calculates the cost for a track at step k using points k-1, k, k+1."""
    p0_idx = tracks[track_idx, k - 1]
    p1_idx = tracks[track_idx, k]
    p2_idx = tracks[track_idx, k + 1]

    if p0_idx == -1:
        return 0.0
    
    if p1_idx == -1 or p2_idx == -1:
        return phimax

    # Map global feature indices to lat/lon
    # We assume features are concatenated: [step0_features, step1_features, ...]
    # step_offsets stores the starting index for each step
    
    lat0 = features_lat[p0_idx]
    lon0 = features_lon[p0_idx]
    lat1 = features_lat[p1_idx]
    lon1 = features_lon[p1_idx]
    lat2 = features_lat[p2_idx]
    lon2 = features_lon[p2_idx]

    return geod_dev(lat0, lon0, lat1, lon1, lat2, lon2, w1, w2)


@nb.njit(cache=True)
def _mge_iteration(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    step_offsets: NDArray[np.int64],
    k: int,
    forward: bool,
    w1: float,
    w2: float,
    dmax: float,
    phimax: float,
) -> bool:
    """A single MGE iteration step at frame k."""
    n_tracks = tracks.shape[0]
    changed = False
    
    # Target frame to swap
    target_k = k + 1 if forward else k - 1
    
    # Convert dmax from degrees to radians for geod_dist
    dmax_rad = dmax * (np.pi / 180.0)

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            # Current costs
            cost_i = _get_cost(tracks, k, i, features_lat, features_lon, step_offsets, w1, w2, phimax)
            cost_j = _get_cost(tracks, k, j, features_lat, features_lon, step_offsets, w1, w2, phimax)
            
            # Potential costs if we swap points at target_k
            # Swap
            p_i_orig = tracks[i, target_k]
            p_j_orig = tracks[j, target_k]
            
            if p_i_orig == p_j_orig: continue
            
            tracks[i, target_k] = p_j_orig
            tracks[j, target_k] = p_i_orig
            
            # Check displacement constraint for both tracks
            # Dist between k and target_k
            idx_i_k = tracks[i, k]
            idx_i_target = tracks[i, target_k]
            idx_j_k = tracks[j, k]
            idx_j_target = tracks[j, target_k]
            
            valid_swap = True
            if idx_i_k != -1 and idx_i_target != -1:
                dist_i = geod_dist(features_lat[idx_i_k], features_lon[idx_i_k], 
                                   features_lat[idx_i_target], features_lon[idx_i_target])
                if dist_i > dmax_rad: valid_swap = False
            
            if valid_swap and idx_j_k != -1 and idx_j_target != -1:
                dist_j = geod_dist(features_lat[idx_j_k], features_lon[idx_j_k], 
                                   features_lat[idx_j_target], features_lon[idx_j_target])
                if dist_j > dmax_rad: valid_swap = False
            
            if valid_swap:
                new_cost_i = _get_cost(tracks, k, i, features_lat, features_lon, step_offsets, w1, w2, phimax)
                new_cost_j = _get_cost(tracks, k, j, features_lat, features_lon, step_offsets, w1, w2, phimax)
                
                if (new_cost_i + new_cost_j) < (cost_i + cost_j) - 1e-8:
                    changed = True
                    # Keep swap
                else:
                    # Revert
                    tracks[i, target_k] = p_i_orig
                    tracks[j, target_k] = p_j_orig
            else:
                # Revert
                tracks[i, target_k] = p_i_orig
                tracks[j, target_k] = p_j_orig
                
    return changed


class HodgesLinker:
    """
    Implements the Modified Greedy Exchange (MGE) tracking algorithm.
    """

    def __init__(
        self,
        w1: float = 0.2,
        w2: float = 0.8,
        dmax: float = 5.0,
        phimax: float = 0.5,
        n_iterations: int = 3,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations

    def link(self, detections: list[RawDetectionStep]) -> Tracks:
        n_frames = len(detections)
        if n_frames < 2:
            return Tracks(tracks=[])

        # 1. Flatten features and store offsets
        all_lats = []
        all_lons = []
        all_vals = []
        step_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        
        for i, (t, lats, lons, data) in enumerate(detections):
            all_lats.extend(lats)
            all_lons.extend(lons)
            # Use the first variable provided in the dict
            varname = list(data.keys())[0]
            all_vals.extend(data[varname])
            step_offsets[i+1] = step_offsets[i] + len(lats)
            
        features_lat = np.array(all_lats, dtype=np.float64)
        features_lon = np.array(all_lons, dtype=np.float64)
        features_val = np.array(all_vals, dtype=np.float64)

        # 2. Initial Linking (Nearest Neighbor)
        # track_matrix[track_idx, frame_idx] = feature_index (-1 for phantom)
        # Start with one track per feature in frame 0
        n_init = step_offsets[1]
        track_matrix = np.full((n_init, n_frames), -1, dtype=np.int64)
        for i in range(n_init):
            track_matrix[i, 0] = i
            
        # Link subsequent frames
        dmax_rad = self.dmax * (np.pi / 180.0)
        current_n_tracks = n_init
        
        for k in range(n_frames - 1):
            # Try to link features in k+1 to existing tracks at k
            features_kp1 = np.arange(step_offsets[k+1], step_offsets[k+2])
            used_kp1 = np.zeros(len(features_kp1), dtype=bool)
            
            for t_idx in range(current_n_tracks):
                idx_k = track_matrix[t_idx, k]
                if idx_k == -1: continue
                
                best_dist = dmax_rad
                best_feat = -1
                
                for f_idx, f_global in enumerate(features_kp1):
                    if used_kp1[f_idx]: continue
                    dist = geod_dist(features_lat[idx_k], features_lon[idx_k],
                                     features_lat[f_global], features_lon[f_global])
                    if dist < best_dist:
                        best_dist = dist
                        best_feat = f_idx
                
                if best_feat != -1:
                    track_matrix[t_idx, k+1] = features_kp1[best_feat]
                    used_kp1[best_feat] = True
            
            # Features in k+1 not linked become new tracks
            unlinked = features_kp1[~used_kp1]
            if len(unlinked) > 0:
                new_rows = np.full((len(unlinked), n_frames), -1, dtype=np.int64)
                for i, f_global in enumerate(unlinked):
                    new_rows[i, k+1] = f_global
                track_matrix = np.vstack((track_matrix, new_rows))
                current_n_tracks += len(unlinked)

        # 3. MGE Optimization
        for _ in range(self.n_iterations):
            # Forward Pass
            for k in range(1, n_frames - 1):
                _mge_iteration(track_matrix, features_lat, features_lon, step_offsets, 
                               k, True, self.w1, self.w2, self.dmax, self.phimax)
            # Backward Pass
            for k in range(n_frames - 2, 0, -1):
                _mge_iteration(track_matrix, features_lat, features_lon, step_offsets, 
                               k, False, self.w1, self.w2, self.dmax, self.phimax)

        # 4. Convert track_matrix back to Tracks model
        tracks_list = []
        times = [d[0] for d in detections]
        for t_idx in range(track_matrix.shape[0]):
            centers = []
            for k in range(n_frames):
                f_idx = track_matrix[t_idx, k]
                if f_idx != -1:
                    centers.append(Center(
                        time=times[k],
                        lat=features_lat[f_idx],
                        lon=features_lon[f_idx],
                        intensity={varname: features_val[f_idx]}
                    ))
            if centers:
                tracks_list.append(Centers(centers=centers))
                
        return Tracks(tracks=tracks_list)
