from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.center import Center
from ..models.tracks import TimeRange, Tracks
from .detector import RawDetectionStep
from .kernels import (
    geod_dev,
    geod_dist,
    get_adaptive_phimax,
    get_regional_dmax,
)


@nb.njit(cache=True)
def _get_cost(
    tracks: NDArray[np.int64],
    k: int,
    track_idx: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
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
    k: int,
    forward: bool,
    w1: float,
    w2: float,
    default_dmax: float,
    phimax: float,
    zones: NDArray[np.float64],
    adapt_thresholds: NDArray[np.float64],
    adapt_values: NDArray[np.float64],
) -> bool:
    """A single MGE iteration step at frame k."""
    n_tracks = tracks.shape[0]
    changed = False
    
    # Target frame to swap
    target_k = k + 1 if forward else k - 1
    
    rad_to_deg = 180.0 / np.pi
    deg_to_rad = np.pi / 180.0

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            # 1. Check displacement constraints first
            idx_i_k = tracks[i, k]
            idx_i_target = tracks[i, target_k]
            idx_j_k = tracks[j, k]
            idx_j_target = tracks[j, target_k]
            
            # Points currently at target_k
            p_i_orig = tracks[i, target_k]
            p_j_orig = tracks[j, target_k]
            
            if p_i_orig == p_j_orig: continue
            
            # Potential points after swap
            # Check displacement for i: dist(k, target_k)
            valid_swap = True
            if idx_i_k != -1 and p_j_orig != -1:
                lat_k = features_lat[idx_i_k]
                lon_k = features_lon[idx_i_k]
                lat_t = features_lat[p_j_orig]
                lon_t = features_lon[p_j_orig]
                
                dmax_i = 0.5 * (
                    get_regional_dmax(lat_k, lon_k, zones, default_dmax) +
                    get_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                dist_i = geod_dist(lat_k, lon_k, lat_t, lon_t)
                if dist_i > dmax_i * deg_to_rad:
                    valid_swap = False
            
            if valid_swap and idx_j_k != -1 and p_i_orig != -1:
                lat_k = features_lat[idx_j_k]
                lon_k = features_lon[idx_j_k]
                lat_t = features_lat[p_i_orig]
                lon_t = features_lon[p_i_orig]
                
                dmax_j = 0.5 * (
                    get_regional_dmax(lat_k, lon_k, zones, default_dmax) +
                    get_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                dist_j = geod_dist(lat_k, lon_k, lat_t, lon_t)
                if dist_j > dmax_j * deg_to_rad:
                    valid_swap = False
                    
            if not valid_swap: continue

            # 2. Calculate current costs
            cost_i = _get_cost(tracks, k, i, features_lat, features_lon, w1, w2, phimax)
            cost_j = _get_cost(tracks, k, j, features_lat, features_lon, w1, w2, phimax)
            
            # 3. Calculate potential costs after swap
            tracks[i, target_k] = p_j_orig
            tracks[j, target_k] = p_i_orig
            
            new_cost_i = _get_cost(tracks, k, i, features_lat, features_lon, w1, w2, phimax)
            new_cost_j = _get_cost(tracks, k, j, features_lat, features_lon, w1, w2, phimax)
            
            # 4. Check dynamic smoothness constraints for triplets of real points
            # For i
            if tracks[i, k-1] != -1 and tracks[i, k] != -1 and tracks[i, k+1] != -1:
                d1 = geod_dist(features_lat[tracks[i, k-1]], features_lon[tracks[i, k-1]],
                               features_lat[tracks[i, k]], features_lon[tracks[i, k]])
                d2 = geod_dist(features_lat[tracks[i, k]], features_lon[tracks[i, k]],
                               features_lat[tracks[i, k+1]], features_lon[tracks[i, k+1]])
                mean_d_deg = 0.5 * (d1 + d2) * rad_to_deg
                phi_max_i = get_adaptive_phimax(mean_d_deg, adapt_thresholds, adapt_values, phimax)
                if new_cost_i > phi_max_i:
                    valid_swap = False
            
            # For j
            if valid_swap and tracks[j, k-1] != -1 and tracks[j, k] != -1 and tracks[j, k+1] != -1:
                d1 = geod_dist(features_lat[tracks[j, k-1]], features_lon[tracks[j, k-1]],
                               features_lat[tracks[j, k]], features_lon[tracks[j, k]])
                d2 = geod_dist(features_lat[tracks[j, k]], features_lon[tracks[j, k]],
                               features_lat[tracks[j, k+1]], features_lon[tracks[j, k+1]])
                mean_d_deg = 0.5 * (d1 + d2) * rad_to_deg
                phi_max_j = get_adaptive_phimax(mean_d_deg, adapt_thresholds, adapt_values, phimax)
                if new_cost_j > phi_max_j:
                    valid_swap = False

            if valid_swap and (new_cost_i + new_cost_j) < (cost_i + cost_j) - 1e-8:
                changed = True
                # Keep swap
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
        max_missing: int = 0,
        zones: NDArray[np.float64] | None = None,
        adapt_thresholds: NDArray[np.float64] | None = None,
        adapt_values: NDArray[np.float64] | None = None,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.dmax = dmax
        self.phimax = phimax
        self.n_iterations = n_iterations
        self.max_missing = max_missing
        
        # Ensure Numba-compatible arrays
        if zones is None:
            self.zones = np.zeros((0, 5), dtype=np.float64)
        else:
            self.zones = np.ascontiguousarray(zones, dtype=np.float64)
            
        if adapt_thresholds is None:
            self.adapt_thresholds = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_thresholds = np.ascontiguousarray(adapt_thresholds, dtype=np.float64)
            
        if adapt_values is None:
            self.adapt_values = np.zeros(0, dtype=np.float64)
        else:
            self.adapt_values = np.ascontiguousarray(adapt_values, dtype=np.float64)

    def link(self, detections: list[RawDetectionStep]) -> Tracks:
        n_frames = len(detections)
        if n_frames < 2:
            return Tracks()

        # 1. Flatten features and store offsets
        all_lats = []
        all_lons = []
        all_vals = []
        step_offsets = np.zeros(n_frames + 1, dtype=np.int64)
        
        varname = "intensity"
        for i, (t, lats, lons, data) in enumerate(detections):
            all_lats.extend(lats)
            all_lons.extend(lons)
            if i == 0:
                varname = list(data.keys())[0]
            all_vals.extend(data[varname])
            step_offsets[i+1] = step_offsets[i] + len(lats)
            
        features_lat = np.array(all_lats, dtype=np.float64)
        features_lon = np.array(all_lons, dtype=np.float64)
        features_val = np.array(all_vals, dtype=np.float64)

        # 2. Initial Linking (Nearest Neighbor)
        n_init = step_offsets[1]
        track_matrix = np.full((n_init, n_frames), -1, dtype=np.int64)
        for i in range(n_init):
            track_matrix[i, 0] = i
            
        current_n_tracks = n_init
        deg_to_rad = np.pi / 180.0
        
        for k in range(n_frames - 1):
            features_kp1 = np.arange(step_offsets[k+1], step_offsets[k+2])
            used_kp1 = np.zeros(len(features_kp1), dtype=bool)
            
            for t_idx in range(current_n_tracks):
                idx_k = track_matrix[t_idx, k]
                if idx_k == -1: continue
                
                lat_k = features_lat[idx_k]
                lon_k = features_lon[idx_k]
                
                # Check ALL available features in k+1 for best match
                best_dist = 1e30
                best_feat = -1
                
                for f_idx, f_global in enumerate(features_kp1):
                    if used_kp1[f_idx]: continue
                    
                    lat_kp1 = features_lat[f_global]
                    lon_kp1 = features_lon[f_global]
                    
                    # Regional dmax
                    dmax_eff = 0.5 * (
                        get_regional_dmax(lat_k, lon_k, self.zones, self.dmax) +
                        get_regional_dmax(lat_kp1, lon_kp1, self.zones, self.dmax)
                    )
                    
                    dist = geod_dist(lat_k, lon_k, lat_kp1, lon_kp1)
                    if dist < dmax_eff * deg_to_rad and dist < best_dist:
                        best_dist = dist
                        best_feat = f_idx
                
                if best_feat != -1:
                    track_matrix[t_idx, k+1] = features_kp1[best_feat]
                    used_kp1[best_feat] = True
            
            # Features in k+1 not linked become new tracks
            unlinked_indices = []
            for i in range(len(features_kp1)):
                if not used_kp1[i]:
                    unlinked_indices.append(features_kp1[i])
            
            if unlinked_indices:
                new_rows = np.full((len(unlinked_indices), n_frames), -1, dtype=np.int64)
                for i, f_global in enumerate(unlinked_indices):
                    new_rows[i, k+1] = f_global
                track_matrix = np.vstack((track_matrix, new_rows))
                current_n_tracks += len(unlinked_indices)

        # 3. MGE Optimization
        for _ in range(self.n_iterations):
            # Forward Pass
            for k in range(1, n_frames - 1):
                _mge_iteration(
                    track_matrix, features_lat, features_lon, k, True,
                    self.w1, self.w2, self.dmax, self.phimax,
                    self.zones, self.adapt_thresholds, self.adapt_values
                )
            # Backward Pass
            for k in range(n_frames - 2, 0, -1):
                _mge_iteration(
                    track_matrix, features_lat, features_lon, k, False,
                    self.w1, self.w2, self.dmax, self.phimax,
                    self.zones, self.adapt_thresholds, self.adapt_values
                )

        # 4. Convert track_matrix back to Tracks model
        tracks = Tracks()
        times = [d[0] for d in detections]
        for t_idx in range(track_matrix.shape[0]):
            centers = []
            consecutive_missing = 0
            
            for k in range(n_frames):
                f_idx = track_matrix[t_idx, k]
                if f_idx != -1:
                    # If we have a gap larger than max_missing, split the track
                    if consecutive_missing > self.max_missing and centers:
                        tracks.add_track(centers)
                        centers = []
                    
                    centers.append(Center(
                        time=times[k],
                        lat=features_lat[f_idx],
                        lon=features_lon[f_idx],
                        vars={varname: float(features_val[f_idx])}
                    ))
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
                    
            if centers:
                tracks.add_track(centers)
                
        return tracks
