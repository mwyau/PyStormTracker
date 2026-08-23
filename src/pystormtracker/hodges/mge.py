from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..models.geo import DEG_TO_RAD, geod_dist


@nb.njit(cache=True, nogil=True)
def geod_dev(
    p0_lat: float,
    p0_lon: float,
    p1_lat: float,
    p1_lon: float,
    p2_lat: float,
    p2_lon: float,
    w1: float,
    w2: float,
) -> float:
    """Spherical track-smoothness cost from the Hodges lineage.

    Measures track deviation over three consecutive points.  The adaptive
    track-smoothness and regional upper-bound constraint lineage is Hodges
    (1999); exact source-compatibility behavior is tracked separately in
    TRACK 1.5.4 ``src/geod_dev.c``:
    https://gitlab.act.reading.ac.uk/track/track/-/blob/TRACK-1.5.4/src/geod_dev.c

    Directional term is normalized by 0.5 to keep total cost in [0, 1].

    Reference: K. I. Hodges (1999), “Adaptive Constraints for Feature
    Tracking,” *Monthly Weather Review*, 127(6), 1362--1373.
    https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2

    Args:
        p0_lat, p0_lon, p1_lat, p1_lon, p2_lat, p2_lon: Triplets of lat/lon coordinates.
        w1, w2: Weights for direction and speed consistency.

    Returns:
        The calculated cost (smoothness penalty).
    """
    alpha1 = geod_dist(p0_lat, p0_lon, p1_lat, p1_lon)
    alpha2 = geod_dist(p1_lat, p1_lon, p2_lat, p2_lon)

    if alpha1 <= 0.0 and alpha2 <= 0.0:
        return 0.0
    if alpha1 <= 0.0 or alpha2 <= 0.0:
        return w2

    # Tangent vector calculation via vector products
    x0 = np.cos(p0_lat * DEG_TO_RAD) * np.cos(p0_lon * DEG_TO_RAD)
    y0 = np.cos(p0_lat * DEG_TO_RAD) * np.sin(p0_lon * DEG_TO_RAD)
    z0 = np.sin(p0_lat * DEG_TO_RAD)

    x1 = np.cos(p1_lat * DEG_TO_RAD) * np.cos(p1_lon * DEG_TO_RAD)
    y1 = np.cos(p1_lat * DEG_TO_RAD) * np.sin(p1_lon * DEG_TO_RAD)
    z1 = np.sin(p1_lat * DEG_TO_RAD)

    x2 = np.cos(p2_lat * DEG_TO_RAD) * np.cos(p2_lon * DEG_TO_RAD)
    y2 = np.cos(p2_lat * DEG_TO_RAD) * np.sin(p2_lon * DEG_TO_RAD)
    z2 = np.sin(p2_lat * DEG_TO_RAD)

    dot01 = x0 * x1 + y0 * y1 + z0 * z1
    dot21 = x2 * x1 + y2 * y1 + z2 * z1

    s1 = np.sin(alpha1)
    s2 = np.sin(alpha2)

    # Unit tangent vectors at p1
    t1x = (x0 - dot01 * x1) / s1
    t1y = (y0 - dot01 * y1) / s1
    t1z = (z0 - dot01 * z1) / s1

    t2x = (dot21 * x1 - x2) / s2
    t2y = (dot21 * y1 - y2) / s2
    t2z = (dot21 * z1 - z2) / s2

    dot_t = t1x * t2x + t1y * t2y + t1z * t2z

    # Combined cost: direction smoothness + speed consistency
    phi = 0.5 * w1 * (1.0 - dot_t) + w2 * (
        1.0 - 2.0 * np.sqrt(alpha1 * alpha2) / (alpha1 + alpha2)
    )

    if np.abs(phi) < 1.0e-8:
        return 0.0
    return float(phi)


@nb.njit(cache=True, nogil=True)
def _select_regional_dmax(
    lat: float, lon: float, zones: NDArray[np.float64], default_dmax: float
) -> float:
    """Select the regional search radius (dmax) for a point."""
    if zones.shape[0] == 0:
        return default_dmax

    zone_longitude = lon
    nonnegative_zone_longitudes = True
    for i in range(zones.shape[0]):
        if zones[i, 0] < 0.0 or zones[i, 1] < 0.0:
            nonnegative_zone_longitudes = False
            break
    if nonnegative_zone_longitudes and zone_longitude < 0.0:
        zone_longitude = np.mod(zone_longitude, 360.0)

    for i in range(zones.shape[0]):
        lon_min, lon_max, lat_min, lat_max, dmax = zones[i]

        if lat < lat_min or lat > lat_max:
            continue

        in_lon = False
        if lon_min > lon_max:  # Longitude wrap-around
            if zone_longitude >= lon_min or zone_longitude <= lon_max:
                in_lon = True
        else:
            if zone_longitude >= lon_min and zone_longitude <= lon_max:
                in_lon = True

        if in_lon:
            return float(dmax)

    return default_dmax


@nb.njit(cache=True, nogil=True)
def _compute_adaptive_phimax(
    mean_dist: float,
    adaptive_smoothness: NDArray[np.float64],
    default_phimax: float,
) -> float:
    """Calculate dynamic smoothness limit based on track speed."""
    if adaptive_smoothness.shape[1] < 4:
        return default_phimax

    adapt_thresholds = adaptive_smoothness[0, :]
    adapt_values = adaptive_smoothness[1, :]

    d = mean_dist

    if d < adapt_thresholds[0]:
        return float(adapt_values[0])

    if d >= adapt_thresholds[3]:
        return float(adapt_values[3])

    if d >= adapt_thresholds[0] and d < adapt_thresholds[1]:
        slope = (adapt_values[1] - adapt_values[0]) / (
            adapt_thresholds[1] - adapt_thresholds[0]
        )
        return float(adapt_values[0] + slope * (d - adapt_thresholds[0]))

    if d >= adapt_thresholds[1] and d < adapt_thresholds[2]:
        slope = (adapt_values[2] - adapt_values[1]) / (
            adapt_thresholds[2] - adapt_thresholds[1]
        )
        return float(adapt_values[1] + slope * (d - adapt_thresholds[1]))

    if d >= adapt_thresholds[2] and d < adapt_thresholds[3]:
        slope = (adapt_values[3] - adapt_values[2]) / (
            adapt_thresholds[3] - adapt_thresholds[2]
        )
        return float(adapt_values[2] + slope * (d - adapt_thresholds[2]))

    return default_phimax


@nb.njit(cache=True, nogil=True)
def _compute_track_cost(
    tracks: NDArray[np.int64],
    k: int,
    track_idx: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    w1: float,
    w2: float,
    phimax: float,
) -> float:
    """Calculate the cost for a track at step k using points k-1, k, k+1."""
    p0_idx = tracks[track_idx, k - 1]
    p1_idx = tracks[track_idx, k]
    p2_idx = tracks[track_idx, k + 1]

    # If first point is phantom, triplet has no cost
    if p0_idx == -1:
        return 0.0

    # If subsequent points are phantom, apply static penalty
    if p1_idx == -1 or p2_idx == -1:
        return phimax

    lat0 = features_lat[p0_idx]
    lon0 = features_lon[p0_idx]
    lat1 = features_lat[p1_idx]
    lon1 = features_lon[p1_idx]
    lat2 = features_lat[p2_idx]
    lon2 = features_lon[p2_idx]

    return float(geod_dev(lat0, lon0, lat1, lon1, lat2, lon2, w1, w2))


@nb.njit(cache=True, nogil=True)
def _mge_iteration(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    k: int,
    forward: bool,
    w1: float,
    w2: float,
    dmax_parameters: NDArray[np.float64],
    phimax_parameters: NDArray[np.float64],
    missing_input_counts: NDArray[np.int64],
    zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
) -> tuple[int, int]:
    """Single MGE iteration step at frame k finding the best candidate pair swap."""
    n_tracks = tracks.shape[0]
    best_gain = 1e-8
    best_i = -1
    best_j = -1

    # Target frame to swap
    target_k = k + 1 if forward else k - 1
    cost_parameter_index = max(missing_input_counts[k - 1], missing_input_counts[k])
    if cost_parameter_index >= phimax_parameters.size:
        cost_parameter_index = phimax_parameters.size - 1
    phimax = phimax_parameters[cost_parameter_index]
    displacement_parameter_index = missing_input_counts[k if forward else k - 1]
    if displacement_parameter_index >= dmax_parameters.size:
        displacement_parameter_index = dmax_parameters.size - 1
    default_dmax = dmax_parameters[displacement_parameter_index]

    # Cache current costs
    costs = np.zeros(n_tracks)
    for i in range(n_tracks):
        costs[i] = _compute_track_cost(
            tracks, k, i, features_lat, features_lon, w1, w2, phimax
        )

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            if (
                np.count_nonzero(tracks[i] != -1) <= 1
                and np.count_nonzero(tracks[j] != -1) <= 1
            ):
                continue

            p_i_orig = tracks[i, target_k]
            p_j_orig = tracks[j, target_k]

            if p_i_orig == p_j_orig:
                continue

            # 1. Displacement Check
            valid_swap = True
            idx_i_k = tracks[i, k]
            if idx_i_k != -1 and p_j_orig != -1:
                lat_k, lon_k = features_lat[idx_i_k], features_lon[idx_i_k]
                lat_t, lon_t = features_lat[p_j_orig], features_lon[p_j_orig]
                dmax_i = 0.5 * (
                    _select_regional_dmax(lat_k, lon_k, zones, default_dmax)
                    + _select_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                if geod_dist(lat_k, lon_k, lat_t, lon_t) > dmax_i * DEG_TO_RAD:
                    valid_swap = False

            idx_j_k = tracks[j, k]
            if valid_swap and idx_j_k != -1 and p_i_orig != -1:
                lat_k, lon_k = features_lat[idx_j_k], features_lon[idx_j_k]
                lat_t, lon_t = features_lat[p_i_orig], features_lon[p_i_orig]
                dmax_j = 0.5 * (
                    _select_regional_dmax(lat_k, lon_k, zones, default_dmax)
                    + _select_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                if geod_dist(lat_k, lon_k, lat_t, lon_t) > dmax_j * DEG_TO_RAD:
                    valid_swap = False

            if not valid_swap:
                continue

            # 2. Swap target frame points
            tracks[i, target_k] = p_j_orig
            tracks[j, target_k] = p_i_orig

            # 3. Cost Gain Calculation
            new_cost_i = _compute_track_cost(
                tracks, k, i, features_lat, features_lon, w1, w2, phimax
            )
            new_cost_j = _compute_track_cost(
                tracks, k, j, features_lat, features_lon, w1, w2, phimax
            )

            # 4. Dynamic Smoothness Check
            if tracks[i, k - 1] != -1 and tracks[i, k] != -1 and tracks[i, k + 1] != -1:
                d1 = geod_dist(
                    features_lat[tracks[i, k - 1]],
                    features_lon[tracks[i, k - 1]],
                    features_lat[tracks[i, k]],
                    features_lon[tracks[i, k]],
                )
                d2 = geod_dist(
                    features_lat[tracks[i, k]],
                    features_lon[tracks[i, k]],
                    features_lat[tracks[i, k + 1]],
                    features_lon[tracks[i, k + 1]],
                )
                phi_max_i = _compute_adaptive_phimax(
                    0.5 * (d1 + d2) / DEG_TO_RAD, adaptive_smoothness, phimax
                )
                if new_cost_i > phi_max_i:
                    valid_swap = False

            if (
                valid_swap
                and tracks[j, k - 1] != -1
                and tracks[j, k] != -1
                and tracks[j, k + 1] != -1
            ):
                d1 = geod_dist(
                    features_lat[tracks[j, k - 1]],
                    features_lon[tracks[j, k - 1]],
                    features_lat[tracks[j, k]],
                    features_lon[tracks[j, k]],
                )
                d2 = geod_dist(
                    features_lat[tracks[j, k]],
                    features_lon[tracks[j, k]],
                    features_lat[tracks[j, k + 1]],
                    features_lon[tracks[j, k + 1]],
                )
                phi_max_j = _compute_adaptive_phimax(
                    0.5 * (d1 + d2) / DEG_TO_RAD, adaptive_smoothness, phimax
                )
                if new_cost_j > phi_max_j:
                    valid_swap = False

            if valid_swap:
                gain = (costs[i] + costs[j]) - (new_cost_i + new_cost_j)
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_j = j

            # Revert swap for next pair check
            tracks[i, target_k] = p_i_orig
            tracks[j, target_k] = p_j_orig

    return best_i, best_j


@nb.njit(cache=True, nogil=True)
def _compute_endpoint_average_dmax(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    dmax_zones: NDArray[np.float64],
    default_dmax: float,
) -> float:
    """Calculate average regional dmax between two endpoints."""
    d1 = _select_regional_dmax(lat1, lon1, dmax_zones, default_dmax)
    d2 = _select_regional_dmax(lat2, lon2, dmax_zones, default_dmax)
    return 0.5 * (d1 + d2)


@nb.njit(cache=True, nogil=True)
def _has_excess_displacement(
    p1: int,
    p2: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    dmax_zones: NDArray[np.float64],
    default_dmax: float,
    use_regional_dmax: bool = True,
) -> bool:
    """Check if link between p1 and p2 exceeds maximum displacement."""
    if p1 == -1 or p2 == -1:
        return False
    lat1 = features_lat[p1]
    lon1 = features_lon[p1]
    lat2 = features_lat[p2]
    lon2 = features_lon[p2]
    if use_regional_dmax:
        dmax = _compute_endpoint_average_dmax(
            lat1, lon1, lat2, lon2, dmax_zones, default_dmax
        )
    else:
        dmax = default_dmax
    dist = geod_dist(lat1, lon1, lat2, lon2)
    return dist > dmax * DEG_TO_RAD


@nb.njit(cache=True, nogil=True)
def _find_first_compatible_empty_row(
    tracks: NDArray[np.int64],
    first_frame: int,
    last_frame: int,
) -> int:
    """Find first row in tracks empty across [first_frame, last_frame]."""
    n_rows = tracks.shape[0]
    for r in range(n_rows):
        is_empty = True
        for col in range(first_frame, last_frame + 1):
            if tracks[r, col] != -1:
                is_empty = False
                break
        if is_empty:
            return r
    return -1


@nb.njit(cache=True, nogil=True)
def _apply_track_fail(
    tracks: NDArray[np.int64],
    track_index: int,
    middle_frame: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    is_forward: bool,
    dmax_zones: NDArray[np.float64],
    default_dmax: float,
) -> None:
    """Apply TRACK track_fail in native nogil code to move failed track segments."""
    n_tracks, n_frames = tracks.shape
    if is_forward:
        first_frame = middle_frame + 2
        if first_frame >= n_frames:
            return
        p_mid_plus_1 = int(tracks[track_index, middle_frame + 1])
        p_first = int(tracks[track_index, first_frame])
        if not _has_excess_displacement(
            p_mid_plus_1, p_first, features_lat, features_lon, dmax_zones, default_dmax
        ):
            return
        last_frame = n_frames
        for frame_index in range(first_frame + 1, n_frames):
            if tracks[track_index, frame_index] == -1:
                last_frame = frame_index
                break
    else:
        last_frame = middle_frame - 1
        if last_frame <= 0:
            return
        p_last = int(tracks[track_index, last_frame])
        p_mid_minus_2 = int(tracks[track_index, middle_frame - 2])
        if not _has_excess_displacement(
            p_last, p_mid_minus_2, features_lat, features_lon, dmax_zones, default_dmax
        ):
            return
        first_frame = 0
        if last_frame - 1 != 0:
            for frame_index in range(last_frame - 1, -1, -1):
                if tracks[track_index, frame_index] == -1:
                    first_frame = frame_index + 1
                    break

    section_length = last_frame - first_frame
    if section_length <= 0:
        return

    compatible_first = max(first_frame - 1, 0)
    compatible_last = last_frame - 1 if last_frame == n_frames else last_frame
    destination = _find_first_compatible_empty_row(
        tracks, compatible_first, compatible_last
    )
    if destination == -1:
        for r in range(n_tracks):
            is_all_phantom = True
            for c in range(n_frames):
                if tracks[r, c] != -1:
                    is_all_phantom = False
                    break
            if is_all_phantom:
                destination = r
                break
    if destination == -1:
        return

    for c in range(first_frame, last_frame):
        tracks[destination, c] = tracks[track_index, c]
        tracks[track_index, c] = -1


@nb.njit(cache=True, nogil=True)
def _forward_mge_sweep(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    n_frames: int,
    missing_input_counts: NDArray[np.int64],
    w1: float,
    w2: float,
    dmax_parameters: NDArray[np.float64],
    phimax_parameters: NDArray[np.float64],
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
) -> bool:
    """Execute one forward MGE sweep."""
    changed = False
    for k in range(1, n_frames - 1):
        best_i, best_j = _mge_iteration(
            tracks,
            features_lat,
            features_lon,
            k,
            True,
            w1,
            w2,
            dmax_parameters,
            phimax_parameters,
            missing_input_counts,
            dmax_zones,
            adaptive_smoothness,
        )
        if best_i != -1:
            p_i = tracks[best_i, k + 1]
            p_j = tracks[best_j, k + 1]
            tracks[best_i, k + 1] = p_j
            tracks[best_j, k + 1] = p_i
            changed = True
            if k + 2 < n_frames:
                miss_cnt = missing_input_counts[k + 1]
                default_dmax = dmax_parameters[miss_cnt]
                _apply_track_fail(
                    tracks,
                    best_i,
                    k,
                    features_lat,
                    features_lon,
                    True,
                    dmax_zones,
                    default_dmax,
                )
                _apply_track_fail(
                    tracks,
                    best_j,
                    k,
                    features_lat,
                    features_lon,
                    True,
                    dmax_zones,
                    default_dmax,
                )
    return changed


@nb.njit(cache=True, nogil=True)
def _backward_mge_sweep(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    n_frames: int,
    missing_input_counts: NDArray[np.int64],
    w1: float,
    w2: float,
    dmax_parameters: NDArray[np.float64],
    phimax_parameters: NDArray[np.float64],
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
) -> bool:
    """Execute one backward MGE sweep."""
    changed = False
    for k in range(n_frames - 2, 0, -1):
        best_i, best_j = _mge_iteration(
            tracks,
            features_lat,
            features_lon,
            k,
            False,
            w1,
            w2,
            dmax_parameters,
            phimax_parameters,
            missing_input_counts,
            dmax_zones,
            adaptive_smoothness,
        )
        if best_i != -1:
            p_i = tracks[best_i, k - 1]
            p_j = tracks[best_j, k - 1]
            tracks[best_i, k - 1] = p_j
            tracks[best_j, k - 1] = p_i
            changed = True
            if k - 2 >= 0:
                miss_cnt = missing_input_counts[k - 2]
                default_dmax = dmax_parameters[miss_cnt]
                _apply_track_fail(
                    tracks,
                    best_i,
                    k,
                    features_lat,
                    features_lon,
                    False,
                    dmax_zones,
                    default_dmax,
                )
                _apply_track_fail(
                    tracks,
                    best_j,
                    k,
                    features_lat,
                    features_lon,
                    False,
                    dmax_zones,
                    default_dmax,
                )
    return changed


@nb.njit(cache=True, nogil=True)
def _run_directional_mge_loop(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    n_frames: int,
    missing_input_counts: NDArray[np.int64],
    w1: float,
    w2: float,
    dmax_parameters: NDArray[np.float64],
    phimax_parameters: NDArray[np.float64],
    dmax_zones: NDArray[np.float64],
    adaptive_smoothness: NDArray[np.float64],
    max_iterations: int,
) -> NDArray[np.int64]:
    """Execute complete directional MGE sweep rounds in native nogil Numba code."""
    if n_frames <= 3:
        return tracks

    forward_active = True
    backward_active = True

    for outer_iteration in range(max_iterations):
        if not (forward_active or backward_active):
            break

        if forward_active:
            forward_changed = False
            while True:
                swapped = _forward_mge_sweep(
                    tracks,
                    features_lat,
                    features_lon,
                    n_frames,
                    missing_input_counts,
                    w1,
                    w2,
                    dmax_parameters,
                    phimax_parameters,
                    dmax_zones,
                    adaptive_smoothness,
                )
                if not swapped:
                    break
                forward_changed = True

            forward_active = False
            if forward_changed:
                backward_active = True

        if outer_iteration == max_iterations - 1:
            break

        if backward_active:
            backward_changed = False
            while True:
                swapped = _backward_mge_sweep(
                    tracks,
                    features_lat,
                    features_lon,
                    n_frames,
                    missing_input_counts,
                    w1,
                    w2,
                    dmax_parameters,
                    phimax_parameters,
                    dmax_zones,
                    adaptive_smoothness,
                )
                if not swapped:
                    break
                backward_changed = True

            backward_active = False
            if backward_changed:
                forward_active = True

    return tracks
