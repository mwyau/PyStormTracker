from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ..utils.geo import DEGTORAD, geod_dist


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _numba_get_centers(
    extrema: NDArray[np.float64],
    frame: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    Extracts the grid indices and values of detected extrema.

    Args:
        extrema: Binary mask from extrema detection.
        frame: The original data frame.

    Returns:
        A tuple of (row_indices, col_indices, values).
    """
    idx = np.where(extrema > 0)
    r_idx = idx[0]
    c_idx = idx[1]
    vals = np.zeros(len(r_idx), dtype=np.float64)
    for i in range(len(r_idx)):
        vals[i] = frame[r_idx[i], c_idx[i]]
    return r_idx, c_idx, vals


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def subgrid_refine(
    frame: NDArray[np.float64],
    r: int,
    c: int,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Refines an extremum position using local quadratic interpolation.

    Fits f(y, x) = a*y^2 + b*x^2 + c*y*x + d*y + e*x + f to a 3x3 neighborhood.
    The refined center is where partial derivatives are zero.
    This provides sub-grid precision without the need for global B-splines
    (which would require Cholesky pre-processing as seen in original TRACK).

    Args:
        frame: 2D data frame.
        r, c: Row and column index of the grid-level extremum.
        lat, lon: Coordinate arrays.

    Returns:
        (refined_lat, refined_lon, refined_intensity).
    """
    ny, nx = frame.shape

    # Boundary check: need 3x3 neighborhood
    if r < 1 or r >= ny - 1:
        return lat[r], lon[c], frame[r, c]

    # Extract 3x3 neighborhood with longitude wrapping
    cm = (c - 1) % nx
    cp = (c + 1) % nx

    z = np.zeros((3, 3))
    z[0, 0] = frame[r - 1, cm]
    z[0, 1] = frame[r - 1, c]
    z[0, 2] = frame[r - 1, cp]
    z[1, 0] = frame[r, cm]
    z[1, 1] = frame[r, c]
    z[1, 2] = frame[r, cp]
    z[2, 0] = frame[r + 1, cm]
    z[2, 1] = frame[r + 1, c]
    z[2, 2] = frame[r + 1, cp]

    # Use finite differences to find quadratic surface coefficients
    f_yy = z[0, 1] - 2 * z[1, 1] + z[2, 1]
    f_xx = z[1, 0] - 2 * z[1, 1] + z[1, 2]
    f_yx = 0.25 * (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0])
    f_y = 0.5 * (z[2, 1] - z[0, 1])
    f_x = 0.5 * (z[1, 2] - z[1, 0])

    det = f_yy * f_xx - f_yx**2
    if abs(det) < 1e-10:
        return lat[r], lon[c], frame[r, c]

    # Offset from grid center
    dy = (f_yx * f_x - f_xx * f_y) / det
    dx = (f_yx * f_y - f_yy * f_x) / det

    # Validation: refined point must remain within the grid cell
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return lat[r], lon[c], frame[r, c]

    # Precision interpolation using local grid intervals
    if dy > 0:
        ref_lat = lat[r] + dy * (lat[r + 1] - lat[r])
    else:
        ref_lat = lat[r] + abs(dy) * (lat[r - 1] - lat[r])

    if dx > 0:
        lon_next = lon[cp] if cp > c else lon[cp] + 360.0
        ref_lon = lon[c] + dx * (lon_next - lon[c])
    else:
        lon_prev = lon[cm] if cm < c else lon[cm] - 360.0
        ref_lon = lon[c] + abs(dx) * (lon_prev - lon[c])

    if ref_lon >= 360.0:
        ref_lon -= 360.0
    if ref_lon < 0.0:
        ref_lon += 360.0

    ref_val = z[1, 1] + 0.5 * (f_y * dy + f_x * dx)

    return ref_lat, ref_lon, ref_val


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
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
    """
    Spherical cost function (Hodges 1995, 1999).

    Measures track deviation over three consecutive points.
    Directional term is normalized by 0.5 to keep total cost in [0, 1].

    Args:
        p0, p1, p2: Triplets of lat/lon coordinates.
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
    x0 = np.cos(p0_lat * DEGTORAD) * np.cos(p0_lon * DEGTORAD)
    y0 = np.cos(p0_lat * DEGTORAD) * np.sin(p0_lon * DEGTORAD)
    z0 = np.sin(p0_lat * DEGTORAD)

    x1 = np.cos(p1_lat * DEGTORAD) * np.cos(p1_lon * DEGTORAD)
    y1 = np.cos(p1_lat * DEGTORAD) * np.sin(p1_lon * DEGTORAD)
    z1 = np.sin(p1_lat * DEGTORAD)

    x2 = np.cos(p2_lat * DEGTORAD) * np.cos(p2_lon * DEGTORAD)
    y2 = np.cos(p2_lat * DEGTORAD) * np.sin(p2_lon * DEGTORAD)
    z2 = np.sin(p2_lat * DEGTORAD)

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

    return float(max(0.0, phi))


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def get_regional_dmax(
    lat: float, lon: float, zones: NDArray[np.float64], default_dmax: float
) -> float:
    """
    Looks up the regional search radius (dmax) for a point.

    Args:
        lat, lon: Coordinates of the point.
        zones: Array of zones [lon_min, lon_max, lat_min, lat_max, dmax].
        default_dmax: Value to return if no zone is matched.

    Returns:
        The applicable dmax value.
    """
    if zones.shape[0] == 0:
        return default_dmax

    for i in range(zones.shape[0]):
        lon_min, lon_max, lat_min, lat_max, dmax = zones[i]

        if lat < lat_min or lat > lat_max:
            continue

        in_lon = False
        if lon_min > lon_max:  # Longitude wrap-around
            if lon >= lon_min or lon <= lon_max:
                in_lon = True
        else:
            if lon >= lon_min and lon <= lon_max:
                in_lon = True

        if in_lon:
            return float(dmax)

    return default_dmax


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def get_adaptive_phimax(
    mean_dist: float,
    adapt_params: NDArray[np.float64],
    default_phimax: float,
) -> float:
    """
    Calculates dynamic smoothness limit based on track speed.

    Args:
        mean_dist: Average displacement over three frames.
        adapt_params: Distance thresholds and smoothness values (2x4 array).
        default_phimax: Value if adaptive logic is disabled.

    Returns:
        The dynamic phimax limit.
    """
    if adapt_params.shape[1] < 4:
        return default_phimax

    adapt_thresholds = adapt_params[0, :]
    adapt_values = adapt_params[1, :]

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


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
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
    """
    Calculates the cost for a track at step k using points k-1, k, k+1.

    Args:
        tracks: Track matrix [n_tracks, n_frames].
        k: Current frame index.
        track_idx: Index of the track to evaluate.
        features_lat, features_lon: Flat arrays of all feature coordinates.
        w1, w2: Weights for cost function.
        phimax: Static penalty for links involving phantom points.

    Returns:
        The computed cost for the track triplet.
    """
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


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _check_max_missing(track: NDArray[np.int64], max_missing: int) -> bool:
    """
    Checks if a track exceeds the maximum allowed consecutive missing frames.

    Args:
        track: Array of feature indices for a single track.
        max_missing: Limit on consecutive phantoms (-1 for unlimited).

    Returns:
        True if the track is valid under the constraint.
    """
    if max_missing < 0:
        return True

    current_missing = 0
    first_real = -1
    last_real = -1
    for i in range(len(track)):
        if track[i] != -1:
            if first_real == -1:
                first_real = i
            last_real = i

    if first_real == -1:
        return True

    # Only count gaps between real start and end
    for i in range(first_real, last_real + 1):
        if track[i] == -1:
            current_missing += 1
            if current_missing > max_missing:
                return False
        else:
            current_missing = 0
    return True


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
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
    adapt_params: NDArray[np.float64],
    max_missing: int,
) -> tuple[int, int]:
    """
    A single MGE iteration step at frame k.

    Iterates through all possible track pairs and identifies the single BEST
    swap that reduces the total cost while satisfying all constraints.

    Args:
        tracks: Track matrix.
        features_lat, features_lon: Feature coordinate arrays.
        k: Current frame index.
        forward: If True, optimizes point k+1; otherwise k-1.
        w1, w2: Cost weights.
        default_dmax: Default search radius.
        phimax: Phantom penalty.
        zones: Regional dmax definitions.
        adapt_params: Adaptive smoothness definitions.
        max_missing: Missing frame limit.

    Returns:
        (best_i, best_j) indices of the track pair to swap, or (-1, -1).
    """
    n_tracks = tracks.shape[0]
    best_gain = 1e-8
    best_i = -1
    best_j = -1

    # Target frame to swap
    target_k = k + 1 if forward else k - 1

    rad_to_deg = 180.0 / np.pi
    deg_to_rad = np.pi / 180.0

    # Cache current costs
    costs = np.zeros(n_tracks)
    for i in range(n_tracks):
        costs[i] = _get_cost(tracks, k, i, features_lat, features_lon, w1, w2, phimax)

    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
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
                    get_regional_dmax(lat_k, lon_k, zones, default_dmax)
                    + get_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                if geod_dist(lat_k, lon_k, lat_t, lon_t) > dmax_i * deg_to_rad:
                    valid_swap = False

            idx_j_k = tracks[j, k]
            if valid_swap and idx_j_k != -1 and p_i_orig != -1:
                lat_k, lon_k = features_lat[idx_j_k], features_lon[idx_j_k]
                lat_t, lon_t = features_lat[p_i_orig], features_lon[p_i_orig]
                dmax_j = 0.5 * (
                    get_regional_dmax(lat_k, lon_k, zones, default_dmax)
                    + get_regional_dmax(lat_t, lon_t, zones, default_dmax)
                )
                if geod_dist(lat_k, lon_k, lat_t, lon_t) > dmax_j * deg_to_rad:
                    valid_swap = False

            if not valid_swap:
                continue

            # 2. Max Missing Check
            tracks[i, target_k] = p_j_orig
            tracks[j, target_k] = p_i_orig

            if not _check_max_missing(tracks[i], max_missing) or not _check_max_missing(
                tracks[j], max_missing
            ):
                tracks[i, target_k] = p_i_orig
                tracks[j, target_k] = p_j_orig
                continue

            # 3. Cost Gain Calculation
            new_cost_i = _get_cost(
                tracks, k, i, features_lat, features_lon, w1, w2, phimax
            )
            new_cost_j = _get_cost(
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
                phi_max_i = get_adaptive_phimax(
                    0.5 * (d1 + d2) * rad_to_deg, adapt_params, phimax
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
                phi_max_j = get_adaptive_phimax(
                    0.5 * (d1 + d2) * rad_to_deg, adapt_params, phimax
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


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _initial_break_pass(
    tracks: NDArray[np.int64],
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    w1: float,
    w2: float,
    phimax: float,
    adapt_params: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Identifies tracks that violate smoothness constraints after initial linking
    and breaks them into separate tracks.

    Args:
        tracks: Track matrix after nearest-neighbor linking.
        features_lat, features_lon: Coordinate arrays.
        w1, w2: Cost weights.
        phimax: Phantom penalty.
        adapt_params: Adaptive smoothness definitions.

    Returns:
        A new track matrix with broken tracks appended as new rows.
    """
    n_tracks, n_frames = tracks.shape
    new_tracks_list = []
    rad_to_deg = 180.0 / np.pi

    for i in range(n_tracks):
        current_track = tracks[i]
        last_break = 0
        for k in range(1, n_frames - 1):
            if (
                current_track[k - 1] != -1
                and current_track[k] != -1
                and current_track[k + 1] != -1
            ):
                cost = geod_dev(
                    features_lat[current_track[k - 1]],
                    features_lon[current_track[k - 1]],
                    features_lat[current_track[k]],
                    features_lon[current_track[k]],
                    features_lat[current_track[k + 1]],
                    features_lon[current_track[k + 1]],
                    w1,
                    w2,
                )

                d1 = geod_dist(
                    features_lat[current_track[k - 1]],
                    features_lon[current_track[k - 1]],
                    features_lat[current_track[k]],
                    features_lon[current_track[k]],
                )
                d2 = geod_dist(
                    features_lat[current_track[k]],
                    features_lon[current_track[k]],
                    features_lat[current_track[k + 1]],
                    features_lon[current_track[k + 1]],
                )
                phi_max = get_adaptive_phimax(
                    0.5 * (d1 + d2) * rad_to_deg, adapt_params, phimax
                )

                if cost > phi_max:
                    # Break track at point k
                    new_tr = np.full(n_frames, -1, dtype=np.int64)
                    new_tr[last_break : k + 1] = current_track[last_break : k + 1]
                    new_tracks_list.append(new_tr)
                    last_break = k + 1

        # Add remaining part
        new_tr = np.full(n_frames, -1, dtype=np.int64)
        new_tr[last_break:] = current_track[last_break:]
        new_tracks_list.append(new_tr)

    # Convert list to 2D array
    out = np.full((len(new_tracks_list), n_frames), -1, dtype=np.int64)
    for i in range(len(new_tracks_list)):
        out[i] = new_tracks_list[i]
    return out


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _break_track(
    tracks: NDArray[np.int64],
    track_idx: int,
    k: int,
    features_lat: NDArray[np.float64],
    features_lon: NDArray[np.float64],
    zones: NDArray[np.float64],
    default_dmax: float,
    forward: bool,
) -> NDArray[np.int64]:
    """
    Breaks a track at frame k if the displacement to the next/previous point
    violates the search radius constraint.

    This matches the TRACK 'track_fail' behavior.

    Args:
        tracks: The track matrix.
        track_idx: Index of the track to check.
        k: Frame index where the potential break starts.
        features_lat, features_lon: Coordinate arrays.
        zones: Regional dmax definitions.
        default_dmax: Default search radius.
        forward: If True, check k to k+1; otherwise k to k-1.

    Returns:
        The updated track matrix (potentially with a new row).
    """
    n_tracks, n_frames = tracks.shape
    deg_to_rad = np.pi / 180.0

    target_k = k + 1 if forward else k - 1
    if target_k < 0 or target_k >= n_frames:
        return tracks

    idx1 = tracks[track_idx, k]
    idx2 = tracks[track_idx, target_k]

    if idx1 == -1 or idx2 == -1:
        return tracks

    lat1, lon1 = features_lat[idx1], features_lon[idx1]
    lat2, lon2 = features_lat[idx2], features_lon[idx2]

    dmax_eff = 0.5 * (
        get_regional_dmax(lat1, lon1, zones, default_dmax)
        + get_regional_dmax(lat2, lon2, zones, default_dmax)
    )

    if geod_dist(lat1, lon1, lat2, lon2) > dmax_eff * deg_to_rad:
        # Violation! Break the track.
        new_tr = np.full(n_frames, -1, dtype=np.int64)
        if forward:
            # Move k+1 onwards to a new track
            new_tr[target_k:] = tracks[track_idx, target_k:]
            tracks[track_idx, target_k:] = -1
        else:
            # Move k-1 backwards to a new track
            new_tr[:k] = tracks[track_idx, :k]
            tracks[track_idx, :k] = -1

        # Append new track as a new row
        # (Numba cannot easily resize the matrix, so we return a new one)
        out = np.zeros((n_tracks + 1, n_frames), dtype=np.int64)
        out[:n_tracks] = tracks
        out[n_tracks] = new_tr
        return out

    return tracks


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _numba_ccl(
    binary_mask: NDArray[np.float64],
) -> tuple[NDArray[np.int32], int]:
    """
    8-connectivity Connected Component Labeling (CCL) in Numba.
    Uses iterative label propagation for compatibility with JIT.

    Args:
        binary_mask: Binary 2D array (1.0 for object, 0.0 for background).

    Returns:
        (labeled_mask, num_objects)
    """
    ny, nx = binary_mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    label_count = 0

    # Initial labeling
    for i in range(ny):
        for j in range(nx):
            if binary_mask[i, j] > 0:
                label_count += 1
                labels[i, j] = label_count

    if label_count == 0:
        return labels, 0

    # Iterative propagation until convergence
    changed = True
    while changed:
        changed = False
        # Forward pass
        for i in range(ny):
            for j in range(nx):
                if labels[i, j] == 0:
                    continue
                cur = labels[i, j]
                # 8-neighbors with longitude wrapping
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = (j + dj) % nx
                        if labels[ni, nj] > 0 and labels[ni, nj] < cur:
                            cur = labels[ni, nj]
                if cur != labels[i, j]:
                    labels[i, j] = cur
                    changed = True

        # Backward pass
        for i in range(ny - 1, -1, -1):
            for j in range(nx - 1, -1, -1):
                if labels[i, j] == 0:
                    continue
                cur = labels[i, j]
                for di in range(-1, 2):
                    ni = i + di
                    if ni < 0 or ni >= ny:
                        continue
                    for dj in range(-1, 2):
                        nj = (j + dj) % nx
                        if labels[ni, nj] > 0 and labels[ni, nj] < cur:
                            cur = labels[ni, nj]
                if cur != labels[i, j]:
                    labels[i, j] = cur
                    changed = True

    # Compact labels
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    num_objects = len(unique_labels)
    label_map = np.zeros(label_count + 1, dtype=np.int32)
    for i in range(num_objects):
        label_map[unique_labels[i]] = i + 1

    for i in range(ny):
        for j in range(nx):
            labels[i, j] = label_map[labels[i, j]]

    return labels, num_objects


@nb.njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _numba_object_extrema(
    frame: NDArray[np.float64],
    labeled_mask: NDArray[np.int32],
    num_objects: int,
    size: int,
    is_min: bool,
    min_points: int,
) -> NDArray[np.float64]:
    """
    Finds local extrema within thresholded objects.

    Matches TRACK's object-based feature identification.

    Args:
        frame: 2D data frame.
        labeled_mask: Labeled object mask from _numba_ccl.
        num_objects: Total number of objects.
        size: Local search diameter.
        is_min: True for minima, False for maxima.
        min_points: Minimum number of grid points in an object to be processed.

    Returns:
        Binary mask of detected extrema.
    """
    ny, nx = frame.shape
    extrema = np.zeros_like(frame)
    half = size // 2

    # Calculate object sizes
    object_sizes = np.zeros(num_objects + 1, dtype=np.int32)
    for i in range(ny):
        for j in range(nx):
            if labeled_mask[i, j] > 0:
                object_sizes[labeled_mask[i, j]] += 1

    for i in range(ny):
        for j in range(nx):
            obj_id = labeled_mask[i, j]
            if obj_id == 0 or object_sizes[obj_id] < min_points:
                continue

            val = frame[i, j]
            is_extrema = True
            for di in range(-half, half + 1):
                ni = i + di
                if ni < 0 or ni >= ny:
                    continue
                for dj in range(-half, half + 1):
                    if di == 0 and dj == 0:
                        continue
                    nj = (j + dj) % nx
                    nval = frame[ni, nj]

                    if is_min:
                        if nval < val:
                            is_extrema = False
                            break
                    else:
                        if nval > val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break

            if is_extrema:
                extrema[i, j] = 1.0

    return extrema
