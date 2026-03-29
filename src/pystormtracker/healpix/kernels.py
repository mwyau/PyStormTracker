from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_healpix_ccl(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    threshold: float,
    is_min: bool,
) -> tuple[NDArray[np.int32], int]:
    """
    Graph-based Connected Component Labeling for 1D HEALPix grids.

    Args:
        data: 1D array of shape (N_pixels,).
        neighbor_table: 2D array of shape (8, N_pixels).
        threshold: Intensity threshold.
        is_min: If True, group pixels BELOW threshold. If False, ABOVE.

    Returns:
        (labels, num_objects)
    """
    npix = len(data)
    labels = np.zeros(npix, dtype=np.int32)

    # 1. Initialize labels
    for p in range(npix):
        val = data[p]
        if is_min:
            if val <= threshold:
                labels[p] = p + 1
        else:
            if val >= threshold:
                labels[p] = p + 1

    # 2. Iterative label propagation
    changed = True
    while changed:
        changed = False
        for p in range(npix):
            if labels[p] == 0:
                continue

            cur_label = labels[p]
            min_label = cur_label

            for i in range(8):
                n_idx = neighbor_table[i, p]
                if n_idx != -1 and labels[n_idx] > 0 and labels[n_idx] < min_label:
                    min_label = labels[n_idx]

            if min_label < cur_label:
                labels[p] = min_label
                changed = True

    # 3. Compact labels to 1..N
    unique_labels = np.unique(labels)
    # unique_labels includes 0 if some pixels are background
    label_map = {0: 0}
    next_label = 1
    for ul in unique_labels:
        if ul != 0:
            label_map[ul] = next_label
            next_label += 1

    num_objects = next_label - 1
    for p in range(npix):
        labels[p] = label_map[labels[p]]

    return labels, num_objects


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_healpix_object_extrema(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    labeled_mask: NDArray[np.int32],
    num_objects: int,
    is_min: bool,
    min_points: int,
) -> NDArray[np.float64]:
    """
    Finds local extrema within thresholded objects on a HEALPix grid.
    """
    npix = len(data)
    extrema = np.zeros(npix, dtype=np.float64)

    # Calculate object sizes
    object_sizes = np.zeros(num_objects + 1, dtype=np.int32)
    for p in range(npix):
        if labeled_mask[p] > 0:
            object_sizes[labeled_mask[p]] += 1

    for p in range(npix):
        obj_id = labeled_mask[p]
        if obj_id == 0 or object_sizes[obj_id] < min_points:
            continue

        val = data[p]
        is_extrema = True

        # Check neighbors
        for i in range(8):
            n_idx = neighbor_table[i, p]
            if n_idx == -1:
                continue

            n_val = data[n_idx]
            if is_min:
                if n_val < val:
                    is_extrema = False
                    break
            else:
                if n_val > val:
                    is_extrema = False
                    break

        if is_extrema:
            extrema[p] = 1.0

    return extrema


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def subgrid_refine_healpix(
    data: NDArray[np.float64],
    p_idx: int,
    neighbor_table: NDArray[np.int64],
    pixel_lats: NDArray[np.float64],
    pixel_lons: NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Refines an extremum position on HEALPix using local quadratic surface fitting.
    Uses a local tangent plane projection centered at pixel p_idx.
    """
    # 1. Gather neighbors (max 9 points: center + 8 neighbors)
    neighbor_indices = np.zeros(9, dtype=np.int64)
    neighbor_indices[0] = p_idx
    n_pts = 1
    for i in range(8):
        n = neighbor_table[i, p_idx]
        if n != -1:
            neighbor_indices[n_pts] = n
            n_pts += 1

    if n_pts < 6:  # Need 6 points for quadratic fit
        return pixel_lats[p_idx], pixel_lons[p_idx], data[p_idx]

    # 2. Local Equirectangular Projection centered at p_idx
    # lat0, lon0 in radians
    lat0 = np.radians(pixel_lats[p_idx])
    lon0 = np.radians(pixel_lons[p_idx])

    x_proj = np.zeros(n_pts, dtype=np.float64)
    y_proj = np.zeros(n_pts, dtype=np.float64)
    z_vals = np.zeros(n_pts, dtype=np.float64)

    cos_lat0 = np.cos(lat0)

    for i in range(n_pts):
        idx = neighbor_indices[i]
        lat = np.radians(pixel_lats[idx])
        lon = np.radians(pixel_lons[idx])

        # Simple local projection
        dlon = lon - lon0
        if dlon > np.pi:
            dlon -= 2 * np.pi
        if dlon < -np.pi:
            dlon += 2 * np.pi

        x_proj[i] = dlon * cos_lat0
        y_proj[i] = lat - lat0
        z_vals[i] = data[idx]

    # Coordinate scaling for stability
    scale_x = np.max(np.abs(x_proj))
    scale_y = np.max(np.abs(y_proj))
    if scale_x < 1e-10:
        scale_x = 1.0
    if scale_y < 1e-10:
        scale_y = 1.0

    xs = x_proj / scale_x
    ys = y_proj / scale_y

    # 3. Fit quadratic surface: z = Ax^2 + By^2 + Cxy + Dx + Ey + F
    # X matrix: [x^2, y^2, x*y, x, y, 1]
    X = np.empty((n_pts, 6), dtype=np.float64)
    for i in range(n_pts):
        X[i, 0] = xs[i] ** 2
        X[i, 1] = ys[i] ** 2
        X[i, 2] = xs[i] * ys[i]
        X[i, 3] = xs[i]
        X[i, 4] = ys[i]
        X[i, 5] = 1.0

    XTX = X.T @ X
    XTz = X.T @ z_vals

    try:
        coeffs = np.linalg.solve(XTX, XTz)
        A, B, C = coeffs[0], coeffs[1], coeffs[2]
        D, E, F = coeffs[3], coeffs[4], coeffs[5]

        # 4. Find extreme: dz/dx = 2Ax + Cy + D = 0, dz/dy = 2By + Cx + E = 0
        det = 4 * A * B - C**2
        if abs(det) < 1e-12:
            return pixel_lats[p_idx], pixel_lons[p_idx], data[p_idx]

        dxs = (C * E - 2 * B * D) / det
        dys = (C * D - 2 * A * E) / det

        # Validation: check if refined point is "near" the original neighborhood
        if abs(dxs) > 2.0 or abs(dys) > 2.0:
            return pixel_lats[p_idx], pixel_lons[p_idx], data[p_idx]

        # Scale back
        dx = dxs * scale_x
        dy = dys * scale_y

        # 5. Inverse Projection
        lat_ref = lat0 + dy
        lon_ref = lon0 + dx / cos_lat0

        val_ref = A * dxs**2 + B * dys**2 + C * dxs * dys + D * dxs + E * dys + F

        return np.degrees(lat_ref), np.degrees(lon_ref) % 360.0, val_ref

    except Exception:
        return pixel_lats[p_idx], pixel_lons[p_idx], data[p_idx]


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_get_healpix_centers(
    extrema_mask: NDArray[np.float64],
    data: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Extracts the pixel indices and values of detected extrema.
    """
    idx = np.where(extrema_mask > 0)[0]
    vals = np.empty(len(idx), dtype=np.float64)
    for i in range(len(idx)):
        vals[i] = data[idx[i]]
    return idx, vals
