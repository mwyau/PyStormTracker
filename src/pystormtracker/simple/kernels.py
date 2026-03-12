from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_extrema_filter(
    data: NDArray[np.float64], size: int, threshold: float, is_min: bool
) -> NDArray[np.float64]:
    rows, cols = data.shape
    out = np.zeros_like(data)
    half_size = size // 2

    for r in range(half_size, rows - half_size):
        for c in range(cols):
            center_val = data[r, c]
            if np.isnan(center_val) or np.isinf(center_val):
                continue

            is_extrema = True
            for i in range(-half_size, half_size + 1):
                rr = r + i
                for j in range(-half_size, half_size + 1):
                    cc = (c + j) % cols
                    if is_min:
                        if data[rr, cc] < center_val:
                            is_extrema = False
                            break
                    else:
                        if data[rr, cc] > center_val:
                            is_extrema = False
                            break
                if not is_extrema:
                    break

            if is_extrema:
                if threshold == 0.0:
                    out[r, c] = 1.0
                else:
                    window = np.empty(size * size, dtype=data.dtype)
                    idx = 0
                    for i in range(-half_size, half_size + 1):
                        rr = r + i
                        for j in range(-half_size, half_size + 1):
                            cc = (c + j) % cols
                            window[idx] = data[rr, cc]
                            idx += 1
                    window.sort()
                    if is_min:
                        if window[8] - center_val > threshold:
                            out[r, c] = 1.0
                    else:
                        if window[-9] - center_val < -threshold:
                            out[r, c] = 1.0

    return out


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_laplace_masked(
    data: NDArray[np.float64], mask: NDArray[np.float64]
) -> NDArray[np.float64]:
    rows, cols = data.shape
    out = np.zeros_like(data)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] != 0:
                up = data[(r - 1) % rows, c]
                down = data[(r + 1) % rows, c]
                left = data[r, (c - 1) % cols]
                right = data[r, (c + 1) % cols]
                center = data[r, c]
                out[r, c] = (up + down + left + right - 4.0 * center) * mask[r, c]
    return out


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_remove_dup(laplacian: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    rows, cols = laplacian.shape
    out = np.zeros_like(laplacian)
    half_size = size // 2

    for r in range(rows):
        for c in range(cols):
            center_val = laplacian[r, c]
            if center_val != 0:
                is_max = True
                for i in range(-half_size, half_size + 1):
                    rr = (r + i) % rows
                    for j in range(-half_size, half_size + 1):
                        cc = (c + j) % cols
                        if laplacian[rr, cc] > center_val:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max:
                    out[r, c] = 1.0
    return out


@nb.njit(nogil=True, cache=True)
def _numba_get_centers(
    extrema: NDArray[np.float64], frame: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    rows, cols = extrema.shape
    # Pre-allocate a large enough buffer to avoid double pass
    # In practice, centers are sparse.
    max_centers = 10000  # Reasonable upper bound for a single frame
    r_idx_tmp = np.empty(max_centers, dtype=np.int64)
    c_idx_tmp = np.empty(max_centers, dtype=np.int64)
    vals_tmp = np.empty(max_centers, dtype=np.float64)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if extrema[r, c] != 0:
                if count < max_centers:
                    r_idx_tmp[count] = r
                    c_idx_tmp[count] = c
                    vals_tmp[count] = frame[r, c]
                    count += 1
                else:
                    # Fallback or error if too many centers
                    # For now, just stop at max_centers
                    break
        if count >= max_centers:
            break

    return r_idx_tmp[:count].copy(), c_idx_tmp[:count].copy(), vals_tmp[:count].copy()
