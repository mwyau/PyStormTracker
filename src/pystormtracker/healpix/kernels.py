from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _numba_healpix_extrema(
    data: NDArray[np.float64],
    neighbor_table: NDArray[np.int64],
    threshold: float | None,
    is_min: bool,
) -> NDArray[np.float64]:
    """
    Finds local extrema on a 1D HEALPix grid.

    Args:
        data: 1D array of shape (N_pixels,).
        neighbor_table: 2D array of shape (8, N_pixels) from Healpix_Base.neighbors().
        threshold: Intensity threshold.
        is_min: True to find minima, False for maxima.

    Returns:
        1D binary mask (same shape as data) with 1.0 at extrema locations.
    """
    npix = len(data)
    extrema_mask = np.zeros(npix, dtype=np.float64)

    for p in range(npix):
        val = data[p]

        # NaN or Inf check
        if np.isnan(val) or np.isinf(val):
            continue

        # Threshold check
        if threshold is not None:
            if is_min and val > threshold:
                continue
            if not is_min and val < threshold:
                continue

        is_extrema = True

        # Check all 8 immediate neighbors
        for i in range(8):
            n_idx = neighbor_table[i, p]

            # -1 indicates a missing neighbor (happens at 24 base corners)
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
            extrema_mask[p] = 1.0

    return extrema_mask

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
