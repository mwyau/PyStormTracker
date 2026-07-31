from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(cache=True, nogil=True)
def subgrid_refine(
    frame: NDArray[np.float64],
    r: int,
    c: int,
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    periodic_x: bool = True,
) -> tuple[float, float, float]:
    """Refine an extremum with a quadratic fit over its 3x3 neighborhood."""
    ny, nx = frame.shape

    if r < 1 or r >= ny - 1 or (not periodic_x and (c < 1 or c >= nx - 1)):
        return y[r], x[c], frame[r, c]

    cm = (c - 1) % nx if periodic_x else c - 1
    cp = (c + 1) % nx if periodic_x else c + 1

    z = np.empty((3, 3), dtype=np.float64)
    z[0, 0] = frame[r - 1, cm]
    z[0, 1] = frame[r - 1, c]
    z[0, 2] = frame[r - 1, cp]
    z[1, 0] = frame[r, cm]
    z[1, 1] = frame[r, c]
    z[1, 2] = frame[r, cp]
    z[2, 0] = frame[r + 1, cm]
    z[2, 1] = frame[r + 1, c]
    z[2, 2] = frame[r + 1, cp]

    f_yy = z[0, 1] - 2.0 * z[1, 1] + z[2, 1]
    f_xx = z[1, 0] - 2.0 * z[1, 1] + z[1, 2]
    f_yx = 0.25 * (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0])
    f_y = 0.5 * (z[2, 1] - z[0, 1])
    f_x = 0.5 * (z[1, 2] - z[1, 0])

    det = f_yy * f_xx - f_yx**2
    if abs(det) < 1e-10:
        return y[r], x[c], frame[r, c]

    dy = (f_yx * f_x - f_xx * f_y) / det
    dx = (f_yx * f_y - f_yy * f_x) / det
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return y[r], x[c], frame[r, c]

    if dy > 0.0:
        refined_y = y[r] + dy * (y[r + 1] - y[r])
    else:
        refined_y = y[r] + abs(dy) * (y[r - 1] - y[r])

    if periodic_x:
        if dx > 0.0:
            delta_x = (x[cp] - x[c] + 180.0) % 360.0 - 180.0
            refined_x = x[c] + dx * delta_x
        else:
            delta_x = (x[cm] - x[c] + 180.0) % 360.0 - 180.0
            refined_x = x[c] + abs(dx) * delta_x

        if np.min(x) < 0.0:
            refined_x = (refined_x + 180.0) % 360.0 - 180.0
        else:
            refined_x = refined_x % 360.0
    elif dx > 0.0:
        refined_x = x[c] + dx * (x[c + 1] - x[c])
    else:
        refined_x = x[c] + abs(dx) * (x[c - 1] - x[c])

    refined_value = z[1, 1] + 0.5 * (f_y * dy + f_x * dx)
    return refined_y, refined_x, refined_value
