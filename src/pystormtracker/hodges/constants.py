from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

# TRACK 1.5.4 compatibility defaults.  Scientific methodology and exact
# source semantics are documented separately from these owning constants.
W1_DEFAULT: Final[float] = 0.2
W2_DEFAULT: Final[float] = 0.8
DMAX_DEFAULT: Final[float] = 6.5
PHIMAX_DEFAULT: Final[float] = 0.5
MGE_MAX_ITERATIONS_DEFAULT: Final[int] = 3
MIN_TRACK_POINTS_DEFAULT: Final[int] = 3
MIN_OBJECT_GRID_POINTS_DEFAULT: Final[int] = 1
SPECTRAL_TAPER_DEFAULT: Final[float] = 1.0
TRACK_SMOOPY_OPTIMIZATION_SCALE_DEFAULT: Final[float] = 1.0

# Hodges object threshold defaults
DEFAULT_MSL_OBJECT_THRESHOLD: Final[float] = 0.0
DEFAULT_VO_OBJECT_THRESHOLD: Final[float] = 1e-5

# Regional dmax zones [lon_min, lon_max, lat_min, lat_max, dmax]
DEFAULT_DMAX_ZONES: Final[NDArray[np.float64]] = np.array(
    [
        [0.0, 360.0, -90.0, -20.0, 6.5],
        [0.0, 360.0, -20.0, 20.0, 3.0],
        [0.0, 360.0, 20.0, 90.0, 6.5],
    ],
    dtype=np.float64,
)

# Adaptive smoothness parameters matching adapt.dat
DEFAULT_ADAPTIVE_SMOOTHNESS: Final[NDArray[np.float64]] = np.array(
    [
        [1.0, 2.0, 5.0, 8.0],
        [1.0, 0.3, 0.1, 0.0],
    ],
    dtype=np.float64,
)
