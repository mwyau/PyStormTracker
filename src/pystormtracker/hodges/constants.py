from __future__ import annotations

import numpy as np

# Standard TRACK legacy defaults (Hodges 1994, 1995, 1999)
W1_DEFAULT = 0.2
W2_DEFAULT = 0.8
DMAX_DEFAULT = 6.5
PHIMAX_DEFAULT = 0.5
ITERATIONS_DEFAULT = 3
LIFETIME_DEFAULT = 3
MISSING_DEFAULT = 0

# Spectral Filter defaults (T5-42)
LMIN_DEFAULT = 5
LMAX_DEFAULT = 42

# Regional dmax zones [lon_min, lon_max, lat_min, lat_max, dmax]
TRACK_ZONES = np.array(
    [
        [0.0, 360.0, -90.0, -20.0, 6.5],
        [0.0, 360.0, -20.0, 20.0, 3.0],
        [0.0, 360.0, 20.0, 90.0, 6.5],
    ],
    dtype=np.float64,
)

# Adaptive smoothness distance thresholds (4 points)
ADAPT_THRESHOLDS = np.array([1.0, 2.0, 5.0, 8.0], dtype=np.float64)

# Adaptive smoothness phi values (4 points)
ADAPT_VALUES = np.array([1.0, 0.3, 0.1, 0.0], dtype=np.float64)
