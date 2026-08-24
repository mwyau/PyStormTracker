from __future__ import annotations

from typing import Final

# The 500-km association limit used in the Yau--Chang (2020) Simple Tracker
# concept; current detector/linker details are PyStormTracker behavior.
MAX_LINK_DISTANCE_KM: Final[float] = 500.0

# Default local extrema search window size
DEFAULT_SEARCH_WINDOW_SIZE: Final[int] = 5

# Simple feature detection threshold defaults
DEFAULT_MSL_FEATURE_THRESHOLD: Final[float] = 0.0
DEFAULT_VO_FEATURE_THRESHOLD: Final[float] = 1e-5
