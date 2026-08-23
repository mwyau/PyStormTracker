"""HEALPix detector defaults."""

from __future__ import annotations

from typing import Final

# These defaults belong to the HEALPix object detector. They currently match
# the corresponding Hodges values, but remain independent of SimpleTracker.
DEFAULT_MSL_OBJECT_THRESHOLD: Final[float] = 0.0
DEFAULT_VO_OBJECT_THRESHOLD: Final[float] = 1.0e-5
SPECTRAL_TAPER_DEFAULT: Final[float] = 0.1
