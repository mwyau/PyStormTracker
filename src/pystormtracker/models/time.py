from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimeRange:
    """Metadata for the time range covered by a set of tracks."""

    start: np.datetime64
    end: np.datetime64
    step: np.timedelta64 | None = None
