from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .time import TimeInput, TimePoint
from .tracks import DetectionMode, Tracks

type TrackingInput = str | Path | xr.DataArray | xr.Dataset


class CenterFrame(NamedTuple):
    """A single time step's detected center coordinates and values."""

    time: TimePoint
    latitudes: NDArray[np.float64]
    longitudes: NDArray[np.float64]
    values: NDArray[np.float64]


class Tracker(Protocol):
    """A structural interface representing a storm tracker."""

    def track(
        self,
        data: TrackingInput,
        variable: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        detection_mode: DetectionMode = "auto",
        engine: str | None = None,
    ) -> Tracks: ...


__all__ = [
    "Tracker",
]
