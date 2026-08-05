from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, Protocol, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .time import TimeInput, TimePoint
from .tracks import Tracks
from .units import ModeOption

Backend: TypeAlias = Literal["serial", "mpi", "dask"]
TrackingInput: TypeAlias = str | Path | xr.DataArray | xr.Dataset


class RawDetectionStep(NamedTuple):
    """A single time step's raw detection arrays."""

    time: TimePoint
    latitudes: NDArray[np.float64]
    longitudes: NDArray[np.float64]
    values: NDArray[np.float64]


class Tracker(Protocol):
    """A structural interface representing a storm tracker."""

    def track(
        self,
        infile: TrackingInput,
        variable_name: str,
        *,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        mode: ModeOption = "auto",
        threshold: float | None = None,
        engine: str | None = None,
    ) -> Tracks: ...
