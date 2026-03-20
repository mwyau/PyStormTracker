from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .tracks import Tracks

# Type alias for a single time step's raw detection arrays
RawDetectionStep: TypeAlias = tuple[
    np.datetime64,
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
]


@runtime_checkable
class Tracker(Protocol):
    """
    A structural interface representing a storm tracker.
    Any class implementing these methods is considered a Tracker.
    """

    def track(
        self,
        infile: str,
        varname: str,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
        engine: str | None = None,
    ) -> Tracks: ...
