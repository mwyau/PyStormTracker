from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .time import TimeRange
from .tracks import Tracks


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
        time_range: TimeRange | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "dask",
        n_workers: int | None = None,
    ) -> Tracks: ...
