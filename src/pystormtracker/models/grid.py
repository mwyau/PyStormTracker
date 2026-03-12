from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .center import DetectedCenters


@runtime_checkable
class Grid(Protocol):
    """
    A structural interface representing a meteorological grid.
    Any class implementing these methods is considered a Grid.
    """

    def get_var(
        self, frame: int | tuple[int, int] | None = None
    ) -> NDArray[np.float64] | None: ...

    def get_time(self) -> NDArray[np.datetime64] | None: ...

    def get_lat(self) -> NDArray[np.float64] | None: ...

    def get_lon(self) -> NDArray[np.float64] | None: ...

    def split(self, num: int) -> list[Grid]: ...

    def detect(
        self,
        size: int,
        threshold: float,
        time_chunk_size: int,
        minmaxmode: Literal["min", "max"],
    ) -> DetectedCenters: ...
