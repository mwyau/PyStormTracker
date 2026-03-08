from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .center import Center


@runtime_checkable
class Grid(Protocol):
    """
    A structural interface representing a meteorological grid.
    Any class implementing these methods is considered a Grid.
    """

    def get_var(
        self, chart: int | tuple[int, int] | None = None
    ) -> np.ndarray | None: ...

    def get_time(self) -> np.ndarray | None: ...

    def get_time_obj(self) -> object | None: ...

    def get_lat(self) -> np.ndarray | None: ...

    def get_lon(self) -> np.ndarray | None: ...

    def split(self, num: int) -> list[Grid]: ...

    def detect(
        self,
        size: int,
        threshold: float,
        chart_buffer: int,
        minmaxmode: Literal["min", "max"],
    ) -> list[list[Center]]: ...
