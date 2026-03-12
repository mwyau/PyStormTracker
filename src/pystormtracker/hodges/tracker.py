from __future__ import annotations

from typing import Literal

import numpy as np

from ..models import Tracks
from ..models.tracker import Tracker


class HodgesTracker(Tracker):
    """
    A tracker implementing the adaptive constraints algorithm from Hodges (1999).
    This implementation is currently a placeholder.
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
    ) -> Tracks:
        """
        Runs the Hodges tracking algorithm. Not yet implemented.
        """
        raise NotImplementedError(
            "HodgesTracker is currently being implemented and is not yet available."
        )
