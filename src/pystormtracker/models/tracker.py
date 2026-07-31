from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .tracks import Tracks

if TYPE_CHECKING:
    from .geo import MapExtent

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
    All classes implementing these methods are considered Trackers.
    """

    def track(
        self,
        infile: str | Path | xr.DataArray | xr.Dataset,
        varname: str,
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        map_proj: Literal["global", "nh_stereo", "sh_stereo", "healpix"] = "global",
        resolution: float = 100.0,
        extent: MapExtent | None = None,
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
        max_chunk_size: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
        overlap: int = 3,
        min_points: int = 1,
        filter: bool = True,
        lmin: int = 5,
        lmax: int = 42,
        taper_points: int = 0,
        subgrid_refine: bool = False,
        **kwargs: float | int | str | None,
    ) -> Tracks: ...
