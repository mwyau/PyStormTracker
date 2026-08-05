from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ..time import TimeInput, TimePoint
from .tracks import Tracks
from .units import ModeOption

if TYPE_CHECKING:
    from .geo import MapExtent

TrackerOption: TypeAlias = float | int | str | None

# Type alias for a single time step's raw detection arrays
RawDetectionStep: TypeAlias = tuple[
    TimePoint,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def get_int_option(
    options: Mapping[str, TrackerOption],
    name: str,
    default: int,
) -> int:
    """Return an integer tracker option after validating its runtime type."""
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


@runtime_checkable
class Tracker(Protocol):
    """
    A structural interface representing a storm tracker.
    All classes implementing these methods are considered Trackers.
    """

    def track(
        self,
        infile: str | Path | xr.DataArray | xr.Dataset,
        variable_name: str,
        start_time: TimeInput | None = None,
        end_time: TimeInput | None = None,
        mode: ModeOption | None = "auto",
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
        lmin: int | None = None,
        lmax: int | None = None,
        taper_points: int = 0,
        nside: int | None = None,
        subgrid_refine: bool = False,
        **kwargs: TrackerOption,
    ) -> Tracks: ...
