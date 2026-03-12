from __future__ import annotations

import os
from typing import Literal

from ..models import TimeRange, Tracks
from .detector import SimpleDetector, RawDetectionStep
from .linker import SimpleLinker


def _link_centers(raw_steps: list[RawDetectionStep]) -> Tracks:
    tracks = Tracks()
    linker = SimpleLinker()
    for step_data in raw_steps:
        linker.append_raw(tracks, step_data)
    return tracks


def _detect_and_link(
    detector: SimpleDetector, size: int, threshold: float, time_chunk_size: int, mode: Literal["min", "max"]
) -> Tracks:
    """Worker task: Detects centers and immediately links them into a local Tracks object."""
    raw_steps = detector.detect_raw(
        size=size, threshold=threshold, time_chunk_size=time_chunk_size, minmaxmode=mode
    )
    return _link_centers(raw_steps)


class SimpleTracker:
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

    def _detect_serial(
        self, infile: str, varname: str, time_range: TimeRange | None, mode: Literal["min", "max"]
    ) -> Tracks:
        detector = SimpleDetector(pathname=infile, varname=varname, time_range=time_range)
        tracks = _detect_and_link(detector, size=5, threshold=0.0, time_chunk_size=360, mode=mode)
        return tracks

    def track(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
    ) -> Tracks:
        if backend == "mpi":
            from .concurrent import run_simple_mpi
            tracks = run_simple_mpi(infile, varname, time_range, mode)
        elif backend == "dask":
            from .concurrent import run_simple_dask
            tracks = run_simple_dask(infile, varname, time_range, mode, n_workers)
        else:
            tracks = self._detect_serial(infile, varname, time_range, mode)
            
        return tracks
