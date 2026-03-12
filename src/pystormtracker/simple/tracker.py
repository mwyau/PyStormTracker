from __future__ import annotations

from typing import Literal

import numpy as np

from ..models import TimeRange, Tracks
from .detector import RawDetectionStep, SimpleDetector
from .linker import SimpleLinker


def _link_centers(raw_steps: list[RawDetectionStep]) -> Tracks:
    tracks = Tracks()
    linker = SimpleLinker()
    for step_data in raw_steps:
        linker.append(tracks, step_data)
    return tracks


def _detect_and_link(
    detector: SimpleDetector, size: int, threshold: float, time_chunk_size: int, mode: Literal["min", "max"]
) -> Tracks:
    """Worker task: Detects centers and immediately links them into a local Tracks object."""
    raw_steps = detector.detect(
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
        start_time: str | np.datetime64 | None = None,
        end_time: str | np.datetime64 | None = None,
        mode: Literal["min", "max"] = "min",
        backend: Literal["serial", "mpi", "dask"] = "serial",
        n_workers: int | None = None,
    ) -> Tracks:

        time_range = None
        if start_time is not None or end_time is not None:
            # Requires opening the file to find exact matching start/end bounds if one is missing,
            # but TimeRange handles exact numpy datetimes well.
            # We'll rely on the Detector's sel(slice()) to handle boundaries.
            st = np.datetime64(start_time) if start_time else None
            et = np.datetime64(end_time) if end_time else None

            # Since TimeRange expects exact np.datetime64, we supply them
            # For simplicity, if one is missing, we use NaT (Not a Time)
            if st is None:
                st = np.datetime64("NaT")
            if et is None:
                et = np.datetime64("NaT")

            time_range = TimeRange(start=st, end=et)
        if backend == "mpi":
            from .concurrent import run_simple_mpi
            tracks = run_simple_mpi(infile, varname, time_range, mode)
        elif backend == "dask":
            from .concurrent import run_simple_dask
            tracks = run_simple_dask(infile, varname, time_range, mode, n_workers)
        else:
            tracks = self._detect_serial(infile, varname, time_range, mode)

        return tracks
