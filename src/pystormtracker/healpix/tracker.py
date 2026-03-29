from __future__ import annotations

import timeit
from typing import Literal

import numpy as np

from ..models import TimeRange, Tracks
from ..models.tracker import RawDetectionStep
from .detector import HealpixDetector


def _link_centers(
    raw_steps: list[RawDetectionStep], time_range: TimeRange | None = None
) -> Tracks:
    """Sequentially links raw detection steps into a global Tracks object."""
    from ..simple.linker import SimpleLinker

    tracks = Tracks()
    if time_range:
        tracks.time_range = time_range
    linker = SimpleLinker()
    for step_data in raw_steps:
        linker.append(tracks, step_data)
    return tracks


def _detect_and_link(
    detector: HealpixDetector,
    threshold: float | None,
    mode: Literal["min", "max"],
) -> list[RawDetectionStep]:
    """Worker task: Detects centers on HEALPix and returns raw results."""
    return detector.detect(
        threshold=threshold,
        minmaxmode=mode,
    )


class HealpixTracker:
    """
    A tracker specifically designed for 1D HEALPix grids.
    Implements the PyStormTracker Tracker protocol.
    """

    def _detect_serial(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None,
        mode: Literal["min", "max"],
        threshold: float | None = None,
        engine: str | None = None,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        t0 = timeit.default_timer()
        detector = HealpixDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )

        raw_steps = _detect_and_link(detector, threshold=threshold, mode=mode)
        t1 = timeit.default_timer()
        print(f"    [Healpix] Detection time: {t1 - t0:.4f}s")

        t2 = timeit.default_timer()
        tracks = _link_centers(raw_steps, time_range=detector.time_range)
        t3 = timeit.default_timer()
        print(f"    [Healpix] Linking time: {t3 - t2:.4f}s")
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
        max_chunk_size: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        t0 = timeit.default_timer()

        time_range = None
        if start_time is not None or end_time is not None:
            st = np.datetime64(start_time) if start_time else np.datetime64("NaT")
            et = np.datetime64(end_time) if end_time else np.datetime64("NaT")
            time_range = TimeRange(start=st, end=et)

        if backend == "serial":
            tracks = self._detect_serial(
                infile,
                varname,
                time_range,
                mode,
                threshold=threshold,
                engine=engine,
                **kwargs,
            )
        else:
            # For now, only serial is implemented for HEALPix.
            # Dask/MPI can be added by implementing run_healpix_dask/mpi
            # similar to simple ones.
            raise NotImplementedError(
                f"Backend '{backend}' not yet implemented for HealpixTracker."
            )

        t_end = timeit.default_timer()
        print(f"Total HEALPix tracking time: {t_end - t0:.4f}s")

        return tracks
