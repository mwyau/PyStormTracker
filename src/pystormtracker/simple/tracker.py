from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from ..hodges import constants
from ..models import TimeRange, Tracks
from ..models.tracker import RawDetectionStep
from .detector import SimpleDetector
from .linker import SimpleLinker


def _link_centers(
    raw_steps: list[RawDetectionStep], time_range: TimeRange | None = None
) -> Tracks:
    """Sequentially links raw detection steps into a global Tracks object."""
    tracks = Tracks()
    if time_range:
        tracks.time_range = time_range
    linker = SimpleLinker()
    for step_data in raw_steps:
        linker.append(tracks, step_data)
    return tracks


def _detect_and_link(
    detector: SimpleDetector,
    size: int,
    threshold: float | None,
    mode: Literal["min", "max"],
) -> list[RawDetectionStep]:
    """Worker task: Detects centers and returns raw results for central linking."""
    return detector.detect(
        size=size,
        threshold=threshold,
        minmaxmode=mode,
    )


class SimpleTracker:
    """
    A tracker implementing the PyStormTracker simple parallel algorithm.
    """

    def preprocess_standard_track(
        self,
        data: xr.DataArray,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
    ) -> xr.DataArray:
        """
        Applies standard spectral preprocessing using ducc0.
        """
        from ..preprocessing.spectral import SpectralFilter
        from ..preprocessing.taper import TaperFilter

        # Ensure data is loaded into memory for spectral filtering
        if data.chunks:
            data = data.compute()

        from typing import cast

        # 1. Tapering
        if taper_points > 0:
            taper = TaperFilter(n_points=taper_points)
            data = cast(xr.DataArray, taper.filter(data))

        # 2. Spectral Filtering
        spectral_filter = SpectralFilter(lmin=lmin, lmax=lmax)
        data = spectral_filter.filter(data)

        return data

    def _detect_serial(
        self,
        infile: str,
        varname: str,
        time_range: TimeRange | None,
        mode: Literal["min", "max"],
        threshold: float | None = None,
        engine: str | None = None,
        filter: bool = False,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()
        detector_peek = SimpleDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )
        data_xr = detector_peek.get_xarray()

        if filter:
            data_xr = self.preprocess_standard_track(
                data_xr,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
            )
        t_pre = timeit.default_timer()
        print(f"    [Serial] Preprocessing time: {t_pre - t0:.4f}s")

        t0_detect = timeit.default_timer()
        detector = SimpleDetector.from_xarray(data_xr)
        size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
        raw_steps = _detect_and_link(
            detector, size=size, threshold=threshold, mode=mode
        )
        t1 = timeit.default_timer()
        print(f"    [Serial] Detection time: {t1 - t0_detect:.4f}s")

        t2 = timeit.default_timer()
        tracks = _link_centers(raw_steps, time_range=detector_peek.time_range)
        t3 = timeit.default_timer()
        print(f"    [Serial] Linking time: {t3 - t2:.4f}s")
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
        overlap: int = 3,
        min_points: int = 1,
        filter: bool = False,
        lmin: int = constants.LMIN_DEFAULT,
        lmax: int = constants.LMAX_DEFAULT,
        taper_points: int = constants.TAPER_DEFAULT,
        **kwargs: float | int | str | None,
    ) -> Tracks:
        import timeit

        t0 = timeit.default_timer()

        time_range = None
        if start_time is not None or end_time is not None:
            st = np.datetime64(start_time) if start_time else None
            et = np.datetime64(end_time) if end_time else None

            if st is None:
                st = np.datetime64("NaT")
            if et is None:
                et = np.datetime64("NaT")

            time_range = TimeRange(start=st, end=et)

        if backend == "mpi":
            from .concurrent import run_simple_mpi

            tracks = run_simple_mpi(
                infile,
                varname,
                time_range,
                mode,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                **kwargs,
            )
        elif backend == "dask":
            from .concurrent import run_simple_dask

            tracks = run_simple_dask(
                infile,
                varname,
                time_range,
                mode,
                n_workers,
                max_chunk_size=max_chunk_size,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                **kwargs,
            )
        else:
            tracks = self._detect_serial(
                infile,
                varname,
                time_range,
                mode,
                threshold=threshold,
                engine=engine,
                filter=filter,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                **kwargs,
            )

        t_end = timeit.default_timer()
        rank = 0
        if backend == "mpi":
            from mpi4py import MPI

            rank = MPI.COMM_WORLD.Get_rank()

        if rank == 0:
            print(f"Tracking time: {t_end - t0:.4f}s")

        return tracks
