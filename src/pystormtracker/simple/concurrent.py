from __future__ import annotations

import os
import timeit
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mpi4py import MPI

from ..hodges import constants
from ..models import TimeRange, Tracks
from ..models.tracker import RawDetectionStep
from .detector import SimpleDetector
from .tracker import _detect_and_link, _link_centers


def run_simple_dask(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    n_workers: int | None,
    max_chunk_size: int | None = None,
    threshold: float | None = None,
    engine: str | None = None,
    filter: bool = True,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
    taper_points: int = constants.TAPER_DEFAULT,
    sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
    **kwargs: float | int | str | None,
) -> Tracks:
    import dask

    if n_workers is None or n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 4)

    detector_peek = SimpleDetector(
        pathname=infile, varname=varname, time_range=time_range, engine=engine
    )
    data_xr = detector_peek.get_xarray()

    if filter:
        from .tracker import SimpleTracker

        data_xr = SimpleTracker().preprocess_standard_track(
            data_xr,
            lmin=lmin,
            lmax=lmax,
            taper_points=taper_points,
            sht_engine=sht_engine,
        )

    detector_obj = SimpleDetector.from_xarray(data_xr)

    # Decouple task chunks from worker count to prevent OOM on high-res data.
    times = detector_obj.get_time()
    total_steps = len(times) if times is not None else 1

    if max_chunk_size is None or max_chunk_size <= 0:
        max_chunk_size = 60
    else:
        max_chunk_size = max(1, max_chunk_size)

    # Ensure we at least split into n_workers tasks
    n_splits = max(n_workers, (total_steps + max_chunk_size - 1) // max_chunk_size)

    detectors = detector_obj.split(n_splits)

    t0 = timeit.default_timer()
    t1 = timeit.default_timer()
    print(f"    [Dask] Setup time: {t1 - t0:.4f}s")
    print(
        f"    [Dask] Splitting {total_steps} steps into {n_splits} "
        f"tasks (across {n_workers} threads)"
    )

    size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
    tasks = [
        dask.delayed(_detect_and_link)(d, size, threshold, mode)  # type: ignore[attr-defined]
        for d in detectors
    ]

    all_raw_chunks = dask.compute(*tasks, scheduler="threads", num_workers=n_workers)  # type: ignore[attr-defined]

    # Flatten chunks into a single sequence of steps
    all_raw_steps: list[RawDetectionStep] = [
        step for chunk in all_raw_chunks for step in chunk
    ]

    t2 = timeit.default_timer()
    print(f"    [Dask] Task execution & gather time: {t2 - t1:.4f}s")

    # Centralized linking guarantees bit-wise identity with Serial
    t3 = timeit.default_timer()
    tracks = _link_centers(all_raw_steps, time_range=time_range)
    t4 = timeit.default_timer()
    print(f"    [Dask] Linking time: {t4 - t3:.4f}s")
    return tracks


def run_simple_mpi(
    infile: str,
    varname: str,
    time_range: TimeRange | None,
    mode: Literal["min", "max"],
    threshold: float | None = None,
    engine: str | None = None,
    filter: bool = True,
    lmin: int = constants.LMIN_DEFAULT,
    lmax: int = constants.LMAX_DEFAULT,
    taper_points: int = constants.TAPER_DEFAULT,
    sht_engine: Literal["auto", "shtns", "ducc0"] = "auto",
    **kwargs: float | int | str | None,
) -> Tracks:
    from mpi4py import MPI

    comm: MPI.Intracomm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0

    t0 = timeit.default_timer()
    if rank == root:
        detector_peek = SimpleDetector(
            pathname=infile, varname=varname, time_range=time_range, engine=engine
        )
        data_xr = detector_peek.get_xarray()

        if filter:
            from .tracker import SimpleTracker

            data_xr = SimpleTracker().preprocess_standard_track(
                data_xr,
                lmin=lmin,
                lmax=lmax,
                taper_points=taper_points,
                sht_engine=sht_engine,
            )

        detector_obj = SimpleDetector.from_xarray(data_xr)
        detectors: list[SimpleDetector] | None = detector_obj.split(size)
    else:
        detectors = None

    detector: SimpleDetector = comm.scatter(detectors, root=root)
    t_scatter = timeit.default_timer()
    if rank == root:
        print(f"    [MPI] Prep & Scatter time: {t_scatter - t0:.4f}s")

    t1 = timeit.default_timer()
    ext_size = int(kwargs.get("size", 5))  # type: ignore[arg-type]
    raw_chunk = _detect_and_link(
        detector, size=ext_size, threshold=threshold, mode=mode
    )

    # Gather all raw chunks at root
    all_raw_chunks = comm.gather(raw_chunk, root=root)
    t3 = timeit.default_timer()

    if rank == root:
        print(f"    [MPI] Detection & Gather time: {t3 - t1:.4f}s")
        assert all_raw_chunks is not None
        all_raw_steps: list[RawDetectionStep] = [
            step for chunk in all_raw_chunks for step in chunk
        ]
        t4 = timeit.default_timer()
        tracks = _link_centers(all_raw_steps, time_range=time_range)
        t5 = timeit.default_timer()
        print(f"    [MPI] Linking time: {t5 - t4:.4f}s")
        return tracks

    # Non-root ranks return empty Tracks
    return Tracks()
