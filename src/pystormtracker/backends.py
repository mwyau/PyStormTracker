from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from typing import Literal, NamedTuple

import numpy as np
import xarray as xr
from numpy.typing import NDArray

type Backend = Literal["serial", "mpi", "dask"]

LOGGER = logging.getLogger(__name__)

NATIVE_THREAD_ENVIRONMENT: tuple[str, ...] = (
    "DUCC0_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


_PENDING_EXECUTORS: list[ThreadPoolExecutor] = []
_DEFER_DASK_INTERRUPT_CLEANUP: ContextVar[bool] = ContextVar(
    "pystormtracker_defer_dask_interrupt_cleanup", default=False
)


@contextmanager
def local_dask_executor(workers: int) -> Iterator[None]:
    """Own one Dask thread pool for the duration of a tracking computation.

    Dask's public ``pool`` configuration selects this executor without using
    private scheduler pool registries. Ordinary exceptions cancel queued work,
    wait for native work already running in a worker, and close the executor
    before control returns to the library caller. The CLI can explicitly defer
    that final drain for its first-Ctrl-C interruption protocol.
    """
    import dask

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="pystormtracker",
    )
    try:
        with dask.config.set(pool=executor):
            yield
    except BaseException as exc:
        defer_interrupt_cleanup = (
            isinstance(exc, KeyboardInterrupt) and _DEFER_DASK_INTERRUPT_CLEANUP.get()
        )
        executor.shutdown(
            wait=not defer_interrupt_cleanup,
            cancel_futures=True,
        )
        if defer_interrupt_cleanup:
            _PENDING_EXECUTORS.append(executor)
        raise
    else:
        executor.shutdown(wait=True)


def drain_pending_dask_executors() -> None:
    """Wait for active tasks left after an interrupted local Dask compute."""
    while _PENDING_EXECUTORS:
        executor = _PENDING_EXECUTORS.pop(0)
        executor.shutdown(wait=True, cancel_futures=False)


@contextmanager
def defer_dask_interrupt_cleanup() -> Iterator[None]:
    """Defer first-Ctrl-C Dask draining to the CLI interruption handler.

    This context is opt-in. Python-library callers get complete
    executor cleanup from :func:`local_dask_executor` for every exception.
    """
    token = _DEFER_DASK_INTERRUPT_CLEANUP.set(True)
    try:
        yield
    finally:
        _DEFER_DASK_INTERRUPT_CLEANUP.reset(token)


def available_cpu_count() -> int:
    """Return the number of CPU cores available to the current process.

    Checks ``os.process_cpu_count()`` (Python 3.13+), ``os.sched_getaffinity(0)``
    on platforms supporting CPU affinity, and falls back to ``os.cpu_count()``
    or 1. Always returns at least 1.
    """
    process_cpus = getattr(os, "process_cpu_count", None)
    if process_cpus is not None:
        try:
            count = process_cpus()
            if count is not None and count >= 1:
                resolved = int(count)
                LOGGER.debug("Available process CPU count: %d", resolved)
                return resolved
        except (AttributeError, OSError, NotImplementedError):
            pass

    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            affinity_count = len(sched_getaffinity(0))
            if affinity_count >= 1:
                resolved = int(affinity_count)
                LOGGER.debug("Available affinity CPU count: %d", resolved)
                return resolved
        except (AttributeError, OSError, NotImplementedError):
            pass

    count = os.cpu_count()
    if count is not None and count >= 1:
        resolved = int(count)
        LOGGER.debug("Available system CPU count: %d", resolved)
        return resolved

    LOGGER.debug("CPU count unavailable; using one worker")
    return 1


def resolve_dask_workers(workers: int | None = None) -> int:
    """Resolve Dask worker count, defaulting to available CPU count when None."""
    if workers is not None:
        _validate_positive_integer("workers", workers)
        resolved = int(workers)
        LOGGER.debug("Resolved explicit Dask worker count: %d", resolved)
        return resolved
    resolved = available_cpu_count()
    LOGGER.debug("Resolved default Dask worker count: %d", resolved)
    return resolved


def _validate_positive_integer(name: str, value: int) -> None:
    """Validate a public positive integer execution setting."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def resolve_frame_workers(
    frame_workers: int | None,
    backend: Backend,
) -> int:
    """Resolve frame-task concurrency for one backend.

    Dask uses the explicit count or available process CPUs. Serial and MPI
    execute frame work within their existing rank/process orchestration and
    therefore resolve this scheduler-only value to one.
    """
    if frame_workers is not None:
        _validate_positive_integer("frame_workers", frame_workers)
    resolved = resolve_dask_workers(frame_workers) if backend == "dask" else 1
    LOGGER.debug(
        "Resolved frame worker count: requested=%r resolved=%d backend=%s",
        frame_workers,
        resolved,
        backend,
    )
    return resolved


def resolve_mge_workers(
    mge_workers: int | None,
    backend: Backend,
) -> int:
    """Resolve MGE segment-task concurrency for one backend."""
    if mge_workers is not None:
        _validate_positive_integer("mge_workers", mge_workers)
    resolved = resolve_dask_workers(mge_workers) if backend == "dask" else 1
    LOGGER.debug(
        "Resolved MGE worker count: requested=%r resolved=%d backend=%s",
        mge_workers,
        resolved,
        backend,
    )
    return resolved


def resolve_sht_threads(
    sht_threads: int | None,
    backend: Backend,
) -> int:
    """Resolve DUCC0 threads per spherical-harmonic transform.

    The serial default is zero, which asks DUCC0 to use its available hardware
    threads. Parallel backends default to one thread per active frame/rank to
    preserve the existing oversubscription-safe behavior. Explicit values are
    always positive and are passed directly to DUCC0.
    """
    if sht_threads is not None:
        _validate_positive_integer("sht_threads", sht_threads)
        resolved = int(sht_threads)
    else:
        resolved = 0 if backend == "serial" else 1
    LOGGER.debug(
        "Resolved SHT thread count: requested=%r resolved=%d backend=%s",
        sht_threads,
        resolved,
        backend,
    )
    return resolved


def configure_sht_threads(sht_threads: int) -> None:
    """Ensure DUCC0 can honor an explicit per-transform thread count.

    DUCC0's nthreads argument is per transform, but its process-wide pool can
    be capped by DUCC0_NUM_THREADS or OMP_NUM_THREADS. The direct DUCC0 pool
    API avoids mutating either environment variable and lets an explicit
    request exceed an inherited native-pool cap.
    """
    environment = {name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT}
    if sht_threads <= 0:
        LOGGER.debug(
            "Using DUCC0 default SHT thread pool: requested=%d native_environment=%s",
            sht_threads,
            environment,
        )
        return

    import ducc0

    ducc0.misc.resize_thread_pool(sht_threads)
    LOGGER.debug(
        "Configured DUCC0 SHT thread pool: requested=%d available=%d "
        "native_environment=%s",
        sht_threads,
        ducc0.misc.thread_pool_size(),
        environment,
    )


def validate_execution_parameters(
    backend: Backend,
    workers: int | None = None,
    segment_frames: int | None = None,
    *,
    frame_workers: int | None = None,
    sht_threads: int | None = None,
    mge_workers: int | None = None,
) -> None:
    """Validate common and Hodges execution parameters."""
    if backend not in ("serial", "dask", "mpi"):
        raise ValueError(
            f"unsupported backend {backend!r}; expected 'serial', 'dask', or 'mpi'"
        )
    if segment_frames is not None and segment_frames <= 0:
        raise ValueError(f"segment_frames must be positive, got {segment_frames}")
    if workers is not None:
        _validate_positive_integer("workers", workers)
        if backend == "mpi":
            raise ValueError(
                "workers parameter cannot be used with MPI backend; "
                "rank count is determined by MPI.COMM_WORLD"
            )
    for name, value in (
        ("frame_workers", frame_workers),
        ("sht_threads", sht_threads),
        ("mge_workers", mge_workers),
    ):
        if value is not None:
            _validate_positive_integer(name, value)

    if backend != "dask":
        if frame_workers is not None:
            raise ValueError(
                f"frame_workers is only supported with the Dask backend, not {backend}"
            )
        if mge_workers is not None:
            raise ValueError(
                f"mge_workers is only supported with the Dask backend, not {backend}"
            )


class DaskTrackingFrames(NamedTuple):
    """Extracted lazy delayed frame blocks and spatial metadata for Dask frame tasks."""

    frame_blocks: list[object]
    times: NDArray[np.datetime64] | NDArray[np.int64] | NDArray[np.object_]
    lat_arr: NDArray[np.float64]
    lon_arr: NDArray[np.float64]
    n_steps: int
    periodic_x: bool
    projected_xy: bool


def extract_dask_frame_delayed_blocks(data_xr: xr.DataArray) -> DaskTrackingFrames:
    """Extract contiguous float64 delayed frame blocks and spatial metadata from an
    Xarray DataArray without eagerly computing the full time series.
    """
    from .io.data_loader import DataLoader

    loader = DataLoader(data_xr)
    time_dim, lat_name, lon_name = loader.get_coords()
    lat_arr = (
        np.asarray(data_xr[lat_name].values, dtype=np.float64)
        if lat_name in data_xr.coords
        else np.empty(0, dtype=np.float64)
    )
    lon_arr = (
        np.asarray(data_xr[lon_name].values, dtype=np.float64)
        if lon_name in data_xr.coords
        else lat_arr
    )
    n_steps = int(data_xr.sizes[time_dim])
    times = np.asarray(data_xr[time_dim].values)
    projected_xy = lon_name == "x"
    periodic_x = not projected_xy and loader.is_global_longitude()

    spatial_dims = [d for d in data_xr.dims if d != time_dim]
    ordered_dims = [time_dim, *spatial_dims]
    data_ordered = data_xr.transpose(*ordered_dims)
    if not hasattr(data_ordered.data, "dask"):
        spatial_chunks = dict.fromkeys(spatial_dims, -1)
        data_ordered = data_ordered.chunk({time_dim: 1, **spatial_chunks})
    elif hasattr(data_ordered.data, "chunks"):
        chunks = data_ordered.data.chunks
        needs_rechunk = chunks[0] != (1,) * n_steps or any(
            len(c) != 1 for c in chunks[1:]
        )
        if needs_rechunk:
            spatial_chunks = dict.fromkeys(spatial_dims, -1)
            data_ordered = data_ordered.chunk({time_dim: 1, **spatial_chunks})

    frame_blocks = list(data_ordered.data.to_delayed().ravel())
    if len(frame_blocks) != n_steps:
        raise ValueError(f"expected {n_steps} frame blocks, got {len(frame_blocks)}")

    LOGGER.debug(
        "Dask frame structure: frames=%d chunks=%r periodic_x=%s projected_xy=%s",
        n_steps,
        data_ordered.chunks,
        periodic_x,
        projected_xy,
    )
    return DaskTrackingFrames(
        frame_blocks=frame_blocks,
        times=times,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        n_steps=n_steps,
        periodic_x=periodic_x,
        projected_xy=projected_xy,
    )
