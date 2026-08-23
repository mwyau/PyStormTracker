#!/usr/bin/env python3
"""Measure the pure 124-frame F320 spectral stage.

This helper excludes Hodges detection, spline construction, refinement, and
MGE.  It supports direct frame execution and the same lazy xarray/Dask SHT
graph used by the benchmark runner.  Each task returns a checksum so the
transforms cannot be optimized away without retaining the full output array.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import ducc0
import numpy as np
import xarray as xr

from pystormtracker.backends import configure_sht_threads, local_dask_executor
from pystormtracker.preprocessing.spectral import SHTFilter, _filter_sht_frame

DEFAULT_SOURCE: Final[Path] = Path(
    "/home/albert/PyStormTracker-Validation/results/"
    "track_comparison-20260818/inputs/ERA5_mslp_6hr_2024-01_DET.nc"
)
L_MIN: Final[int] = 6
L_MAX: Final[int] = 42
TAPER: Final[float] = 0.1
FRAME_COUNT: Final[int] = 124


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--backend", choices=("direct", "dask"), default="direct")
    parser.add_argument("--frame-workers", type=int, default=1)
    parser.add_argument(
        "--sht-threads",
        type=int,
        default=1,
        help="DUCC0 nthreads; use 0 for its default pool in direct mode",
    )
    parser.add_argument("--target", choices=("T42", "F320"), default="F320")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frame_workers <= 0 or args.repetitions <= 0 or args.sht_threads < 0:
        parser.error("frame-workers and repetitions must be positive; sht-threads >= 0")
    if args.backend == "dask" and args.sht_threads == 0:
        parser.error("dask mode requires an explicit positive sht-threads value")
    return args


def _target_shape(target: str) -> tuple[int, int]:
    return (64, 128) if target == "T42" else (640, 1280)


def _direct_frame(frame: np.ndarray, nlat: int, nlon: int, sht_threads: int) -> float:
    output = _filter_sht_frame(
        frame,
        L_MIN,
        L_MAX,
        lat_reverse=True,
        nthreads=sht_threads,
        taper_val=TAPER,
        geometry="GL",
        out_geometry="GL",
        out_ntheta=nlat,
        out_nphi=nlon,
    )
    return float(np.sum(output, dtype=frame.dtype))


def _run_direct(
    frames: np.ndarray,
    target: str,
    frame_workers: int,
    sht_threads: int,
) -> float:
    nlat, nlon = _target_shape(target)
    if sht_threads > 0:
        configure_sht_threads(sht_threads)
    with ThreadPoolExecutor(max_workers=frame_workers) as executor:
        checksums = executor.map(
            _direct_frame,
            frames,
            (nlat,) * frames.shape[0],
            (nlon,) * frames.shape[0],
            (sht_threads,) * frames.shape[0],
        )
        return float(sum(checksums))


def _run_dask(
    data: xr.DataArray,
    target: str,
    frame_workers: int,
    sht_threads: int,
) -> float:
    nlat, nlon = _target_shape(target)
    filtered = SHTFilter(
        lmin=L_MIN,
        lmax=L_MAX,
        taper_val=TAPER,
        geometry="auto",
        out_geometry="GL",
        out_ntheta=nlat,
        out_nphi=nlon,
        sht_threads=sht_threads,
    ).filter(data.chunk({"time": 1}), backend="dask")
    if not isinstance(filtered, xr.DataArray):
        raise TypeError("SHTFilter returned a non-xarray result")

    import dask

    block_arrays = filtered.data.to_delayed().ravel()
    block_checksums = [dask.delayed(np.sum)(block) for block in block_arrays]
    with local_dask_executor(frame_workers):
        checksums = dask.compute(*block_checksums)
    return float(sum(float(value) for value in checksums))


def main() -> None:
    args = _parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    with xr.open_dataset(args.source) as dataset:
        data = dataset["msl"].transpose("time", "latitude", "longitude")
        if data.sizes["time"] != FRAME_COUNT or data.shape[1:] != (640, 1280):
            raise ValueError(f"expected 124 F320 frames, got {data.shape}")
        started = time.perf_counter()
        frames = np.asarray(data.values)
        materialize_seconds = time.perf_counter() - started

        # Warm the imports, DUCC plan/cache, and Dask graph path before timing.
        if args.backend == "direct":
            _run_direct(frames[:1], args.target, 1, args.sht_threads)
        else:
            _run_dask(data.isel(time=slice(0, 1)), args.target, 1, args.sht_threads)

        wall_samples: list[float] = []
        user_samples: list[float] = []
        system_samples: list[float] = []
        checksums: list[float] = []
        for _ in range(args.repetitions):
            before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter()
            if args.backend == "direct":
                checksum = _run_direct(
                    frames,
                    args.target,
                    args.frame_workers,
                    args.sht_threads,
                )
            else:
                checksum = _run_dask(
                    data,
                    args.target,
                    args.frame_workers,
                    args.sht_threads,
                )
            finished = resource.getrusage(resource.RUSAGE_SELF)
            wall_samples.append(time.perf_counter() - started)
            user_samples.append(finished.ru_utime - before.ru_utime)
            system_samples.append(finished.ru_stime - before.ru_stime)
            checksums.append(checksum)

    payload: dict[str, object] = {
        "source": str(args.source),
        "frames": FRAME_COUNT,
        "source_shape": [640, 1280],
        "backend": args.backend,
        "target": args.target,
        "frame_workers": args.frame_workers,
        "sht_threads": args.sht_threads,
        "effective_ducc_pool_size": int(ducc0.misc.thread_pool_size()),
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "source_materialize_seconds": materialize_seconds,
        "wall_seconds": float(statistics.median(wall_samples)),
        "user_cpu_seconds": float(statistics.median(user_samples)),
        "system_cpu_seconds": float(statistics.median(system_samples)),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "checksum": checksums[-1],
        "checksum_equal_repetitions": len(set(checksums)) == 1,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
