from __future__ import annotations

import timeit
from typing import Literal

from pystormtracker.simple.tracker import SimpleTracker
from pystormtracker.utils.data_utils import fetch_era5_msl


def benchmark() -> None:
    # Fetch the 0.25x0.25 MSL data
    infile = fetch_era5_msl(resolution="0.25x0.25")
    varname = "msl"
    mode: Literal["min", "max"] = "min"

    workers_list = [1, 8]
    results = {}

    print(f"Benchmarking with file: {infile}")
    print(f"{'Workers':<10} | {'Total Time (s)':<18}")
    print("-" * 35)

    tracker = SimpleTracker()

    for w in workers_list:
        print(f"\n--- Running with {w} workers ---")

        start_time = timeit.default_timer()

        if w > 1:
            tracker.track(infile, varname, mode=mode, backend="dask", n_workers=w)
        else:
            tracker.track(infile, varname, mode=mode, backend="serial")

        total_time = timeit.default_timer() - start_time
        results[w] = total_time
        print(f"Workers: {w:<2} | Total Time: {total_time:>10.4f}s")

    summary = "\nBenchmark Summary (0.25 MSL - Detect + Link + Reduce):\n"
    for w in sorted(results.keys()):
        t = results[w]
        speedup = results[1] / t if t > 0 else 0
        summary += f"Workers: {w}, Time: {t:.4f}s, Speedup: {speedup:.2f}x\n"

    print(summary)

    with open("benchmark_results_025.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    benchmark()
