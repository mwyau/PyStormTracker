from __future__ import annotations

import timeit
import dask
from dask.delayed import delayed
from dask.base import compute
from pystormtracker.simple import SimpleDetector
from pystormtracker.utils.data_utils import fetch_era5_msl


def benchmark() -> None:
    # Fetch the 0.25x0.25 MSL data
    infile = fetch_era5_msl(resolution="0.25x0.25")
    varname = "msl"

    workers_list = [1, 8]
    results = {}

    print(f"Benchmarking with file: {infile}")
    print(f"{'Workers':<10} | {'Detector Time (s)':<18}")
    print("-" * 35)

    for w in workers_list:
        print(f"\n--- Running with {w} workers ---")
        
        # Detector configuration
        size = 5
        threshold = 0.0
        time_chunk_size = 360
        mode = "min"
        
        start_time = timeit.default_timer()
        
        if w > 1:
            detector = SimpleDetector(pathname=infile, varname=varname)
            grids = detector.split(w)
            # Use detect_raw to avoid GIL overhead of Center objects
            delayed_results = [
                delayed(g.detect_raw)(
                    size=size,
                    threshold=threshold,
                    time_chunk_size=time_chunk_size,
                    minmaxmode=mode,
                )
                for g in grids
            ]
            with dask.config.set(num_workers=w):
                compute(*delayed_results)
        else:
            detector = SimpleDetector(pathname=infile, varname=varname)
            detector.detect_raw(
                size=size,
                threshold=threshold,
                time_chunk_size=time_chunk_size,
                minmaxmode=mode,
            )
            
        detector_time = timeit.default_timer() - start_time
        
        results[w] = detector_time
        print(f"Workers: {w:<2} | Detector Time: {detector_time:>10.4f}s")

    summary = "\nBenchmark Summary (0.25 MSL - Detector Time Only):\n"
    for w in sorted(results.keys()):
        t = results[w]
        speedup = results[1] / t if t > 0 else 0
        summary += f"Workers: {w}, Time: {t:.4f}s, Speedup: {speedup:.2f}x\n"
        
    print(summary)
    
    with open("benchmark_results_025.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    benchmark()
