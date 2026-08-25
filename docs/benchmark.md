# Benchmarks

The `benchmarks/` directory contains tools for measuring current PyStormTracker
performance. The benchmark runner uses a selected input, records the runtime
configuration, and writes both TrackJSON output and machine-readable metadata.
It does not download data or compare against another implementation.

For example, to benchmark 124 frames with local Dask execution:

```bash
uv run python benchmarks/run_benchmark_detailed.py \
    --input tests/data/era5/era5_msl_2025-12_2.5x2.5.nc \
    --output /tmp/pystormtracker-benchmark.trackjson \
    --metadata /tmp/pystormtracker-benchmark.json \
    --frames 124 \
    --backend dask \
    --frame-workers 4 \
    --sht-threads 4 \
    --mge-workers 4
```

Benchmark results depend on the input, hardware, software environment, and
execution settings. Record these with published results. Detailed benchmark
outputs, timing results, and reproducibility records are stored in
[`PyStormTracker-Validation`](https://github.com/mwyau/PyStormTracker-Validation).

## 2024 TRACK comparison

The 2024 comparison against TRACK 1.5.4 uses the same ERA5 mean sea-level
pressure record for both implementations: 1,464 six-hourly frames on the F320
Gaussian grid. PyStormTracker uses the TRACK-compatible rectangular B-spline
refinement path. TRACK is the implementation reference for this comparison.

The benchmark ran on an AMD Ryzen 9 5950X with 16 physical cores and 32 logical
CPUs. The configuration, commands, individual timings, output hashes, and
trajectory diagnostics are stored in
[`BENCHMARK_2024.md`](https://github.com/mwyau/PyStormTracker-Validation/blob/main/results/BENCHMARK_2024.md).

Track counts and trajectory-agreement metrics below use raw trajectories before
RSPLICE filtering.

### Full-year results

| Filtered output grid | TRACK 1.5.4 | PyStormTracker | Relative runtime | One-to-one F1 | Median matched-track separation |
| --- | ---: | ---: | ---: | ---: | ---: |
| F320 → T42 | 59.43 s | 27.48 s | **2.16× faster** | 0.992 | 6 m |
| F320 → F320 | 1,997.16 s | 57.38 s | **34.81× faster** | 0.989 | 7 m |

Both workflows apply T6–42 spectral filtering. `F320 → T42` synthesizes the
filtered field onto the 64×128 T42 Gaussian grid before feature detection.
`F320 → F320` synthesizes the filtered field back onto the 640×1280 F320
Gaussian grid and performs detection and refinement at that resolution.

The timing comparison is machine-specific. The trajectory statistics quantify
agreement between the two implementations under this configuration.

### Timing results

Each TRACK and PyStormTracker value below is the median of five sequential
end-to-end runs.

| Period | Filtered output grid | TRACK 1.5.4 | PyStormTracker | PyStormTracker vs TRACK | TRACK tracks | PyStormTracker tracks |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| January | F320 → T42 | 5.08 s | 7.23 s | 1.42× slower | 709 | 718 |
| Full year | F320 → T42 | 59.43 s | 27.48 s | **2.16× faster** | 7,761 | 7,859 |
| January | F320 → F320 | 167.96 s | 11.61 s | **14.47× faster** | 779 | 789 |
| Full year | F320 → F320 | 1,997.16 s | 57.38 s | **34.81× faster** | 8,595 | 8,747 |

January F320 → T42 is the only case in this matrix where PyStormTracker is
slower than TRACK. The benchmark does not isolate the cause of this timing
difference.

### Trajectory agreement

`TRACK coverage` is the fraction of TRACK trajectories matched by the directed
nearest-track comparison. `Mutual agreement` requires the two trajectories to
select each other as nearest matches. `One-to-one F1` comes from the global
one-to-one assignment. Distances describe the matched trajectories from that
assignment.

| Period | Filtered output grid | TRACK coverage | Mutual agreement | One-to-one F1 | Median separation | 95th-percentile separation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| January | F320 → T42 | 100.0% | 100.0% | 0.994 | 6 m | 48 m |
| Full year | F320 → T42 | 99.8% | 99.8% | 0.992 | 6 m | 50 m |
| January | F320 → F320 | 99.9% | 99.9% | 0.992 | 7 m | 59 m |
| Full year | F320 → F320 | 99.8% | 99.7% | 0.989 | 7 m | 61 m |

The Validation report includes bidirectional and unmatched-track diagnostics.

### Parallel settings

The frame/SHT 4/4 configuration had the lowest measured wall time among the
tested worker profiles. `mge_workers` remained at its default value of 16.

The single-run timings, worker profiles, frame mapping, output hashes, and
raw-record locations are stored in the Validation report and its
[`summary.json`](https://github.com/mwyau/PyStormTracker-Validation/blob/main/results/benchmark-2024/summary.json).
