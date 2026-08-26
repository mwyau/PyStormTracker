# Benchmarks

The `benchmarks/` directory contains the generic PyStormTracker performance
runner. It records the input, runtime configuration, TrackJSON output, and
machine-readable metadata; it does not download data or compare with another
implementation.

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
execution settings.

## 2024 TRACK 1.5.4 comparison

The 2024 comparison uses the same 1,464 six-hourly ERA5 MSLP frames on the F320
Gaussian grid. TRACK 1.5.4 is the implementation reference, not scientific
ground truth. PyStormTracker uses rectangular B-spline refinement with T6–42
spectral filtering. The selected PST timing profile is
`frame=4 / SHT=4 / DUCC0=4`; those are PST controls and have no TRACK
counterpart.

The main trajectory comparison uses the RSPLICE-filtered tracks. TRACK
`ff_trs_neg` and PyStormTracker `filter_rsplice()` outputs use the same
8-point/10° endpoint criteria. RSPLICE is postprocessing and is not included in
the tracking wall times.

| Case      | Output grid | Filtered TRACK/PST tracks | One-to-one F1 | Median separation | 95th-percentile separation |
| --------- | ----------- | ------------------------: | ------------: | ----------------: | -------------------------: |
| January   | F320 → T42  |                 126 / 126 |        100.0% |             3.9 m |                     26.5 m |
| Full year | F320 → T42  |             1,471 / 1,471 |         99.7% |             4.1 m |                     33.2 m |
| January   | F320 → F320 |                 132 / 132 |        100.0% |             4.7 m |                     31.8 m |
| Full year | F320 → F320 |             1,544 / 1,550 |         99.7% |             4.9 m |                     38.9 m |

Raw pre-RSPLICE counts and F1 are also reported for comparison.

| Case      | Output grid | Raw TRACK/PST tracks | Raw one-to-one F1 |
| --------- | ----------- | -------------------: | ----------------: |
| January   | F320 → T42  |            709 / 718 |             99.4% |
| Full year | F320 → T42  |        7,761 / 7,859 |             99.2% |
| January   | F320 → F320 |            779 / 789 |             99.2% |
| Full year | F320 → F320 |        8,595 / 8,747 |             98.9% |

## Timing

Each value is the median of five sequential successful end-to-end runs on an
AMD Ryzen 9 5950X with 16 physical cores and 32 logical CPUs. The timing
comparison is machine-specific and does not explain causation.

| Period    | Output grid | TRACK 1.5.4 | PyStormTracker | Relative runtime |
| --------- | ----------- | ----------: | -------------: | ---------------: |
| January   | F320 → T42  |     5.080 s |        7.230 s |     1.42× slower |
| Full year | F320 → T42  |    59.430 s |       27.480 s |     2.16× faster |
| January   | F320 → F320 |   167.960 s |       11.610 s |    14.47× faster |
| Full year | F320 → F320 | 1,997.160 s |       57.380 s |    34.81× faster |

The F320 → T42 workflow reconstructs T6–42 onto a 64×128 T42 Gaussian grid.
The F320 → F320 workflow reconstructs onto the native 640×1280 grid. Within
each case, TRACK and PyStormTracker use the same input field, time period,
T6–42 spectral-filter settings, and matching criteria.
