# PyStormTracker Benchmark

The `benchmarks/` directory contains generic package-performance tools. The
benchmark runner measures the current CLI on an explicitly selected input and
does not download data or compare historical implementations.

Run a local benchmark with:

```bash
uv run python benchmarks/run_benchmark_detailed.py \
    --input tests/data/era5/era5_msl_2025-12_2.5x2.5.nc \
    --backends serial dask --workers 4
```

Benchmark results are workload- and machine-specific. Use representative
inputs, record the command and environment with any published result, and
keep validation-specific TRACK timing work in
`PyStormTracker-Validation`.

## 2024 TRACK comparison

The completed four-case TRACK 1.5.4 comparison is maintained in the public
[`PyStormTracker-Validation` report](https://github.com/mwyau/PyStormTracker-Validation/blob/main/results/BENCHMARK_2024.md).
It uses the fixed 2024 ERA5 F320 MSLP input and the rectangular B-spline
Hodges configuration at the source revision recorded in that report. TRACK
and the PST default profile each have five sequential repetitions; the report
also includes PST frame/SHT/DUCC0 profiles `4/4/4`, `2/8/8`, and `8/2/2`,
single-run timings, medians, output hashes, raw paths, and trajectory
comparison diagnostics. TRACK is the implementation reference for this
comparison, not scientific ground truth.
