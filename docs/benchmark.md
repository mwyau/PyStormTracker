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
