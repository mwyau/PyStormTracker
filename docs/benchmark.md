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

### Primary medians

The speed label is derived from `TRACK median / PST median` and reports whether
PST is faster or slower.

| Case                   | Period    | Grid         | TRACK N | TRACK median s | PST N | PST median s | PST vs TRACK | TRACK tracks | PST tracks |
| ---------------------- | --------- | ------------ | ------: | -------------: | ----: | -----------: | ------------ | -----------: | ---------: |
| f320-to-t42-january    | January   | F320 -> T42  |       5 |          5.080 |     5 |        7.230 | 1.42x slower |          709 |        718 |
| f320-to-t42-full-year  | Full year | F320 -> T42  |       5 |         59.430 |     5 |       27.060 | 2.20x faster |         7761 |       7859 |
| f320-to-f320-january   | January   | F320 -> F320 |       5 |        167.960 |     5 |       43.440 | 3.87x faster |          779 |        789 |
| f320-to-f320-full-year | Full year | F320 -> F320 |       5 |       1997.160 |     5 |      342.470 | 5.83x faster |         8595 |       8747 |

### Compact trajectory agreement

These diagnostics compare the retained TRACK and PST default outputs using the
public nearest, mutual-nearest, and global-assignment matchers. They are
implementation-comparison diagnostics, not scientific ground truth.

| Case                   | Nearest TRACK coverage | Mutual agreement | Assignment precision | Assignment recall | Assignment F1 | Assignment median km | Assignment p95 km |
| ---------------------- | ---------------------: | ---------------: | -------------------: | ----------------: | ------------: | -------------------: | ----------------: |
| f320-to-t42-january    |                  1.000 |            1.000 |                0.987 |             1.000 |         0.994 |                0.006 |             0.048 |
| f320-to-t42-full-year  |                  0.998 |            0.998 |                0.986 |             0.998 |         0.992 |                0.006 |             0.050 |
| f320-to-f320-january   |                  0.999 |            0.999 |                0.986 |             0.999 |         0.992 |                0.007 |             0.059 |
| f320-to-f320-full-year |                  0.998 |            0.997 |                0.980 |             0.997 |         0.989 |                0.007 |             0.061 |

The full single-run timing table, worker-profile medians, provenance, frame
mapping, output hashes, and surviving raw-record paths are in the linked
Validation report and its [`summary.json`](https://github.com/mwyau/PyStormTracker-Validation/blob/main/results/benchmark-2024/summary.json).
