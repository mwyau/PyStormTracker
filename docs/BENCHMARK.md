# PyStormTracker Benchmark

This document provides a detailed breakdown of execution time (in seconds) comparing the legacy nested-object architecture (`v0.3.3`) and the modern, Numba JIT-compiled array architecture (`v0.4.0`).

## Methodology
- **Hardware**: AMD Ryzen 7 5800X (16 Threads), 48GB WSL Memory Limit.
- **Datasets**: ERA5 Mean Sea Level Pressure (MSL).
  - `2.5x2.5`: 144x73 grid, 360 time steps.
  - `0.25x0.25`: 1440x721 grid, 60 time steps.
- **Execution**: Component timings (Detection, Linking, Export, IO/Overhead) were extracted from the CLI.

## Resolution: 2.5x2.5
![2.5x2.5 Breakdown](https://raw.githubusercontent.com/mwyau/PyStormTracker/main/benchmark/benchmark_2_5x2_5_breakdown.png)

| Version | Backend | Workers | Det (s) | Link (s) | Exp (s) | IO (s) | Total (s) |
|---|---|---|---|---|---|---|---|
| v0.3.3 | SERIAL | 1 | 9.57 | 0.98 | 0.00 | 17.97 | **28.53** |
| v0.4.0 | SERIAL | 1 | 1.92 | 0.95 | 0.37 | 0.08 | **9.87** |
| v0.3.3 | DASK | 2 | 13.69 | 1.27 | 0.00 | 9.73 | **24.69** |
| v0.4.0 | DASK | 2 | 5.98 | 0.99 | 0.36 | 1.53 | **17.99** |
| v0.3.3 | DASK | 4 | 9.92 | 1.18 | 0.00 | 7.66 | **18.76** |
| v0.4.0 | DASK | 4 | 7.04 | 0.97 | 0.33 | 1.65 | **16.43** |
| v0.3.3 | DASK | 8 | 12.26 | 1.20 | 0.00 | 9.92 | **23.38** |
| v0.4.0 | DASK | 8 | 8.77 | 0.92 | 0.34 | 1.85 | **20.56** |
| v0.3.3 | MPI | 2 | 5.17 | 0.60 | 0.00 | 5.85 | **11.61** |
| v0.4.0 | MPI | 2 | 1.38 | 0.94 | 0.36 | 0.07 | **8.04** |
| v0.3.3 | MPI | 4 | 3.66 | 0.48 | 0.00 | 8.44 | **12.58** |
| v0.4.0 | MPI | 4 | 1.71 | 1.01 | 0.37 | 0.08 | **12.59** |
| v0.3.3 | MPI | 8 | 1.95 | 0.39 | 0.00 | 8.70 | **11.03** |
| v0.4.0 | MPI | 8 | 1.92 | 1.10 | 0.36 | 0.08 | **11.68** |

## Resolution: 0.25x0.25
![0.25x0.25 Breakdown](https://raw.githubusercontent.com/mwyau/PyStormTracker/main/benchmark/benchmark_0_25x0_25_breakdown.png)

| Version | Backend | Workers | Det (s) | Link (s) | Exp (s) | IO (s) | Total (s) |
|---|---|---|---|---|---|---|---|
| v0.3.3 | SERIAL | 1 | 158.94 | 94.92 | 0.00 | 26.78 | **280.65** |
| v0.4.0 | SERIAL | 1 | 3.93 | 17.51 | 1.69 | 0.60 | **32.38** |
| v0.3.3 | DASK | 2 | 95.30 | 118.92 | 0.00 | 22.49 | **236.72** |
| v0.4.0 | DASK | 2 | 8.16 | 16.98 | 1.69 | 2.00 | **37.11** |
| v0.3.3 | DASK | 4 | 50.97 | 117.18 | 0.00 | 20.75 | **188.90** |
| v0.4.0 | DASK | 4 | 8.30 | 17.87 | 1.84 | 2.02 | **38.58** |
| v0.3.3 | DASK | 8 | 39.83 | 117.42 | 0.00 | 19.98 | **177.23** |
| v0.4.0 | DASK | 8 | 11.96 | 17.58 | 1.86 | 2.36 | **44.23** |
| v0.3.3 | MPI | 2 | 85.68 | 56.25 | 0.00 | 13.04 | **154.97** |
| v0.4.0 | MPI | 2 | 2.74 | 17.16 | 1.71 | 0.61 | **29.37** |
| v0.3.3 | MPI | 4 | 51.10 | 35.84 | 0.00 | 3.92 | **90.86** |
| v0.4.0 | MPI | 4 | 2.82 | 17.64 | 1.76 | 0.67 | **32.23** |
| v0.3.3 | MPI | 8 | 34.18 | 28.50 | 0.00 | 4.95 | **67.63** |
| v0.4.0 | MPI | 8 | 3.29 | 18.04 | 1.87 | 0.67 | **32.90** |
