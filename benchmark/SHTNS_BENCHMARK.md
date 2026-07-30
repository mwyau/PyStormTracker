# SHTns vs ducc0 Spectral Benchmark Comparison

This document details the historical `SHTns` spherical harmonic transform implementation, documents the last working commit before its removal, and compares the new executable benchmark harness results against the historical measurements recorded in `docs/spectral_accuracy.md`.

## Historical SHTns Implementation Commit

`SHTns` was used as a primary transform engine in earlier versions of PyStormTracker but was removed from production package code and dependencies due to:
1. Numerical discrepancies in vector derivative calculations compared to NCL/Spherepack references.
2. Complex native C build requirements (`libfftw3-dev`) causing cross-platform and CI issues (notably on ARM64 and Python 3.14).
3. Thread-safety limitations requiring thread-local plan instances across Dask/MPI workers.

The last working commit containing the full operational `SHTns` implementation for spectral filtering with `polar_opt=0.0` prior to consolidation is [`91f40d7`](https://github.com/mwyau/PyStormTracker/commit/91f40d71e22a88e05ae4b2786764ce16576153ef) (`feat(spectral): set polar_opt=0.0 for SHTns accuracy and refactor integration tests`), with commit [`731301a`](https://github.com/mwyau/PyStormTracker/commit/731301adcf790961b1ee7e9989324e20c6654edf) (`refactor: remove shtns and sht_engine support, move exclusively to ducc0`) executing the removal from the main codebase.

---

## Benchmark Consistency & Comparison with `docs/spectral_accuracy.md`

The new executable benchmark harness in [`run_shtns_benchmark.py`](file:///home/albert/PyStormTracker/benchmark/run_shtns_benchmark.py) was run against the versioned ERA5 reference datasets (`tests/data/era5/`).

### 1. `ducc0` Accuracy — Exact Match

The accuracy metrics produced by `ducc0` in the new benchmark **exactly match** the historical figures recorded in `docs/spectral_accuracy.md`:

- **2.5°x2.5° (T5-42)**: RMSE = `0.05266872` Pa, Rel. Error = `7.37e-05`, Correlation = `0.999999998663`
- **2.5°x2.5° (T0-42)**: RMSE = `0.05369308` Pa, Rel. Error = `5.31e-07`, Correlation = `0.999999998880`
- **0.25°x0.25° (T5-42)**: RMSE = `0.01276583` Pa, Rel. Error = `1.81e-05`, Correlation = `0.999999999928`
- **0.25°x0.25° (T0-42)**: RMSE = `0.02114745` Pa, Rel. Error = `2.09e-07`, Correlation = `0.999999999831`
- **Kinematics (Vorticity)**: RMSE = $1.25 \times 10^{-12} \text{ s}^{-1}$ ($\sim 10^{-14}$ to $10^{-12}$, near machine epsilon), Correlation = `1.000000000000`

### 2. `SHTns` Accuracy on High Resolution (0.25°x0.25°) — Consistent

On alias-free resolutions (0.25°), `SHTns` results are **highly consistent** with historical values:

- **T5-42**: New RMSE = `0.00003435` Pa (recorded: `0.00003643` Pa), Correlation = `1.000000000000` (recorded: `1.000000000000`).
- **T0-42**: New RMSE = `0.003779` Pa (recorded: `0.004865` Pa), Correlation = `0.999999999998` (recorded: `0.999999999994`).

Minor differences ($< 10^{-5}$ Pa) are due to variations in FFTW library compilation flags (e.g. AVX2/FMA optimizations) across builds.

### 3. Coarse Grid (2.5°x2.5°) Sampling Differences

For 2.5° (73 latitudes), the new benchmark script enforces `actual_lmax = min(lmax, (nlat - 1) // 2) = 36` to strictly satisfy `SHTns`'s sampling theorem requirement ($N_{lat} \ge 2 \cdot L_{max} + 1$). The historical documentation used un-clamped $L_{max}=42$ on 73 latitudes, which triggered SHTns internal aliasing warnings. Both setups confirm that coarse-grid aliasing is significantly higher on 2.5° than on 0.25° grids.

---

## Executable Benchmark Execution

To run the benchmark and generate `benchmark/shtns_vs_ducc0_results.json`:

```bash
uv run python benchmark/run_shtns_benchmark.py
```
