# PyStormTracker Repository Instructions

Foundational mandates and engineering standards for `PyStormTracker`. These take precedence over general defaults.

## 1. Core Mandates

- **Hodges Parity**: MUST maintain algorithmic parity with TRACK (Hodges 1994, 1995, 1999) as detailed in `docs/HODGES.md`. Core kernels MUST use Numba.
- **Vectorized Architecture**: MUST use the array-backed data model and JIT-optimized kernels described in `docs/ARCHITECTURE.md`.
- **Validation**: Parallel results MUST be bit-wise identical to serial execution. Use Gather-then-Link orchestration.
- **Geometry**: All distance calculations MUST use the great-circle dot product formula with precision clamping.

## 2. Engineering Standards

- **Flexible APIs**: All `track()` implementations MUST accept `**kwargs` for cross-algorithm compatibility.
- **I/O**: MUST use coordinate-aware Xarray NetCDF/GRIB handling via `DataLoader`. Centralize remote data fetching in `src/pystormtracker/utils/data.py`.
- **Plain Language**: Use objective, scientific language in documentation. Avoid superlatives and subjective terms.
- **No Auto-Commit**: NEVER stage or commit changes unless specifically and explicitly requested by the user.
- **Documentation Persistence**: Preserve existing technical documentation and design rationale unless explicitly obsoleted.
