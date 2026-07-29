# PyStormTracker Repository Instructions

Foundational mandates and engineering standards for `PyStormTracker`. These take precedence over general defaults.

## 1. Core Mandates

- **Hodges Parity**: MUST maintain algorithmic parity with TRACK (Hodges 1994, 1995, 1999) as detailed in `docs/hodges.md`. Core kernels MUST use Numba.
- **Vectorized Architecture**: MUST use the array-backed data model and JIT-optimized kernels described in `docs/architecture.md`.
- **Validation**: Parallel results MUST be bit-wise identical to serial execution. Use Gather-then-Link orchestration.
- **Geometry**: All distance calculations MUST use the great-circle dot product formula with precision clamping.
- **Backends**: Simple supports serial, Dask, and MPI tracking. Hodges and HEALPix are serial-only until Gather-then-Link parallel implementations exist.
- **Coordinates**: Use `DataLoader.is_global_longitude()` for periodicity. Projected `x/y` grids are nonperiodic; propagate `map_proj`, `resolution`, `extent`, and `lmax` through every backend and convert detections back to latitude/longitude before linking.

## 2. Engineering Standards

- **Flexible APIs**: All `track()` implementations MUST accept `**kwargs` for cross-algorithm compatibility.
- **I/O**: MUST use coordinate-aware Xarray NetCDF/GRIB handling via `DataLoader`. Centralize remote test data in `tests/utils.py`; integrations using checked-in or versioned PyStormTracker-Data fixtures count as verified for their tested scope.
- **Typing**: MUST NOT use `Any` typing. Provide explicit type annotations for all declarations.
- **Documentation**: State behavior, defaults, support limits, and evidence. Distinguish implemented, repository-tested, externally validated, and planned work. Preserve domain-specific scientific and algorithmic terms; define acronyms on first use and simplify prose without replacing precise terminology with generic wording. Do not use promotional wording or claim parity/performance without a named comparison.
- **No Auto-Commit**: NEVER stage or commit changes unless specifically and explicitly requested by the user.
- **Tooling**: MUST use `uv` for package management, environment handling, and running tools (e.g., `uv run ruff`, `uv run pytest`).
- **Documentation Persistence**: Preserve existing technical documentation and design rationale unless explicitly obsoleted.
- **Defaults**: CLI/orchestrator filtering and subgrid refinement are tri-state. Omitted means off for Simple and on for Hodges/HEALPix; direct tracker defaults must remain algorithm-specific.
- **VO Tests**: Production vorticity defaults to `1e-5`. The legacy Simple VO regression and bounded Hodges integration use `1e-4` only to match historical data and control runtime.
- **SHTns**: Keep SHTns out of production dependencies. It may be used by a standalone, reproducible comparison benchmark against ducc0.
