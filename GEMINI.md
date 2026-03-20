# PyStormTracker Repository Instructions

Foundational mandates and engineering standards for `PyStormTracker`. These take precedence over general defaults.

## 1. Algorithmic Mandates

### 1.1 Hodges (TRACK) Parity
- **Detection Pipeline**: MUST use the object-based sequence: `Threshold -> CCL -> Object Filtering -> Local Extrema`. Discard objects smaller than `min_points`.
- **CCL**: Implemented as iterative label propagation in Numba (proxy for legacy quad-trees).
- **Smoothness ($\psi$)**: Directional term weight MUST be normalized by $0.5$.
- **MGE Optimization**: MUST use alternating passes with "one best swap per frame" logic. Convergence is recursive until zero swaps occur.
- **Physical Safety**: Apply `track_fail` logic post-swap to split trajectories violating displacement limits ($d_{max}$).

### 1.2 Performance Architecture
- **Array-Backed Models**: All trajectories MUST be stored in flat NumPy arrays within the `Tracks` container to ensure zero-copy serialization and minimal overhead in Dask/MPI.
- **JIT Kernels**: All mathematical loops (Geodesic, Laplacian, Optimization) MUST be GIL-free Numba JIT kernels (`nogil=True`).
- **Sub-grid Refinement**: Use 2D local quadratic surface fitting as the standard proxy for B-splines.

## 2. Engineering Standards

### 2.1 Design Patterns
- **Flexible APIs**: All `track()` implementations MUST accept `**kwargs` to allow cross-algorithm flag passing without failures.
- **CLI Groups**: Arguments MUST be organized into logical groups (Required, General, Performance, Algorithm-Specific).
- **Spherical Geometry**: All distance calculations MUST use the great-circle dot product formula with precision clamping.

### 2.2 Modernization & I/O
- **Native Xarray**: Replace legacy binary/ASCII logic with coordinate-aware Xarray NetCDF/GRIB handling.
- **Centralized Data**: Use `src/pystormtracker/utils/data.py` (`CACHED_DATA`) for all remote asset fetching.
- **TOC Style**: Documentation titles in `index.md` MUST follow `PyStormTracker [Name]` and avoid colons.

## 3. General Guidelines
- **Documentation Persistence**: Avoid deleting existing technical documentation, architectural context, or design rationale unless it is explicitly obsoleted by new changes. Historical context should be preserved to aid future parity maintenance.
