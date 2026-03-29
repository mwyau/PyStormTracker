# HEALPix Support: Regridding and Tracking

This document provides a comprehensive technical overview of the HEALPix support implemented in PyStormTracker, covering both the spectral regridding capabilities and the 1D graph-based tracking algorithm.

## 1. Overview
The HEALPix implementation provides a high-performance path for processing global atmospheric datasets (like ERA5) natively on 1D spherical grids. By avoiding 2D meshes, we eliminate pole singularities and cell-area distortion, while maintaining mathematical exactness through spectral transforms.

## 2. Spectral Regridding (`SpectralRegridder`)
The `SpectralRegridder` (located in `src/pystormtracker/preprocessing/regrid.py`) utilizes `ducc0.sht` to perform mathematically precise transformations between various grid geometries.

- **Supported Inputs**: Clenshaw-Curtis (CC) and Gauss-Legendre (GL).
- **Supported Outputs**: CC, GL, and HEALPix.
- **Spectral Logic**:
    - **Analysis**: Extracts spherical harmonic coefficients ($a_{lm}$) from 2D grids using `ducc0.sht.analysis_2d`.
    - **Synthesis**: Projects coefficients onto the target grid. For HEALPix, it uses `ducc0.sht.synthesis` with `geometry` parameters derived from `ducc0.healpix.Healpix_Base.sht_info()`.
    - **Resolution Control**: Supports explicit $L_{max}$ and $M_{max}$ truncation to ensure band-limited consistency (defaulting to the input resolution).

## 3. HEALPix Tracking Algorithm (`HealpixTracker`)
The `HealpixTracker` implements the standard `Tracker` protocol but is specifically engineered for the 1D graph topology of HEALPix pixel arrays.

### 3.1. 1D Graph Topology
Unlike 2D meshes where neighbors are found via index offsets, HEALPix neighbors are determined via a precomputed adjacency list.
- During initialization, the `HealpixDetector` generates a **neighbor table** of shape `(8, N_pixels)` using `ducc0.healpix.Healpix_Base.neighbors()`.
- All detection kernels operate on this 1D graph, using the table for topological lookups.

### 3.2. Connected Component Labeling (CCL)
The tracker groups adjacent pixels into "objects" using a high-performance Numba kernel (`_numba_healpix_ccl`).
- **Algorithm**: Iterative label propagation over the 1D graph until convergence.
- **Constraints**: Supports `threshold` filtering and `min_points` object-size constraints.

### 3.3. Spherical Subgrid Refinement
To achieve high-precision coordinate accuracy, the tracker implements a sophisticated spherical refinement pipeline (`subgrid_refine_healpix`):
1.  **Local Projection**: For each detected extremum at pixel $P$, it projects $P$ and its 8 neighbors onto a local **equirectangular plane** centered at $P$.
2.  **Numerical Stability**: Coordinate scaling/normalization is applied to the local projected coordinates to prevent matrix ill-conditioning when solving the least-squares system.
3.  **Surface Fitting**: An unconstrained least-squares fit is performed to find the coefficients of a local quadratic surface: $z = Ax^2 + By^2 + Cxy + Dx + Ey + F$.
4.  **Analytical Optimization**: The precise extremum location $(dx, dy)$ is found by solving the system where partial derivatives $\frac{\partial z}{\partial x} = 0$ and $\frac{\partial z}{\partial y} = 0$.
5.  **Inverse Projection**: The refined coordinates are projected back to standard Latitude and Longitude.

## 4. Engineering Standards
- **Zero New Dependencies**: The entire implementation relies on the existing `ducc0` dependency. No `healpy` or other libraries are required.
- **Numba Acceleration**: All core graph-traversal and matrix-solving kernels are JIT-compiled with Numba (`nogil=True`, `cache=True`).
- **Standard Defaults**: Default spectral filtering is set to $L_{min}=5, L_{max}=42$ for all trackers to ensure meteorological consistency.
- **Protocol Adherence**: The `HealpixTracker` is a drop-in replacement for `SimpleTracker` or `HodgesTracker`, returning standard `Tracks` objects.

## 5. Usage Example
```python
from pystormtracker import HealpixTracker, SpectralRegridder

# 1. Regrid ERA5 (CC) to HEALPix (Nside=64)
regridder = SpectralRegridder(lmax=42)
da_hp = regridder.to_healpix(ds["msl"], nside=64)

# 2. Track on the HEALPix grid
tracker = HealpixTracker()
tracks = tracker.track(
    infile="hp_data.nc", # or pass DataArray directly via backend wrappers
    varname="msl",
    mode="min",
    threshold=1000.0,
    filter=True, # Apply T5-42 spectral filtering
)
```
