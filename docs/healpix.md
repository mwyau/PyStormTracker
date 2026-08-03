# HEALPix Support: Regridding and Tracking

This document describes spectral regridding to HEALPix and object detection on a one-dimensional HEALPix grid.

## 1. Overview
HEALPix represents a global field as equal-area, iso-latitude pixels with a neighbor graph. The one-dimensional topology avoids the pole singularity and latitude-dependent cell areas of a regular latitude-longitude mesh. PyStormTracker can regrid a regular or reduced-Gaussian field to HEALPix and run object detection on that graph.

## 2. Spectral Regridding (`SpectralRegridder`)
`SpectralRegridder` uses `ducc0.sht` for spherical harmonic analysis of an input grid and synthesis on a target grid.

- **Supported Inputs**: Clenshaw-Curtis (CC) and Gauss-Legendre (GL).
- **Supported Outputs**: CC, GL, and HEALPix.
- **Spectral Logic**:
    - **Analysis**: Extracts spherical harmonic coefficients ($a_{lm}$) from 2D grids using `ducc0.sht.analysis_2d`.
    - **Synthesis**: Projects coefficients onto the target grid. For HEALPix, it uses `ducc0.sht.synthesis` with `geometry` parameters derived from `ducc0.healpix.Healpix_Base.sht_info()`.
    - **Spectral Truncation**: Supports explicit $L_{max}$ and $M_{max}$ band limits. If omitted, the truncation is inferred from the input longitude count.

## 3. HEALPix Tracking Algorithm (`HealpixTracker`)
`HealpixTracker` implements the `Tracker` protocol for HEALPix pixel arrays. Tracking currently supports the serial backend.

### 3.1. 1D Graph Topology
Unlike 2D meshes where neighbors are found via index offsets, HEALPix neighbors are determined via a precomputed adjacency list.
- During initialization, the `HealpixDetector` generates a **neighbor table** of shape `(8, N_pixels)` using `ducc0.healpix.Healpix_Base.neighbors()`.
- All detection kernels operate on this 1D graph, using the table for topological lookups.

### 3.2. Connected Component Labeling (CCL)
The tracker groups adjacent pixels into objects with the Numba kernel `_numba_healpix_ccl`.
- **Algorithm**: Iterative label propagation over the 1D graph until convergence.
- **Constraints**: Supports `threshold` filtering and `min_points` object-size constraints.

### 3.3. Spherical Subgrid Refinement
Optional `subgrid_refine_healpix` applies these steps:
1.  **Local Projection**: For each detected extremum at pixel $P$, it projects $P$ and its 8 neighbors onto a local **equirectangular plane** centered at $P$.
2.  **Numerical Stability**: Coordinate scaling/normalization is applied to the local projected coordinates to prevent matrix ill-conditioning when solving the least-squares system.
3.  **Surface Fitting**: An unconstrained least-squares fit is performed to find the coefficients of a local quadratic surface: $z = Ax^2 + By^2 + Cxy + Dx + Ey + F$.
4.  **Analytical Stationary Point**: The candidate offset $(dx, dy)$ is found by solving the system where partial derivatives $\frac{\partial z}{\partial x} = 0$ and $\frac{\partial z}{\partial y} = 0$.
5.  **Inverse Projection**: The refined coordinates are projected back to standard Latitude and Longitude.

## 4. Engineering Standards
- **Dependencies**: HEALPix operations use the existing `ducc0` dependency; `healpy` is not required.
- **Numba JIT**: Graph traversal and local quadratic fitting kernels are compiled with `cache=True` and `nogil=True`.
- **Defaults**: HEALPix tracking does not apply optional spectral filtering
  unless both `lmin` and `lmax` are supplied. `taper_points` is independent.
  Regridding derives a finite transform bandwidth from the source grid and
  target `nside`; that bandwidth is not an optional filter. Subgrid refinement
  remains enabled by default. The `min_lifetime` constructor value is currently
  stored but not applied during HEALPix linking.
- **Tracker protocol and output**: `HealpixTracker` implements the common `Tracker` protocol and returns the array-backed `Tracks` model. Dask and MPI tracking are not implemented.

## 5. Usage Example
```python
import xarray as xr

from pystormtracker import HealpixTracker, SpectralRegridder

# Regrid one ERA5 frame (CC) to HEALPix (Nside=64)
ds = xr.open_dataset("data.nc")
field = ds["msl"]
frame = field.isel({field.dims[0]: 0})
regridder = SpectralRegridder()
da_hp = regridder.to_healpix(frame, nside=64)

# Track the first eight time steps after automatic HEALPix conversion
tracker = HealpixTracker()
tracks = tracker.track(
    infile=field.isel({field.dims[0]: slice(0, 8)}),
    variable_name="msl",
    mode="min",
    threshold=1000.0,
    lmin=5,
    lmax=42,
)
```

The first part demonstrates one-frame regridding; the second passes a regular
three-dimensional latitude-longitude field to `HealpixTracker.track()`, which
requests T5-42 filtering and HEALPix conversion. If `lmin` and `lmax` are
omitted, only the transform needed for HEALPix conversion is performed. For an
already regridded time-by-cell field, preprocessing must be performed before
tracking.
