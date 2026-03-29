# Phase 2 Plan: HEALPix-Native Hodges Parity (TRACK Algorithm)

## Goal
Achieve algorithmic parity with the Hodges (TRACK) algorithm on HEALPix grids by implementing graph-based versions of the object-based detection pipeline.

## Architectural Requirements

### 1. Graph-Based Connected Component Labeling (CCL)
The current `HealpixTracker` uses point-wise extrema detection. Hodges parity requires group-based identification.
*   **Implement `_numba_healpix_ccl`**: A 1D CCL kernel that groups adjacent pixels (from the neighbor table) that exceed a threshold.
*   **Kernel logic**: Use a Union-Find or iterative labeling approach adapted for an arbitrary 1D graph.

### 2. Object Filtering
Once components are labeled, apply standard TRACK constraints:
*   **`min_points`**: Filter out objects smaller than a certain area.
*   **Object Properties**: Calculate object area and centroid directly on the sphere.

### 3. Spherical Subgrid Refinement
This is the most critical component for parity.
*   **Current implementation**: Fits a 2D quadratic surface to a regular 3x3 mesh.
*   **HEALPix implementation**: Use **unconstrained least-squares surface fitting**.
    *   For a detected local extremum at pixel $P$, gather all immediate neighbors (and potentially 2nd-hop neighbors).
    *   Map pixel indices to Cartesian $(x, y, z)$ on the unit sphere.
    *   Fit a local quadratic surface $f(x, y, z)$ to these irregular points.
    *   Solve for the precise maximum/minimum coordinate on the spherical surface.

### 4. Default Configuration Consistency
Ensure all trackers adhere to the standard meteorological bandpass defaults:
*   **Default Filtering**: $L_{min}=5, L_{max}=42$ for all trackers (`Simple`, `Hodges`, `Healpix`).
*   **API**: Ensure `HealpixTracker.track()` exposes the same `filter`, `lmin`, and `lmax` arguments as the 2D trackers.

## Implementation Steps
1.  **Kernels**: Add `_numba_healpix_ccl` and `_numba_healpix_subgrid_refine` to `src/pystormtracker/healpix/kernels.py`.
2.  **Detector**: Update `HealpixDetector.detect()` to call the object-based pipeline: `CCL -> Filter -> Extrema -> Refine`.
3.  **Validation**: Verify that coordinates produced on a high-resolution HEALPix grid are sub-pixel accurate and match `HodgesTracker` results when both are run on high-density regular grids.
