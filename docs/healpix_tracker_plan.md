# Implementation Plan: Spectral Regridding & HEALPix Tracker

## Background & Motivation
PyStormTracker currently operates exclusively on 2D regular grids (Clenshaw-Curtis or Gauss-Legendre). To support native 1D grids like HEALPix and perform highly accurate spectral transformations, a new architecture is required. This initiative introduces a mathematically precise spectral regridding module using `ducc0` and a new, graph-based tracking algorithm designed specifically for 1D HEALPix grids.

## Scope & Impact
This update consists of two major, decoupled components:
1.  **`SpectralRegridder`**: A preprocessing module utilizing `ducc0` to spectrally transform data between CC, GL, and HEALPix grids.
2.  **`HealpixTracker`**: A new tracking algorithm implementing the `Tracker` protocol, capable of detecting extrema on a 1D graph topology using precomputed adjacency lists.

## Architecture

### Phase 1: `SpectralRegridder` Module
Create `src/pystormtracker/preprocessing/regrid.py`.

*   **Capabilities**:
    *   **Inputs**: `CC` (Clenshaw-Curtis, equidistant) and `GL` (Gauss-Legendre).
    *   **Outputs**: `CC`, `GL`, and `HEALPix`.
*   **Implementation**:
    *   Use `ducc0.sht.analysis_2d` to extract spherical harmonic coefficients ($a_{lm}$) from the 2D input.
    *   Use `ducc0.sht.synthesis_2d` to project onto a new 2D target grid (`CC` or `GL`).
    *   Use `ducc0.sht.synthesis` with `geometry="healpix"` to project directly into a 1D pixel array.
    *   Handle Xarray metadata and dimension reshaping (from `(time, lat, lon)` to `(time, cell)` for HEALPix).

### Phase 2: `HealpixTracker` Module
Create a new package at `src/pystormtracker/healpix/`.

*   **`kernels.py`**:
    *   Implement `_numba_healpix_extrema(data: 1D, neighbor_table: 2D, is_min: bool)`.
    *   The kernel will use the precomputed `neighbor_table` to perform graph-traversal neighborhood checks instead of spatial 2D array slicing.
*   **`detector.py`**:
    *   Implement `HealpixDetector`.
    *   Handle 1D Xarray ingestion.
    *   Compute $N_{side}$ and generate the `neighbor_table` using `ducc0.healpix.Healpix_Base` during initialization (No `healpy` needed).
    *   Map detected 1D pixel indices back to standard `(lat, lon)` using `ducc0.healpix.Healpix_Base.pix2ang`.
*   **`tracker.py`**:
    *   Implement `HealpixTracker` adhering to the `Tracker` protocol.
    *   Reuse the existing MPI/Dask temporal splitting architecture.
    *   Reuse the existing `SimpleLinker` to connect the resulting `Center` objects.

## Dependency Management: Zero New Dependencies
*   We **DO NOT need `healpy`**.
*   Research confirms that `ducc0` has a built-in `ducc0.healpix.Healpix_Base` class. 
*   We will use `Healpix_Base(nside, 'RING').neighbors(pixel_idx)` to generate the lookup table, and `.pix2ang(pixel_idx)` for coordinate resolution. 
*   This keeps the project dependencies lean and utilizes the high-performance C++ backend we already depend on for spectral transforms.

## Verification & Testing
*   **Regridder Tests:** 
    *   Assert `CC -> CC` transformation with identical resolution yields near-zero residuals.
    *   Assert `CC -> HEALPix` outputs the correct 1D shape ($12 \times N_{side}^2$).
*   **Tracker Tests:**
    *   Create a synthetic HEALPix dataset with known extrema locations.
    *   Verify `HealpixTracker` correctly identifies these extrema.
    *   Verify parallel execution (Dask/MPI) yields bit-wise identical tracks to serial execution.

## Migration & Rollback
*   No new dependencies are required. `ducc0` is already an active dependency for SHT parity.
*   If issues arise, users can continue using `SimpleTracker` and `HodgesTracker` as they are isolated in their respective modules. No core API signatures are changing.
