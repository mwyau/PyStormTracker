# PyStormTracker Roadmap

This document records completed, partial, and planned engineering work. `TODO.md` tracks feature-level verification in more detail.

## 1. Performance & Scalability

*   **Prevent CPU Oversubscription (Planned):**
    *   *Current State:* Simple Dask uses threads and Simple MPI uses processes. ducc0 thread counts are limited in distributed preprocessing paths, but thread policy is not centralized.
    *   *Action:* Define and test thread limits for ducc0, Numba, Dask, and MPI execution.
*   **Vectorize the `SimpleLinker`:**
    *   *Current State:* Linking uses a vectorized great-circle distance matrix but remains $O(N \times M)$ as trajectory counts increase.
    *   *Action:* Evaluate `scipy.spatial.cKDTree` or another spherical candidate index while preserving deterministic matches.
*   **Manage Memory Pressure (Chunking) (Partial):**
    *   Simple Dask partitions detection tasks. Hodges can partition serial detection and gathers all detections before linking. Preprocessing still performs eager loads in several paths.
*   **Array-Backed Data Model (Completed):** 
    *   Transitioned from nested Python objects to flat, C-contiguous NumPy arrays for trajectories and centers.
*   **JIT-Optimized Numba Kernels (Completed):**
    *   Implemented core mathematical filters (Laplacian, Extrema, MGE, CCL) in GIL-free Numba JIT.
*   **GPU Preprocessing and Detection (Proposed):**
    *   No JAX backend is present in the current code or dependencies. A future implementation would require numerical reference tests and a separate dependency group.

## 2. CI/CD & Testing

*   **Implement Performance Regression Testing:**
    *   *Current State:* Historical benchmark tables exist, but CI does not enforce performance limits.
    *   *Action:* Add a deterministic benchmark fixture and define statistically stable comparison criteria before making CI fail on timing changes.
*   **Restore SHTns Spherical Harmonic Transform Benchmark:**
    *   *Action:* Add a standalone script that compares the SHTns and ducc0 SHT engines on identical scalar-filter and kinematic-derivative reference fields. Keep SHTns outside production dependencies and record grid geometry, truncation, normalization, compiler, library versions, thread count, and hardware with the results.
*   **Dependency Audit:**
    *   *Action:* Add a weekly scheduled CI run of `uv sync --resolution lowest-direct` combined with `pytest` to ensure minimum versions in `pyproject.toml` remain accurate.
*   **Tiered Integration Testing (Completed):** 
    *   Implemented "Short" vs "Full" integration test suites to balance local dev speed with CI thoroughness.

## 3. Architecture

*   **Xarray Generalized Ufunc Integration (`apply_ufunc`) (Partial):**
    *   *Current State:* Spectral filtering and kinematics use `xr.apply_ufunc`; feature detection uses explicit NumPy/Numba orchestration.
    *   *Action:* Evaluate `apply_ufunc` for detection without changing serial/Dask/MPI results.
*   **Distributed Backends (Partial):**
    *   Simple supports serial, Dask, and MPI Gather-then-Link execution. Hodges and HEALPix are serial-only and reject unsupported backends.
*   **CLI and API (Completed):**
    *   The CLI groups tracking, sampling, comparison, and conversion commands.
    *   Flexible `Tracker` Protocol for cross-algorithm support.
*   **Remote Data Support (Completed):**
    *   Remote Zarr datasets can be opened over HTTP, S3, and GS with format detection.

## 4. Distribution & Ecosystem

*   **Modular Dependencies (Completed):**
    *   Optional groups cover MPI, GRIB, NetCDF4, Zarr, metrics, documentation, and visualization. Hodges dependencies are part of the core package.
*   **Conda-forge Distribution (Completed):**
    *   Available on `conda-forge`.

## 5. Feature Implementation

*   **HodgesTracker Integration (Partial):**
    *   MGE, adaptive constraints, object detection, and properties are implemented in Python/Numba. Direct end-to-end comparison with TRACK-1.5.2 remains.
*   **Preprocessing (Partial):**
    *   Spherical harmonic transform (SHT), discrete cosine transform (DCT), Sardeshmukh-Hoskins spectral tapering, polar stereographic, HEALPix, full-Gaussian, and reduced-Gaussian paths are implemented. Real-N320 end-to-end validation remains optional and has not yet been completed.
*   **HodgesTracker Refinement (Partial):**
    *   Quadratic centers and spherical-spline values are implemented. Direct B-spline center optimization and TRACK coordinate comparison remain.
*   **Regional Model Support (Partial):**
    *   DCT filtering and nonperiodic boundary behavior are implemented and tested for execution. Numerical comparison with the Denis et al. (2002) limited-area DCT method remains.
*   **Spectral Tapering (Completed):**
    *   Harmonic coefficient tapering and a separate spatial boundary taper are implemented.
*   **Spherical Kernel Gridding (Completed):**
    *   Constant-radius, Fisher exponential, and Cressman rational weighting kernels are implemented with Numba for track-frequency and track-density grids.
*   **Statistical Distributions and Confidence Intervals (Planned):**
    *   Define statistical methods and reference cases separately from kernel gridding.
*   **Object Size and Morphological Properties (Completed):**
    *   Calculates feature size in km² and intensity-weighted ellipse parameters.
*   **Vorticity Tracking and Variable Support (Partial):**
    *   Existing vorticity fields can be tracked, kinematics are available in Python, and `sample` attaches secondary variables. A CLI command that derives vorticity from wind remains planned.
*   **Postprocessing (Track Metrics) (Completed):**
    *   Implemented Accumulated Track Activity (ATA), ACA, and density metrics from **Yau and Chang (2020)** with monthly aggregation and Xarray-based cross-validation (CCA/PCA).
*   **JAX-Based Feature Detection (Proposed):**
    *   Evaluate only after reference behavior and optional dependency boundaries are defined.
