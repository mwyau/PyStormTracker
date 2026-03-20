# PyStormTracker Roadmap

This document outlines the strategic plan for improving PyStormTracker's performance, CI/CD pipelines, and overall architecture, with a focus on high-resolution climate data scalability.

## 1. Performance & Scalability

*   **Fix OOM Risks (Lazy Loading):**
    *   *Current State:* The `SimpleDetector` eagerly loads entire assigned time slices into memory (`full_var = self.get_var()`).
    *   *Action:* Refactor `get_var()` to yield frames lazily. Iterate over the xarray dataset loading only the current frame needed to compute extrema, preventing Out-Of-Memory errors on native-resolution (e.g., 0.25°) datasets.
*   **Prevent CPU Oversubscription (Numba vs. Dask/MPI):**
    *   *Current State:* Dask/MPI orchestrates processes, but Numba kernels lack explicit thread constraints. If `parallel=True` is used in Numba, it will oversubscribe CPU cores and cause thrashing.
    *   *Action:* Explicitly control thread topology inside worker tasks (e.g., `numba.set_num_threads(1)` when scaling via Dask/MPI processes).
*   **Vectorize the `SimpleLinker`:**
    *   *Current State:* Linking is currently sequential, resulting in an $O(N^2)$ bottleneck as trajectory counts scale.
    *   *Action:* Leverage `scipy.spatial.cKDTree` for nearest-neighbor lookups across time steps to convert spatial proximity searches to highly optimized C-level trees.
*   **Array-Backed Data Model (Completed):** 
    *   Transitioned from nested Python objects to flat, C-contiguous NumPy arrays for trajectories and centers.
    *   Zero-copy serialization for high-performance parallel execution.
*   **JIT-Optimized Kernels (Completed):** 
    *   Implemented core mathematical filters (Laplacian, Extrema, MGE, CCL) in GIL-free Numba JIT.

## 2. CI/CD & Testing

*   **Implement Performance Regression Testing:**
    *   *Current State:* No automated guardrails against JIT performance degradation.
    *   *Action:* Integrate `pytest-benchmark` with a deterministic synthetic dataset fixture. Add a CI job that fails if Numba execution time drops significantly compared to `main`.
*   **Dependency Audit:**
    *   *Action:* Add a weekly scheduled CI run of `uv sync --resolution lowest-direct` combined with `pytest` to ensure minimum versions in `pyproject.toml` remain accurate.
*   **Tiered Integration Testing (Completed):** 
    *   Implemented "Short" vs "Full" integration test suites to balance local dev speed with CI thoroughness.

## 3. Architecture

*   **Idiomatic Xarray Integration (`apply_ufunc`):**
    *   *Current State:* Xarray is primarily used as an I/O loader before dropping down to manual NumPy arrays, manual chunking, and custom MPI/Dask orchestration.
    *   *Action:* Wrap the `_numba_extrema_filter` inside `xr.apply_ufunc(..., dask="parallelized")`. This allows Xarray to natively handle chunking, distributed execution, and reassembly, potentially eliminating the need for manual orchestration code in `concurrent.py`.
*   **Distributed Backends (Completed):** 
    *   Native support for Dask (local scaling) and MPI (distributed scaling).
    *   Perfect bit-wise identity between Serial and Parallel results.
*   **Modern CLI & API (Completed):** 
    *   Grouped, logical command-line interface.
    *   Flexible `Tracker` Protocol for cross-algorithm support.

## 4. Distribution & Ecosystem

*   **Conda-forge Distribution:**
    *   *Current State:* PyStormTracker is available on PyPI, Docker Hub, and GHCR.
    *   *Action:* Create a feedstock for `conda-forge` to enable installation via `conda` or `mamba`, improving accessibility for the scientific community.

## 5. Feature Implementation

*   **HodgesTracker Integration (Completed):** 
    *   Native Python/Numba implementation of the Modified Greedy Exchange (MGE) algorithm.
    *   Spherical geodesic cost functions and adaptive constraints ($d_{max}$, $\psi_{max}$).
    *   Object-based detection pipeline (Threshold -> CCL -> Filter -> Extrema).
    *   Unified legacy standards (constants.py).
*   **Preprocessing (Completed):** 
    *   Added support for spectral filtering (e.g., T42/T63 truncation) using `pyshtools` and `SphericalHarmonicFilter`.
    *   Added `TaperFilter` for boundary smoothing.
*   **HodgesTracker Refinement (In Progress):** 
    *   *Action:* Implement Dierckx B-spline surface fitting and evaluation in Numba to achieve bit-wise coordinate identity with the original TRACK software.
*   **Postprocessing (Track Metrics):**
    *   *Action:* Implement Accumulated Track Activity (ATA) and other storm track metrics from **Yau and Chang (2020)** to provide physically meaningful analysis of detected trajectories.
