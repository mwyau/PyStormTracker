# PyStormTracker Roadmap

This document outlines the strategic plan for improving PyStormTracker's performance, CI/CD pipelines, and overall architecture, with a focus on high-resolution climate data scalability.

## 1. Performance & Scalability

*   **Prevent CPU Oversubscription (Numba vs. Dask/MPI):**
    *   *Current State:* Dask/MPI orchestrates processes, but Numba kernels lack explicit thread constraints. If `parallel=True` is used in Numba, it will oversubscribe CPU cores and cause thrashing.
    *   *Action:* Explicitly control thread topology inside worker tasks (e.g., `numba.set_num_threads(1)` when scaling via Dask/MPI processes).
*   **Vectorize the `SimpleLinker`:**
    *   *Current State:* Linking uses a vectorized Haversine matrix but remains $O(N \times M)$, which can be a bottleneck as trajectory counts scale.
    *   *Action:* Leverage `scipy.spatial.cKDTree` for nearest-neighbor lookups across time steps to convert spatial proximity searches to highly optimized C-level trees.
*   **Manage Memory Pressure (Chunking) (Completed):** 
    *   Implemented time-chunking across backends to prevent memory exhaustion on large datasets. This maintains optimal block-IO performance by avoiding metadata/locking overhead.
*   **Array-Backed Data Model (Completed):** 
    *   Transitioned from nested Python objects to flat, C-contiguous NumPy arrays for trajectories and centers.
*   **JIT-Optimized Kernels (Completed):** 
    *   Implemented core mathematical filters (Laplacian, Extrema, MGE, CCL) in GIL-free Numba JIT.
*   **GPU-Accelerated Preprocessing & Detection (Experimental):**
    *   *Action:* Expand JAX-native capabilities beyond spherical harmonic transforms and kinematic derivatives to include local extrema detection and Laplacian filtering. This will enable full end-to-end GPU/TPU acceleration for high-resolution datasets.
    *   *Status:* JAX-based spectral filtering and vector derivatives have been implemented as an experimental backend.

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
    *   *Current State:* Xarray is primarily used as an I/O loader before dropping down to NumPy arrays and manual parallel orchestration.
    *   *Action:* Wrap core Numba filters inside `xr.apply_ufunc(..., dask="parallelized")`. This allows Xarray to natively handle chunking and distributed execution.
*   **Distributed Backends (Completed):** 
    *   Native support for Dask and MPI backends with **automatic environment detection** and fallback logic.
*   **Modern CLI & API (Completed):** 
    *   Grouped, logical command-line interface with auto-configuration of parallel workers.
    *   Flexible `Tracker` Protocol for cross-algorithm support.
*   **Remote Data Support (Completed):**
    *   Native support for remote Zarr datasets via HTTP, S3, and GS protocols with automatic format detection.

## 4. Distribution & Ecosystem

*   **Modular Dependencies (Completed):**
    *   Optional dependency groups (e.g., `[hodges]`, `[mpi]`, `[grib]`) to minimize build-time requirements and simplify installation in constrained environments like ReadTheDocs.
*   **Conda-forge Distribution (Completed):**
    *   Available on `conda-forge` for easy cross-platform installation.

## 5. Feature Implementation

*   **HodgesTracker Integration (Completed):** 
    *   Native Python/Numba implementation of the Modified Greedy Exchange (MGE) algorithm with algorithmic parity to TRACK-1.5.2.
*   **Preprocessing (Completed):** 
*   **HodgesTracker Refinement (In Progress):** 
    *   *Action:* Implement Dierckx B-spline surface fitting and evaluation in Numba to achieve bit-wise coordinate identity with original TRACK software.
*   **Postprocessing (Track Metrics):**
    *   *Action:* Implement Accumulated Track Activity (ATA) and other storm track metrics from **Yau and Chang (2020)**.
*   **JAX-Based Feature Detection (Proposed):**
    *   *Action:* Develop JAX-native implementations of the extrema detection and intensity refinement kernels to support high-throughput, GPU-resident tracking pipelines.
