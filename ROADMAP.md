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

## 2. CI/CD & Testing

*   **Implement Performance Regression Testing:**
    *   *Current State:* No automated guardrails against JIT performance degradation.
    *   *Action:* Integrate `pytest-benchmark` with a deterministic synthetic dataset fixture. Add a CI job that fails if Numba execution time drops significantly compared to `main`.
*   **Dependency Audit:**
    *   *Action:* Add a weekly scheduled CI run of `uv sync --resolution lowest-direct` combined with `pytest` to ensure minimum versions in `pyproject.toml` remain accurate.

## 3. Architecture

*   **Idiomatic Xarray Integration (`apply_ufunc`):**
    *   *Current State:* Xarray is primarily used as an I/O loader before dropping down to manual NumPy arrays, manual chunking, and custom MPI/Dask orchestration.
    *   *Action:* Wrap the `_numba_extrema_filter` inside `xr.apply_ufunc(..., dask="parallelized")`. This allows Xarray to natively handle chunking, distributed execution, and reassembly, potentially eliminating the need for manual orchestration code in `concurrent.py`.

## 4. Distribution & Ecosystem

*   **Conda-forge Distribution:**
    *   *Current State:* PyStormTracker is available on PyPI, Docker Hub, and GHCR.
    *   *Action:* Create a feedstock for `conda-forge` to enable installation via `conda` or `mamba`, improving accessibility for the scientific community.

## 5. Feature Implementation

*   **Complete `HodgesTracker`:**
    *   *Current State:* Scaffolding added.
    *   *Action:* Integrate spherical distance math (e.g., `haversine` or Numba-compiled Vincenty formula). For track optimization (cost function minimization), port the original C logic using `scipy.optimize.linear_sum_assignment` which perfectly maps to the tracking assignment problem.
