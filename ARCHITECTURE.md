# PyStormTracker Architecture

This document describes the modern, high-performance architecture of PyStormTracker, detailing how it leverages vectorization and array-backed models to process massive climate datasets efficiently.

## 1. High-Level Design Philosophy

PyStormTracker is built for scale and extensibility. The architecture is centered around three core principles:
1.  **Unified API (Tracker Protocol):** A structural interface that allows the CLI and Python API to support multiple tracking algorithms (e.g., `SimpleTracker`, `HodgesTracker`) interchangeably.
2.  **Vectorization & JIT:** Heavy mathematical operations are offloaded to **Numba** JIT-compiled kernels and **NumPy** broadcasting, bypassing Python's loop overhead and Global Interpreter Lock (GIL).
3.  **Zero-Copy Parallelism:** Core data structures are maintained as flat NumPy arrays, enabling rapid serialization across Dask processes and MPI workers with near-zero overhead.

---

## 2. Modern Core Components

### 2.1 Array-Backed Data Models (`Tracks`, `Track`, `Center`)
The data models are the foundation of the library's performance. 
*   **`Tracks`**: The central container holding contiguous 1D NumPy arrays for `track_ids`, `times`, `lats`, `lons`, and a dictionary of scientific variables.
*   **`Track`**: A lightweight "view" into the `Tracks` arrays for a specific ID.
*   **`Center`**: A simple dataclass used strictly for iteration or final data export.

**Benefits:** By avoiding the creation of millions of Python objects, memory usage is minimized, and parallel data transfer (via Pickle) is transformed from a major bottleneck into a trivial operation.

### 2.2 The Detector (`SimpleDetector` & Numba Kernels)
Data loading and feature detection are decoupled:
*   **DataLoader**: Uses Xarray to perform single-block contiguous reads from NetCDF or GRIB files, bypassing the HDF5 global lock contention common in threaded environments.
*   **Kernels**: All spatial filters are implemented in Numba with `nogil=True` and `cache=True`, ensuring they run at raw C speeds and remain compiled on disk for future runs.

### 2.3 Vectorized Linker (`SimpleLinker`)
Trajectory construction uses NumPy broadcasting to calculate Haversine distance matrices between track tails and new storm centers. This replaces nested Python `for` loops with optimized C-level matrix operations, reducing linking time from seconds to milliseconds.

### 2.4 Parallel Pipeline & Tree Reduction
PyStormTracker uses a binary tree reduction strategy for distributed execution:
1.  **Local Phase**: Workers independently detect and link centers within their assigned time chunks, producing local `Tracks` objects.
2.  **Reduction Phase**: Adjacent `Tracks` objects are merged by matching the "tails" of one chunk to the "heads" of the next, using the vectorized linker to maintain global track identity.

---

## 3. The `Tracker` Protocol

The `Tracker` Protocol (defined in `src/pystormtracker/models/tracker.py`) provides a standardized interface for all tracking algorithms:

```python
import pystormtracker as pst

# Instantiate any compliant tracker
tracker = pst.SimpleTracker()

# Standardized .track() method handles dates, backends, and engines
tracks = tracker.track(
    infile="era5_msl.nc", 
    varname="msl", 
    start_time="2025-01-01",
    backend="dask"
)
```

This abstraction allows for the future addition of complex algorithms like the **Hodges Tracker** without altering the CLI or user-facing API.

---

## Appendix: Evolution from Legacy Architecture

The current architecture represents a fundamental shift from the legacy nested-object design used in earlier versions.

| Feature | Legacy Architecture (v0.3.x and earlier) | Modern Architecture (v0.4.0+) |
| :--- | :--- | :--- |
| **Data Storage** | Nested lists of `Center` and `Track` objects. | Flat, C-contiguous NumPy arrays. |
| **Parallelism** | Threads (bottlenecked by GIL and HDF5 lock). | Processes/MPI (true concurrent I/O). |
| **Serialization** | Slow `pickle` traversal of object trees. | Instant zero-copy array buffer transfer. |
| **Linker** | $O(N^2)$ nested Python loops. | Vectorized NumPy matrix broadcasting. |
| **I/O** | Many small lazy-loaded chunks. | Single-block contiguous reads. |

The modern architecture was necessitated by the shift toward high-resolution (0.25°) datasets, where the overhead of the legacy object-oriented approach made global tracking across multiple decades computationally infeasible.
