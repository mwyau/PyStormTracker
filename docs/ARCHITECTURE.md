# PyStormTracker Architecture

This document describes the vectorized architecture of PyStormTracker, detailing how it leverages vectorization and decoupled components to process massive climate datasets efficiently.

## 1. High-Level Design Philosophy

PyStormTracker is built for scale and extensibility. The architecture is centered around three core principles:
1.  **Unified API (Tracker Protocol):** A structural interface that allows the CLI and Python API to support multiple tracking algorithms (e.g., `SimpleTracker`, `HodgesTracker`) interchangeably.
2.  **Centralized Threshold Management:** The `SimpleDetector` is responsible for managing variable-specific detection thresholds (e.g., `1e-4` for vorticity), ensuring consistent behavior across different parallel backends.
3.  **Vectorization & JIT:** Heavy mathematical operations are offloaded to **Numba** JIT-compiled kernels and **NumPy** broadcasting, bypassing Python's loop overhead and Global Interpreter Lock (GIL).
4.  **Hybrid Parallelism:** The architecture parallelizes the computationally intensive **Detection** phase while centralizing the **Linking** phase to ensure exact serial-parallel consistency.

---

## 2. Core Components

### 2.1 Array-Backed Data Models (`Tracks`, `Track`, `Center`)
The data models utilize a contiguous memory paradigm:
*   **`Tracks`**: The central container holding contiguous 1D NumPy arrays for `track_ids`, `times`, `lats`, `lons`, and a dictionary of scientific variables.
*   **`Track`**: A lightweight "view" into the `Tracks` arrays for a specific ID.
*   **`Center`**: A simple dataclass used strictly for iteration or final data export.

**Benefits:** By avoiding the creation of many Python objects, memory usage is minimized, and data serialization between parallel processes is efficient. Raw NumPy arrays also enable efficient distance calculations via C-level broadcasting.

### 2.2 Shared DataLoader
Data loading is encapsulated in a dedicated `DataLoader` class (`io/loader.py`). This component handles:
*   **Format Abstraction**: Seamlessly detects and opens NetCDF (via `h5netcdf` or `netcdf4`) and GRIB (via `cfgrib`) files.
*   **Variable Mapping**: Automatically maps common variable aliases (e.g., `msl`/`slp`, `vo`/`rv`) and coordinate names (`latitude`/`lat`), allowing the same tracking logic to work across different data providers.
*   **Contiguous I/O**: Performs single-block contiguous reads from disk, bypassing HDF5 lock contention.

### 2.3 Heuristic Tracking Implementation (SimpleTracker)
Trajectory construction in the heuristic tracker uses NumPy broadcasting to calculate Haversine distance matrices between existing track tails and new storm centers. By sorting points spatially before matching, the Linker ensures deterministic, greedy nearest-neighbor linking.

### 2.4 Optimization-Based Tracking Implementation (HodgesTracker)
The `HodgesTracker` implements the industry-standard TRACK algorithm (Hodges 1994, 1995, 1999). Unlike the heuristic tracker, it uses a global optimization approach:
*   **Object-Based Detection**: Features are identified using a multi-stage pipeline: `Thresholding -> Connected Component Labeling (CCL) -> Object Filtering -> Local Extrema`. This ensures only significant meteorological features are tracked.
*   **Modified Greedy Exchange (MGE)**: An iterative algorithm that swaps points between tracks to minimize a total cost function.
*   **Spherical Cost Function**: A mathematical model that penalizes changes in track direction (directional smoothness) and speed.
*   **Adaptive Constraints**: Dynamically adjusts search radii ($d_{max}$) and smoothness limits ($\psi_{max}$) based on regional zones and track velocity.
*   **Sub-grid Refinement**: Fits a local quadratic surface to each extremum to identify feature centers with precision higher than the grid resolution.

### 2.5 Parallel Pipeline (Gather-then-Link)

To ensure that parallel results are bit-wise identical to serial runs, PyStormTracker uses a hybrid parallel strategy:
1.  **Parallel Detection**: Assigned time chunks are distributed across Dask or MPI workers. Each worker runs Numba kernels to find centers and returns raw coordinate arrays.
2.  **Centralized Linking**: The main process gathers the raw detections from all workers and performs a single sequential link. 

**Why this works:** In storm tracking, the **Detection** phase (finding local extrema in 3D grids) consumes >95% of the runtime. The **Linking** phase (connecting coordinate lists) is efficient once vectorized. Centralizing the link eliminates the complex "merging" bugs found in tree-reduction strategies while maintaining scaling.

---

## 3. The `Tracker` Protocol

The `Tracker` Protocol (defined in `src/pystormtracker/models/tracker.py`) provides a standardized interface for all tracking algorithms:

```python
import pystormtracker as pst

# Instantiate any compliant tracker
tracker = pst.SimpleTracker()

# Standardized .track() method
tracks = tracker.track(
    infile="era5_msl.nc", 
    varname="msl", 
    start_time="2025-01-01",
    backend="dask"
)

# Standardized export
tracks.write("output.txt", format="imilast")
```

---

## 4. Future Architectural Direction

To further optimize scalability and memory efficiency for native-resolution climate datasets (e.g., 0.25° ERA5), the architecture is evolving towards deeper integration with the scientific Python ecosystem:

*   **Idiomatic Xarray (`apply_ufunc`):** Transitioning away from custom MPI/Dask chunking in favor of Xarray's native `apply_ufunc(..., dask="parallelized")`. This delegates chunk management and distributed execution entirely to Xarray/Dask, reducing custom orchestration code.
*   **Lazy Evaluation & Thread Topology:** Shifting from eager chunk-loading to lazy, frame-by-frame memory access to eliminate out-of-memory risks on large domains. Concurrently, strictly pinning Numba thread topologies to prevent CPU oversubscription in multi-process backends.
*   **Tree-based Linking:** Upgrading the current NumPy-broadcasting linker to utilize C-level tree structures (e.g., `scipy.spatial.cKDTree`), breaking the $O(N^2)$ scaling barrier for extremely long or dense trajectory sequences.

For more details on specific planned implementations, see the [Roadmap](ROADMAP.md).

---

## 5. Performance Benchmarks

To quantify the efficiency gains of the array-backed JIT architecture, a comprehensive performance comparison was conducted between the legacy object-oriented system (`v0.3.3`) and the current implementation.

Detailed execution timings (breaking down Detection, Linking, Export, and I/O Overhead) across Serial, Dask, and MPI backends for both standard and high-resolution ERA5 datasets are available in the [Benchmark Report](BENCHMARK.md).

---

## Appendix: Evolution from Legacy Architecture

The current architecture represents a fundamental shift from the legacy nested-object design used in earlier versions.

| Feature | Legacy Architecture (v0.3.x and earlier) | Current Architecture (v0.4.0+) |
| :--- | :--- | :--- |
| **Data Storage** | Nested lists of `Center` and `Track` objects. | Flat, C-contiguous NumPy arrays. |
| **Parallelism** | Threads (bottlenecked by GIL). | Processes/MPI (concurrent I/O). |
| **Linking Strategy** | Tree-reduction (prone to boundary splits). | Parallel Detect + Centralized Link (serial consistency). |
| **Linker** | $O(N^2)$ nested Python loops. | Vectorized NumPy matrix broadcasting. |
| **Algorithms** | Simple heuristic only. | Dual: Simple + Hodges (TRACK) Parity. |
| **I/O** | Many small lazy-loaded chunks. | Contiguous shared `DataLoader`. |
