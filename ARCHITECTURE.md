# PyStormTracker Architecture

This document describes the modern, high-performance architecture of PyStormTracker, specifically detailing the transition from the legacy nested-object approach to the current vectorized, array-backed paradigm.

## 1. High-Level Design Philosophy

PyStormTracker is designed to analyze massive meteorological datasets (e.g., decades of ERA5 reanalysis data at 0.25° resolution) to detect and link storm centers into continuous tracks.

The architecture is built upon three core pillars:
1.  **Tracker Protocols:** The codebase is designed to support multiple completely different tracking algorithms (e.g., `SimpleTracker`, `HodgesTracker`) under a single, unified CLI and API.
2.  **Vectorization First:** The heavy lifting is done in C-speed environments via NumPy broadcasting and Numba JIT compilation, keeping the Python layer strictly for orchestration.
3.  **Zero-Copy Parallelism:** The data models are structured as flat, contiguous NumPy arrays, allowing Dask processes and MPI workers to serialize and transmit millions of data points with near-zero pickling overhead.

---

## 2. Comparing the Old vs. New Architecture

### 2.1 Data Models (`Tracks`, `Track`, `Center`)

**The Old Architecture:**
Previously, the data was modeled using a deeply nested structure of custom Python objects:
*   A `Tracks` object contained a `list[Track]`.
*   A `Track` object contained a `list[Center]`.
*   A `Center` was a dataclass holding individual `float` and `datetime` values.

*Why it bottlenecked:* Python is terrible at managing millions of tiny objects. Creating millions of `Center` instances invoked massive memory overhead and garbage collection pauses. More importantly, when running in parallel (Dask Processes or MPI), passing these objects between processes required the `pickle` module to traverse the entire nested tree, serialize every object, transmit it, and deserialize it on the other side. This serialization overhead often took longer than the actual scientific computation, resulting in a 1.0x (or worse) parallel speedup.

**The New Architecture:**
The models have been rewritten to be **Array-Backed**. The concept of a single "point" or "list" is now merely a logical view over contiguous memory.
*   The `Tracks` class is the sole owner of the data. Internally, it holds exactly five 1D NumPy arrays: `track_ids`, `times`, `lats`, `lons`, and `vars`.
*   A `Track` object no longer holds a list of centers. It is instantiated with a `track_id` and a reference to the parent `Tracks` object. It simply yields a "view" of the arrays where the ID matches.
*   The `Center` class still exists strictly for backward compatibility and simple iteration, but it is entirely bypassed during heavy computation.

*Why it's fast:* NumPy arrays are blocks of C-contiguous memory. When Dask or MPI needs to send a worker's results to the main node, it simply drops the raw memory buffer into the socket. There is zero object traversal.

### 2.2 The Detector (`SimpleDetector` & Numba Kernels)

**The Old Architecture:**
The detector relied on Xarray combined with Dask's `chunks={}` to lazy-load chunks of NetCDF data. Inside the computation loop, it created new Python `Center` objects one-by-one and appended them to a list. Furthermore, the underlying mathematical filters (Numba) were not configured to cache or run outside the GIL.

*Why it bottlenecked:* 
1.  Xarray's HDF5/NetCDF4 backend relies on `h5py`, which requires a strict global lock (`HDF5_LOCK`). When Dask threads tried to read chunks simultaneously, they were forced into a single-file line, defeating the purpose of threading.
2.  Creating Python objects inside the hot detection loop choked the system.

**The New Architecture:**
The detector and the math have been fundamentally separated.
*   **The DataLoader (`SimpleDetector`):** Drops the Dask `chunks={}` argument. Instead, when a worker process is assigned a time slice (e.g., steps 100 to 200), the worker uses Xarray to slice that specific chunk and read it *entirely into memory at once* as a native NumPy array. This guarantees a single, fast contiguous disk read and bypasses the HDF5 lock contention.
*   **The Math (`kernels.py`):** All mathematical logic (extrema finding, laplacian filters, duplicate removal) has been extracted into a pure Numba module. The decorators now use `nogil=True` and `cache=True`, meaning they execute at raw C speeds and don't recompile on every run.
*   **Flat Outputs:** `detect_raw()` no longer creates `Center` objects. It simply returns a tuple of 1D NumPy arrays containing the raw coordinates of every detected storm in that time chunk.

### 2.3 The Linker (`SimpleLinker`)

**The Old Architecture:**
To connect detected storms from Time $T$ to existing tracks from Time $T-1$, the linker used a nested `for` loop, calculating the Haversine distance between every single Python `Center` object using the `math` module.

*Why it bottlenecked:* Calculating distances point-by-point in pure Python scales at $O(M \times N)$ and is incredibly slow when processing thousands of active storm centers.

**The New Architecture:**
The Linker is now completely **Vectorized**.
Instead of using `for` loops, it extracts the `tail` coordinates from the `Tracks` array and the `new` coordinates from the detector. It passes both into `haversine_matrix()`, which uses NumPy Broadcasting to instantly calculate the full $M \times N$ distance matrix in C. Finding the mutually closest points is reduced to highly optimized `np.argmin()` calls along the matrix axes.

### 2.4 Parallel Orchestration and Tree Reduction

**The Old Architecture:**
In the previous setup, Dask parallelism was attempted at the lowest possible level: wrapping individual `detect()` calls in `dask.delayed` and having the main thread sequentially link the millions of returned `Center` objects.

**The New Architecture:**
The orchestration (located in `src/pystormtracker/simple/tracker.py`) now treats workers as independent, intelligent nodes.
1.  **Detect & Link Locally:** A worker process receives its time slice. It reads the data, runs the Numba kernels, and immediately uses its own local Vectorized Linker to connect the centers. It returns a fully formed, lightweight, array-backed `Tracks` object.
2.  **Tree Reduction:** Whether using MPI or Dask Processes, the architecture utilizes a Binary Tree Reduction. If Worker 1 finishes Block A and Worker 2 finishes Block B, they don't send their data to the main thread. Instead, a reduction task takes both `Tracks` objects, instantly matches the tails of A to the heads of B using the Vectorized Linker, concatenates the underlying NumPy arrays via `np.concatenate`, and passes the merged object up the tree.

---

## 3. Extensibility: The `Tracker` Protocol

The ultimate goal of this refactor is to prepare the codebase for future algorithms, such as the `HodgesTracker`.

To achieve this, the concept of a rigid `Grid` interface has been deprecated in favor of a top-level `Tracker` Protocol (located in `src/pystormtracker/models/tracker.py`). 

In this paradigm, the CLI does not dictate how data is loaded or split. The CLI simply instantiates a concrete tracker (e.g., `SimpleTracker`) and calls `.track(infile, varname)`. The `SimpleTracker` encapsulates its own specific logic for NetCDF reading, chunking, and worker dispatch. When a new tracker is introduced, it will implement the same `.track()` interface but is entirely free to handle data loading (e.g., reading spectral harmonics instead of 2D frames) in whatever way is most optimal for its mathematics.

## 4. Summary of Improvements
- **Zero Object Pickling:** Dask/MPI serialize flat arrays instantly.
- **I/O Optimization:** Single-block contiguous reads bypass HDF5 lock contention.
- **Vectorized Linking:** C-level broadcasting replaces pure-Python nested loops.
- **Unified Interface:** The `Tracker` Protocol cleanly prepares the package for multi-algorithm support.