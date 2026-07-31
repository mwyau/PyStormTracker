# PyStormTracker Architecture

This document describes the data model, tracker interfaces, preprocessing, and execution paths in PyStormTracker.

## 1. Architecture Principles

The architecture has four main properties:
1.  **Unified API (Tracker Protocol):** A structural interface that allows the CLI and Python API to support multiple tracking algorithms (e.g., `SimpleTracker`, `HodgesTracker`) interchangeably.
2.  **Centralized Threshold Management:** Standard detection thresholds (for example, `1e-5` for vorticity and `0.0` for MSL) are defined in `models/constants.py` and shared by tracker paths.
3.  **Vectorization and Numba JIT:** Numerical operations use NumPy broadcasting and cached, GIL-free Numba kernels where suitable, avoiding Python loops in the detection and linking kernels.
4.  **Gather-then-Link:** Simple Dask and MPI execution distribute detection, gather raw detections in time order, and run one linking pass.

---

## 2. Core Components

### 2.1 Array-Backed Data Models (`Tracks`, `Track`, `Center`)
The data models use contiguous array-backed storage:
*   **`Tracks`**: The central container holding one-dimensional NumPy arrays for `track_ids`, `times`, `lats`, `lons`, and a dictionary of additional meteorological variables.
*   **`Track`**: A lightweight view into the `Tracks` arrays for one identifier.
*   **`Center`**: A dataclass used for iteration and export.

This layout avoids storing one persistent Python object per center, reduces object-allocation overhead, and supports NumPy broadcasting, selection, and serialization between parallel workers.

### 2.2 Shared DataLoader
Data loading is encapsulated in a dedicated `DataLoader` class (`io/data_loader.py`). This component handles:
*   **Format handling**: Opens NetCDF through `h5netcdf` or `netCDF4`, GRIB through `cfgrib`, and Zarr through Xarray engines.
*   **Remote data**: Supports Zarr stores over HTTP, S3, and GS through `fsspec` when the optional Zarr dependencies are installed.
*   **Variable and coordinate mapping**: Resolves common field aliases such as `msl`/`slp` and latitude/longitude/time coordinate aliases.
*   **Grid metadata**: Detects regular latitude-longitude, full Gaussian, reduced Gaussian, projected `x/y`, and HEALPix coordinates and retains metadata required by spherical harmonic transforms and map projections.

### 2.3 Heuristic Tracking Implementation (SimpleTracker)
Trajectory construction in the Simple tracker uses NumPy broadcasting to construct great-circle distance matrices from clamped unit-vector dot products. Candidate centers are sorted lexicographically before deterministic greedy nearest-neighbor matching between consecutive time steps.

### 2.4 Optimization-Based Tracking Implementation (HodgesTracker)
The `HodgesTracker` implements methods based on TRACK (Hodges 1994, 1995, 1999):
*   **Object-Based Detection**: Features are identified using `Thresholding -> Connected Component Labeling (CCL) -> Object Filtering -> Local Extrema`.
*   **Modified Greedy Exchange (MGE)**: An iterative algorithm that swaps points between tracks to minimize a total cost function.
*   **Spherical Cost Function**: Penalizes changes in the tangent direction and displacement magnitude of consecutive track segments.
*   **Adaptive Constraints**: Adjusts maximum displacement ($d_{max}$) and smoothness limits ($\psi_{max}$) from regional zones and track displacement.
*   **Sub-grid Refinement**: Fits a local quadratic surface around each eligible extremum. On periodic global grids, a spherical spline value is also evaluated at that center.

### 2.5 Parallel Pipeline (Gather-then-Link)

Simple parallel execution uses the following sequence:
1.  **Parallel Detection**: Assigned time chunks are distributed across Dask or MPI workers. Each worker runs Numba kernels to find centers and returns raw coordinate arrays.
2.  **Centralized Linking**: The main process gathers the raw detections from all workers and performs a single sequential link. 

Detection contains the frame-local Numba kernels and can be partitioned without making link decisions at chunk boundaries. Centralized linking therefore avoids track merging across independently linked chunks. Repository integrations compare complete serial, Dask, and MPI Simple outputs using versioned test data. Hodges supports serial chunked detection followed by one linking pass. Hodges and HEALPix do not currently provide Dask or MPI tracking.

---

## 3. Command Line Interface Architecture

PyStormTracker uses `argparse` subcommands.

### 3.1 Modular Router Pattern
The `cli.py` module acts as a thin entry point. It utilizes `argparse` subparsers to delegate argument definition and execution to specialized modules:
*   **`track.py`**: Core tracking pipeline.
*   **`sample.py`**: Post-processing for variable enrichment (e.g., sampling precipitation along tracks).
*   **`compare.py`**: Intercomparison utilities for matching tracks between datasets.
*   **`convert.py`**: IO conversion and visualization generation.

### 3.2 Decoupled Analysis Commands
Tracking, secondary-variable sampling, track comparison, and format conversion are separate commands. A primary track file can be reused when sampling additional meteorological fields or comparing datasets, without rerunning detection and trajectory linking on the original three-dimensional field.

---

## 4. The `Tracker` Protocol

The `Tracker` Protocol (defined in `src/pystormtracker/models/tracker.py`) provides a standardized interface for all tracking algorithms:

```python
import pystormtracker as pst

# Instantiate any compliant tracker
tracker = pst.SimpleTracker()

# Standardized .track() method
tracks = tracker.track(
    infile="era5_msl.nc", varname="msl", start_time="2025-01-01", backend="dask"
)

# Standardized export
tracks.write("output.txt", format="imilast")
```

---

## 5. Planned Architecture Work

Current planned work includes:

*   **Xarray generalized ufuncs:** Spectral filtering and kinematic derivatives use `xr.apply_ufunc(..., dask="parallelized")`; detection still uses explicit time partitioning.
*   **Lazy evaluation and thread topology:** Reduce eager frame loading and define ducc0 and Numba thread counts when Dask threads or MPI processes distribute work.
*   **Spatial indexing:** Evaluate `scipy.spatial.cKDTree` or another spherical candidate index to reduce the $O(N \times M)$ candidate-search cost in dense Simple linking workloads.
*   **Additional backends:** Hodges and HEALPix parallel tracking require Gather-then-Link implementations and serial-parallel equality tests.

For more details on specific planned implementations, see the [Roadmap](roadmap.md).

---

## 6. Performance Benchmarks

The benchmark page records Simple tracker timings for versions `v0.3.3` and `v0.4.0` on one workstation.

Detailed execution timings (breaking down Detection, Linking, Export, and I/O Overhead) across Serial, Dask, and MPI backends for both standard and high-resolution ERA5 datasets are available in the [Benchmark Report](benchmark.md).

---

## Appendix: Evolution from Legacy Architecture

The current architecture differs from the earlier nested-object design as follows.

| Feature | Legacy Architecture (v0.3.x and earlier) | Current Architecture (v0.4.0+) |
| :--- | :--- | :--- |
| **Data Storage** | Nested lists of `Center` and `Track` objects. | Flat, C-contiguous NumPy arrays. |
| **Parallelism** | Threaded tree reduction. | Simple: threaded Dask or MPI detection, then centralized linking. |
| **Linking Strategy** | Tree reduction across chunks. | Parallel detection and centralized linking with serial-equality tests. |
| **Linker** | $O(N^2)$ nested Python loops. | Vectorized NumPy great-circle distance matrix and deterministic greedy matching. |
| **Algorithms** | Simple only. | Simple, Hodges-based, and HEALPix trackers. |
| **I/O** | Many small lazy-loaded chunks. | Xarray reads coordinated through `DataLoader`. |
