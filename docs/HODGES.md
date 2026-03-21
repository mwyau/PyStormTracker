# PyStormTracker Hodges Implementation

This document details the architecture, mathematical implementation, and design rationale for the Hodges tracking algorithm in `PyStormTracker`. The primary goal is **algorithmic parity** with the TRACK software (Hodges 1994, 1995, 1999) while updating the interface and performance.

---

## 1. Feature Identification Design

### 1.1 Object-Based Detection
**Design Choice**: Feature detection is implemented as a multi-stage pipeline: `Thresholding -> Connected Component Labeling (CCL) -> Object Filtering -> Local Extrema`.
- **References**: *Hodges 1994*, Section 2; `threshold.c` and `object_local_maxs.c`.

**Reasoning**: 
Original TRACK identifies "objects" (contiguous clusters of grid points exceeding a threshold) and then searches for extrema *only* within those objects. This prevents tracking isolated grid points that may represent noise.
- **Refinement**: The `min_points` parameter allows discarding small, insignificant features before identifying local extrema. (Ref: `object_filter.c`).

### 1.2 Connected Component Labeling (CCL)
**Design Choice**: Implemented `_numba_ccl` using **iterative label propagation** rather than TRACK's quad-tree approach.
- **References**: `hierarc_segment.c`, `form_objects.c`.

**Reasoning**: 
Original TRACK uses a quad-tree structure for segmentation, which was developed for limited memory environments but involves pointer-based recursion. In a Numba/Python implementation, flat-array iterative propagation is efficient, parallelizable, and produces identical object masks. This ensures algorithmic parity with legacy data structures.

### 1.3 Sub-grid Refinement
**Design Choice**: Used 2D local quadratic surface fitting for sub-grid precision.
- **References**: *Hodges 1995*, Section 3; `surfit.c`, `gdfp_optimize.c`.

**Reasoning**: 
While TRACK supports B-spline surfaces, this requires significant pre-processing (Cholesky decomposition). Quadratic fitting on a 3x3 neighborhood is a standard equivalent for identifying peaks between grid points. *Note: Global B-spline parity remains a target if absolute coordinate identity is required.*

---

## 2. Trajectory Linking (MGE Optimization)

### 2.1 Spherical Cost Function ($\psi$)
**Mathematical Formula**:
$$\psi = 0.5 w_1 [1 - \mathbf{\hat{T}}_1 \cdot \mathbf{\hat{T}}_2] + w_2 \left[ 1 - \frac{2\sqrt{d_1 d_2}}{d_1 + d_2} \right]$$
- **References**: *Hodges 1995*; `geod_dev.c`, `devn.c`.

**Reasoning**: 
The $0.5$ factor applied to the directional weight $w_1$ is a critical detail found in `geod_dev.c`. It normalizes the directional term (range 0 to 2) to [0, 1], ensuring that if $w_1 + w_2 = 1$, the total cost $\psi$ is also bounded by 1.0.

### 2.2 Modified Greedy Exchange (MGE Optimization)
**Design Choice**: Implemented alternating forward/backward passes with **one best swap per frame**.
- **References**: *Hodges 1999*, Section 2; `fel_mge.c`, `mge_tracks.c`, `initialize_mge.c`.

**Reasoning**: 
Original TRACK's `fel_mge.c` searches the *entire set of track pairs* at a specific frame $k$ to find the single swap that yields the maximum global gain. It then moves to frame $k+1$.
- **Convergence**: The optimization repeats the entire forward/backward cycle until no more swaps occur or `n_iterations` is reached.

### 2.3 Track Breaking (Track Fail)
**Design Choice**: Integrated displacement checks directly into the MGE passes.
- **References**: `track_fail.c`, `ub_disp.c`.

**Reasoning**: 
Original TRACK (`track_fail.c`) includes a mechanism to split trajectories if an exchange causes a point to exceed the maximum displacement ($d_{max}$). This implementation checks this after each swap, ensuring that the optimization never creates impossible physical links in an attempt to improve smoothness.

---

## 3. Adaptive Constraints

### 3.1 Regional $d_{max}$ (Zones)
**Implementation**: Passed via `zones` argument during tracker initialization.
- **References**: *Hodges 1999*, Section 3a; `read_zones.c`.

**Reasoning**: Storms move faster in the extratropics than the tropics. Applying a single $d_{max}$ globally either misses fast mid-latitude storms or creates noise in the tropics.

### 3.2 Speed-Dependent Smoothness (Adaptive $\psi_{max}$)
**Implementation**: Passed via `adapt_thresholds` and `adapt_values` arguments during tracker initialization.
- **References**: *Hodges 1999*, Section 3b; `read_adptp.c`.

**Reasoning**: As displacement (speed) increases, the directional constraint must become stricter.
- **Interpolation**: `PyStormTracker` uses piecewise linear interpolation between the 4 provided threshold/value pairs, matching the logic found in `read_adptp.c`.

---

## 4. Implementation Details

While maintaining algorithmic parity, the internal orchestration in `PyStormTracker` uses modern Python/Numba structures:

### 4.1 Track Seeding and Matrix Representation
**Design Choice**: All detected features across all frames are flattened into a single coordinate array. Tracks are managed as a **2D integer matrix** (`n_tracks` x `n_frames`), where each cell stores the index of a feature or a "phantom point" (-1).
- **Initial Linking**: Tracks are seeded with all features from the first frame. Subsequent frames are linked using a greedy nearest-neighbor approach within regional $d_{max}$ limits to create the starting state for MGE optimization.
- **References**: `initialize_mge.c`.

### 4.2 Initial Smoothness Breaking
**Design Choice**: An initial pass (`_initial_break_pass`) is performed right after greedy linking to split trajectories that exceed the adaptive smoothness threshold ($\psi_{max}$) before MGE begins.
- **References**: `track_fail.c`.

### 4.3 Gather-then-Link Orchestration
**Design Choice**: To ensure 100% bit-wise identity between serial and parallel runs, `PyStormTracker` parallelizes the **Detection** phase but gathers all results to a single process for the **Linking** (MGE) phase.
- **Impact**: This avoids the "track merging" complexities at process boundaries found in the original TRACK RSPLICE/LSPLICE utilities while maintaining performance, as MGE is efficient once detection is offloaded.

---

## 5. Implementation Strategy

- **Numba JIT**: All heavy mathematical loops (MGE, CCL, Geodesic math) are implemented as GIL-free, cache-enabled Numba kernels. This allows Python to match or exceed the speed of the original C code.
- **Xarray/NetCDF**: Legacy binary/ASCII I/O is replaced with Xarray, enabling seamless integration with climate data ecosystems (ERA5, CMIP6).
- **CLI**: A standard argparse-based CLI replaces the interactive `scanf`-heavy prompts of the original TRACK binary, facilitating HPC batch processing.
- **Phantom Points**: Maintained the concept of "phantom" points (-1 index) to handle missing frames and initialization of tracks of varying lengths, ensuring the MGE matrix remains rectangular.

---

## 5. Known Differences & Parity Status

While `PyStormTracker` aims for 100% parity, the following minor differences exist between this implementation and the legacy C version of TRACK:

### 5.1 Sub-grid Refinement (Peak Finding)
- **TRACK**: Fits a global B-spline surface to the data and uses a constrained conjugate gradient optimizer to find the extremum.
- **PyStormTracker**: Uses a 2D local quadratic surface fit to a 3x3 neighborhood.
- **Impact**: Coordinates will differ at the 2nd or 3rd decimal place. The topology of tracks (which points link together) is rarely affected on high-resolution grids ($< 1.0^\circ$).

### 5.2 CCL Implementation
- **TRACK**: Uses a quad-tree data structure for segmentation (`hierarc_segment.c`).
- **PyStormTracker**: Uses iterative label propagation in a Numba kernel (`_numba_ccl`).
- **Impact**: **None**. Both methods produce identical object masks from the same thresholded binary field. The Numba version is more efficient on flat-memory architectures.

### 5.3 Optimization Passes
- **TRACK**: Implements a recursive strategy that alternates forward/backward passes until no swaps occur, but often includes limits (e.g., 3 global iterations) in standard run scripts.
- **PyStormTracker**: Fully replicates the recursive forward/backward "one best swap per frame" logic. Convergence is guaranteed to match the original algorithm's local minimum.

### 5.4 Specialized Constraints
- **TRACK**: Supports some features like elliptical search areas or directional bias based on steering flows (e.g., $u, v$ components).
- **PyStormTracker**: Currently implements the standard Hodges 1995/1999 circular search radius ($d_{max}$) and adaptive smoothness ($\psi_{max}$).

## 6. Performance
All heavy mathematical loops are implemented as **GIL-free Numba-optimized JIT kernels** to ensure efficiency even with large numbers of feature points. The use of `xarray` I/O allows for faster data loading and preprocessing compared to TRACK's legacy binary formats.

---

## 7. References

The Hodges implementation in `PyStormTracker` is designed for algorithmic parity with the **TRACK-1.5.2** source code (specifically referencing `threshold.c`, `object_local_maxs.c`, `hierarc_segment.c`, `surfit.c`, `gdfp_optimize.c`, `geod_dev.c`, `fel_mge.c`, and `track_fail.c`).

### Key Literature

- **Hodges, K. I.**, 1999: Adaptive Constraints for Feature Tracking. *Mon. Wea. Rev.*, **127**, 1362–1373, [https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2).

- **Hodges, K. I.**, 1995: Feature Tracking on the Unit Sphere. *Mon. Wea. Rev.*, **123**, 3458–3465, [https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2](https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2).

- **Hodges, K. I.**, 1994: A General Method for Tracking Analysis and Its Application to Meteorological Data. *Mon. Wea. Rev.*, **122**, 2573–2586, [https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2](https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2).
