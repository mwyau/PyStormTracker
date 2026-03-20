# PyStormTracker Hodges Implementation

This document details the architecture, mathematical implementation, and design rationale for the Hodges tracking algorithm in `PyStormTracker`. The primary goal is **100% algorithmic parity** with the TRACK software (Hodges 1994, 1995, 1999) while modernizing the interface and performance.

---

## 1. Feature Identification Design

### 1.1 Object-Based Detection (Hodges 1994)
**Design Choice**: Replaced global extrema search with a multi-stage pipeline: `Thresholding -> CCL -> Object Filtering -> Local Extrema`.

**Reasoning**: 
Original TRACK (see `threshold.c` and `object_local_maxs.c`) does not find all grid-point extrema. It identifies "objects" (contiguous clusters of points exceeding a threshold) and then searches for extrema *only* within those objects. This prevents noise in small, isolated grid points from being tracked.
- **Refinement**: Added `min_points` parameter to the CLI/API to match TRACK's `object_filter` logic, allowing users to discard small, insignificant features early.

### 1.2 Connected Component Labeling (CCL)
**Design Choice**: Implemented `_numba_ccl` using **iterative label propagation** rather than TRACK's quad-tree approach (`hierarc_segment.c`).

**Reasoning**: 
Original TRACK uses a quad-tree structure for segmentation, which was memory-efficient for 1990s hardware but involves complex pointer-based recursion. In a modern Numba/Python environment, flat-array iterative propagation is highly performant, easily parallelizable, and produces identical object masks. This ensures algorithmic parity without the technical debt of legacy data structures.

### 1.3 Sub-grid Refinement
**Design Choice**: Used 2D local quadratic surface fitting for sub-grid precision.

**Reasoning**: 
While TRACK supports B-spline surfaces (`surfit.c`, `gdfp_optimize.c`), this requires significant pre-processing (Cholesky decomposition). Quadratic fitting on a 3x3 neighborhood is the standard "modern" equivalent for identifying peaks between grid points. *Note: Global B-spline parity remains a future target if absolute coordinate identity is required.*

---

## 2. Trajectory Linking (MGE Optimization)

### 2.1 Spherical Cost Function ($\psi$)
**Mathematical Formula**:
$$\psi = 0.5 w_1 [1 - \mathbf{\hat{T}}_1 \cdot \mathbf{\hat{T}}_2] + w_2 \left[ 1 - \frac{2\sqrt{d_1 d_2}}{d_1 + d_2} \right]$$

**Reasoning**: 
The $0.5$ factor applied to the directional weight $w_1$ is a critical detail found in `geod_dev.c` and `Hodges 1995`. It normalizes the directional term (range 0 to 2) to [0, 1], ensuring that if $w_1 + w_2 = 1$, the total cost $\psi$ is also bounded by 1.0.

### 2.2 Modified Greedy Exchange (Hodges 1999)
**Design Choice**: Implemented alternating forward/backward passes with **one best swap per frame**.

**Reasoning**: 
Unlike simpler greedy algorithms that swap as they go, TRACK's `fel_mge.c` searches the *entire set of track pairs* at a specific frame $k$ to find the single swap that yields the maximum global gain. It then moves to frame $k+1$.
- **Recursion**: The optimization now repeats the entire forward/backward cycle until no more swaps occur, matching TRACK's convergence behavior.

### 2.3 Track Breaking (Track Fail)
**Design Choice**: Integrated `_break_track` logic directly into the MGE passes.

**Reasoning**: 
Original TRACK (`track_fail.c`) includes a mechanism to split trajectories if an exchange causes a point to exceed the maximum displacement ($d_{max}$). My implementation checks this post-swap, ensuring that the optimization never creates "illegal" physical links in an attempt to improve smoothness.

---

## 3. Adaptive Constraints (Hodges 1999)

### 3.1 Regional $d_{max}$ (Zones)
**Implementation**: Supports `zone.dat` format.
**Reasoning**: Storms move faster in the extratropics than the tropics. Applying a single $d_{max}$ globally either misses fast mid-latitude storms or creates noise in the tropics.

### 3.2 Speed-Dependent Smoothness (Adaptive $\psi_{max}$)
**Implementation**: Supports `adapt.dat` format via `get_adaptive_phimax`.
**Reasoning**: As displacement (speed) increases, the directional constraint must become stricter. A storm moving 10°/step should not make sharp turns, whereas a slow-moving system (1°/step) can "wobble" more without being unphysical. This is implemented as a piecewise linear function of mean displacement.

---

## 4. Modernization Strategy

- **Numba JIT**: All heavy mathematical loops (MGE, CCL, Geodesic math) are implemented as GIL-free, cache-enabled Numba kernels. This allows Python to match or exceed the speed of the original C code.
- **Xarray/NetCDF**: Legacy binary/ASCII I/O is replaced with Xarray, enabling seamless integration with climate data ecosystems (ERA5, CMIP6).
- **CLI**: A modern argparse-based CLI replaces the interactive `scanf`-heavy prompts of the original TRACK binary, facilitating batch processing.
- **Phantom Points**: Maintained the concept of "phantom" points (-1 index) to handle missing frames and initialization of tracks of varying lengths, ensuring the MGE matrix remains rectangular.
