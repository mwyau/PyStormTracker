# Plan: Hodges Tracker Integration (Final)

Integrate the Hodges (TRACK) adaptive constraints tracking algorithm (Hodges 1994, 1995, 1999) as a native Python/Numba implementation in `PyStormTracker`.

## Objective
Implement the Modified Greedy Exchange (MGE) algorithm with adaptive constraints, ensuring bit-wise conceptual parity with the original TRACK C implementation while leveraging Pythonic data structures and Numba performance.

## Configurable Parameters (Matching TRACK)
- `w1`, `w2`: Weights for the cost function (default: 0.2, 0.8). $w_1$ for direction, $w_2$ for speed.
- `dmax`: Maximum displacement distance (geodesic degrees).
- `phimax`: Penalty value for phantom points in the cost function.
- `min_lifetime`: Minimum number of time steps for a track to be considered valid (post-processing).
- `n_iterations`: Number of forward/backward MGE passes (default: 3).

## Core Components

### 1. Data Structures
- **Feature Points**: (lat, lon, value) with conversion to (x, y, z) unit cartesian coordinates.
- **Track Matrix**: A 2D array of shape `(n_tracks, n_frames)` storing indices to feature points. A special index (e.g., `-1`) represents a **phantom point**.

### 2. Hodges Detector (`src/pystormtracker/hodges/detector.py`)
- **Extrema Detection**: Use the Numba kernel for local min/max.
- **Sub-grid Refinement**: Implement local quadratic interpolation (fitting a 2D second-order polynomial to the 3x3 neighborhood of an extremum) to refine (lat, lon) coordinates, as per TRACK's `adjust_fpt.c`.

### 3. Hodges Linker (`src/pystormtracker/hodges/linker.py`)
- **Initialization**: 
    - Seed tracks using nearest-neighbor logic across all frames.
    - Pad tracks with phantom points to ensure all tracks span the full time range.
- **Modified Greedy Exchange (MGE)**:
    - **Forward Pass**: From $k=1$ to $n-1$, attempt to swap point $k+1$ between all pairs of tracks $(i, j)$.
    - **Backward Pass**: From $k=n-2$ down to $0$, attempt to swap point $k-1$ between all pairs of tracks $(i, j)$.
    - **Swap Condition**: A swap is made if:
        1. Both resulting tracks satisfy $d < d_{max}$.
        2. Both resulting tracks satisfy $\psi < \psi_{max}$ (if 3 real points exist).
        3. The total cost $\Xi$ for the pair $(i, j)$ is reduced.
- **Adaptive Constraints**:
    - $d_{max}$ and $\psi_{max}$ are updated based on the mean displacement of the track.

### 4. Spherical Geometry Kernels (`src/pystormtracker/hodges/kernels.py`)
- **`geod_dev`**: Implementation of the spherical cost function using tangent vector projections.
- **`subgrid_refine`**: Numba-optimized local quadratic refinement.
- **`mge_swap_logic`**: Numba-accelerated inner loop for the $O(N_{tracks}^2)$ exchange step.

### 5. Integration (`src/pystormtracker/hodges/tracker.py`)
- **Full Workflow**:
    1. **Preprocessing**: Apply `TaperFilter` and `SphericalHarmonicFilter` if configured.
    2. **Detection**: Extract refined feature points for all time steps.
    3. **Linking**: Initialize and optimize tracks using MGE.
    4. **Post-processing**: Prune tracks based on `min_lifetime` and intensity.

## Verification
1.  **Unit Tests**: Verify the cost function against manual calculations for known trajectories.
2.  **Comparison**: Use the provided `pyTRACK` wrapper to run the reference C implementation on the same data and compare output tracks.
3.  **Synthetic Tests**: Test robustness against "track crossing" scenarios and accelerating features.
