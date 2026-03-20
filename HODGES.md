# Architecture: Hodges (TRACK) Tracker Implementation

This document describes the architecture and mathematical implementation of the Hodges tracking algorithm in `PyStormTracker`, providing parity with the industry-standard TRACK software (Hodges 1994, 1995, 1999).

## 1. Overview
The Hodges tracker identifies and links atmospheric features using a spherical cost function optimization approach known as **Modified Greedy Exchange (MGE)**. It is specifically designed to handle features with varying speeds and directions using adaptive constraints.

## 2. Core Components

### 2.1 Preprocessing (`preprocessing/taper.py`)
- **`TaperFilter`**: Applies a cosine taper to the edges of the spatial domain. This minimizes "ringing" artifacts during subsequent spherical harmonic filtering, which is common in TRACK workflows.

### 2.2 Feature Detection (`hodges/detector.py` & `hodges/kernels.py`)
- **Extrema Detection**: Identifies local minima or maxima within a specified search window, handling longitude periodicity.
- **Sub-grid Refinement**: For each grid-level extremum, a 2D quadratic surface is fitted to its 3x3 neighborhood:
  $f(y, x) = ay^2 + bx^2 + cyx + dy + ex + f$
  The refined center $(y_{ref}, x_{ref})$ is found where the partial derivatives $\frac{\partial f}{\partial y} = \frac{\partial f}{\partial x} = 0$.
- **Validation**: Refined positions must remain within the original grid cell. Precise coordinate intervals are used between neighbors to handle non-uniform or Gaussian grids.

### 2.3 Spherical Geometry (`hodges/kernels.py`)
- **Geodesic Distance**: Calculated using the Haversine formula or the dot product of unit cartesian vectors.
- **Cost Function ($\psi$)**: Measures the deviation over three consecutive points ($P_{k-1}, P_k, P_{k+1}$):
  $\psi = 0.5 w_1(1 - \mathbf{\hat{T}}_1 \cdot \mathbf{\hat{T}}_2) + w_2 \left( 1 - \frac{2\sqrt{d_1 d_2}}{d_1 + d_2} \right)$
  - The first term represents **directional smoothness** (normalized to [0, 1]).
  - The second term represents **speed consistency** (ratio of distances).
  - Default weights: $w_1 = 0.2$, $w_2 = 0.8$.
- **Phantom Handling**: If $P_{k-1}$ is a phantom, cost is 0. If $P_{k-1}$ is real but $P_k$ or $P_{k+1}$ are phantoms, cost is `phimax`.

### 2.4 Linking & Optimization (`hodges/linker.py`)
- **Initialization**: 
  1. Seed tracks using a nearest-neighbor approach.
  2. Pad tracks with **phantom points**.
  3. **Initial Breaking Pass**: Process all tracks forward once. If any triplet $(P_{k-1}, P_k, P_{k+1})$ exceeds the local $\psi_{max}$, the track is broken at point $k$.
- **Modified Greedy Exchange (MGE)**:
  - **Optimization**: For each time step, the algorithm identifies the **best swap** (largest gain in cost reduction) among all track pairs and executes it. This repeats for the current step until **local convergence**.
  - **Passes**: Iterates forward from $k=1$ to $n-1$ and backward from $k=n-2$ down to $0$.
  - **Convergence**: Global passes repeat until no further gains are possible.
- **Displacement Logic**: If one point in a pair is a phantom, the displacement is assumed to be `dmax`.

### 2.5 Adaptive Constraints
- **Regional Search Radius ($d_{max}$)**: Supports latitude/longitude zones. The effective $d_{max}$ for a pair of points is the average of the $d_{max}$ values assigned to their respective zones.
- **Adaptive Smoothness ($\psi_{max}$)**: A piecewise linear function adjusts the upper-bound smoothness penalty based on the mean displacement of the track triplet.
- **`max_missing`**: Limits the number of consecutive phantom points allowed in a track.

## 3. Segmented Tracking & Splicing
For long time series, processing is performed in overlapping chunks.
- **Splicing**: Tracks from adjacent chunks are matched if they end/start with identical points (same time, lat, lon).

## 4. Configuration & IO
The `HodgesTracker` can be configured programmatically or by loading standard TRACK configuration files (`zone.dat`, `adapt.dat`).
- **Output Support**: Supports the industry-standard **Hodges (TRACK) ASCII** format (`tdump`), enabling interoperability with legacy tools.

## 5. Performance
All heavy mathematical loops are implemented as **GIL-free Numba-optimized JIT kernels** to ensure high performance even with large numbers of feature points.
