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
- **Validation**: Refined positions must remain within the original grid cell; otherwise, the algorithm falls back to the grid center.

### 2.3 Spherical Geometry (`hodges/kernels.py`)
- **Geodesic Distance**: Calculated using the Haversine formula or the dot product of unit cartesian vectors.
- **Cost Function ($\psi$)**: Measures the deviation over three consecutive points ($P_{k-1}, P_k, P_{k+1}$):
  $\psi = w_1(1 - \mathbf{\hat{T}}_1 \cdot \mathbf{\hat{T}}_2) + w_2 \left( 1 - \frac{2\sqrt{d_1 d_2}}{d_1 + d_2} \right)$
  - The first term represents **directional smoothness** (dot product of tangent vectors at the central point $P_k$).
  - The second term represents **speed consistency** (ratio of distances).
  - Default weights: $w_1 = 0.2$, $w_2 = 0.8$.

### 2.4 Linking & Optimization (`hodges/linker.py`)
- **Initialization**: Tracks are seeded using a nearest-neighbor approach across all frames. Tracks are padded with **phantom points** (placeholders) so that every track matrix entry has a value, enabling global optimization.
- **Modified Greedy Exchange (MGE)**:
  - **Forward Pass**: Iterates from $k=1$ to $n-1$, attempting to swap point $k+1$ between all pairs of tracks $(i, j)$.
  - **Backward Pass**: Iterates from $k=n-2$ down to $0$, attempting to swap point $k-1$ between all pairs of tracks.
  - **Swap Condition**: A swap is executed if the total cost $\Xi = \text{cost}_i + \text{cost}_j$ is reduced and both resulting tracks satisfy the displacement and smoothness constraints.

### 2.5 Adaptive Constraints
- **Regional Search Radius ($d_{max}$)**: Supports latitude/longitude zones. The effective $d_{max}$ for a pair of points is the average of the $d_{max}$ values assigned to their respective zones.
- **Adaptive Smoothness ($\psi_{max}$)**: A piecewise linear function adjusts the upper-bound smoothness penalty based on the mean displacement of the track triplet. Slower systems are subject to stricter smoothness constraints.

## 3. Configuration & Compatibility
The `HodgesTracker` can be configured programmatically or by loading standard TRACK configuration files:
- **`zone.dat`**: Defines regional $d_{max}$ zones.
- **`adapt.dat`**: Defines the 4-point linear interpolation for $\psi_{max}$.

## 4. Performance
All heavy mathematical loops, including distance matrices, quadratic fits, and the $O(N_{tracks}^2)$ MGE exchange loops, are implemented as **Numba-optimized JIT kernels** to ensure high performance even with large numbers of feature points.
