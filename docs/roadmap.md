# PyStormTracker Roadmap

This roadmap lists planned implementation. Completed work is documented in the
changelog and technical documentation.

## 1. Weather-system workflows

### 1.1 Weather-system configurations

Add reusable configurations for extratropical and tropical cyclones, with
extension to systems such as easterly waves and mesocyclones. Configurations
define the tracked field, detection mode, spectral filter, grid treatment,
feature refinement, tracking constraints, and post-processing defaults.

### 1.2 Tropical-cyclone classification

Add Hart cyclone phase-space diagnostics and attach them to trajectories through
the track model and secondary-variable sampling.

Implement the warm-core, symmetric, and intensifying classification described
by Saffin et al. (2026), *An improved approach to identifying tropical cyclones
in model data using the cyclone phase space: warm core, symmetric, and
intensifying*, DOI
[10.22541/essoar.15003790/v1](https://doi.org/10.22541/essoar.15003790/v1).

### 1.3 Storm-centred composites

Add storm-centred composites with storm-relative coordinates, field
interpolation, optional lifecycle alignment, and aggregation across tracks.

## 2. Tracking and post-processing

### 2.1 Wind-derived vorticity and divergence

Expose spherical-harmonic vorticity and divergence from wind components through
the public preprocessing API and CLI.

### 2.2 Regional and rotated grids

Add projection-aware support for rotated latitude-longitude and other regional
model grids, including conversion of detected feature points to geographic
latitude and longitude before trajectory linking.

### 2.3 Ensemble and track-file utilities

Add track-file combination with metadata and identifier handling, and
ensemble-forecast matching corresponding to the TRACK EPS workflow.

### 2.4 Statistical distributions and significance

Implement spherical statistical distributions, confidence intervals, and
significance tests following Hodges (2008), *Confidence Intervals and
Significance Tests for Spherical Data Derived from Feature Tracking*.

### 2.5 User-defined adaptive constraints

Add a public constraint strategy for constructing time-step-aware displacement
and smoothness inputs for Hodges tracking.

### 2.6 Four-dimensional feature tracking

Implement STACKER-style multidimensional cyclone tracking following Lakkis et
al. (2019), with a compatible data model and Python API.

## 3. Performance and acceleration

### 3.1 `SimpleLinker` candidate search

Replace the all-pairs great-circle candidate matrix with a spherical spatial
index or bounded-neighbour search using geodesic distance and deterministic assignment.

### 3.2 GPU spectral preprocessing

Use `spharmgrid.jax` and S2FFT for GPU-accelerated rectangular global
preprocessing. Target Gauss--Legendre and Clenshaw--Curtis filtering,
regridding, and wind kinematics. Add a preprocessing backend selector separate
from the serial/Dask/MPI tracking backend.

Use JAX x64 and `float64` spatial fields. CUDA or TPU JAX installations are
user-selected environments.

Add a `spharmgrid.torch` path only for a tensor-native PyStormTracker workflow.
Extend GPU support to additional grid geometries as spharmgrid gains those
backends.
