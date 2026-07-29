# TRACK Feature Porting TODO

This document tracks the features to be ported from the original TRACK C source code into PyStormTracker to achieve full algorithmic parity and support legacy workflows.

## Planned Features and TRACK Source Mapping

- [x] **B-Spline Detection/Smoothing**
  - **Description**: Off-grid detection using a global spherical B-spline and steepest descent to produce smoother tracks, especially useful for lower resolution grids (e.g., CMIP).
  - **TRACK References**: `src/spline_smooth.c`, Dierckx library in `lib/src/` (e.g., `sphery.f`, `smoopy.f`)
  - **Notes**: Implemented using `scipy.interpolate.RectSphereBivariateSpline` for a global fit on the sphere (once per frame).

- [x] **Regional Model Support (DCT)**
  - **Description**: Support for tracking on regional models (e.g., WRF) utilizing Discrete Cosine Transforms (DCT) for spectral filtering and domain transformations.
  - **TRACK References**: `src/track.c`, `src/statspl.F`
  - **Relevant Papers**: MWRE 2002 (DCT implementation)
  - **Notes**: Implemented `DCTFilter` using `ducc0.fft.dct` for fast 2D transforms with Hoskins spatial tapering.

- [x] **Spectral Tapering**
  - **Description**: Spatial spectral tapering windowing (Hoskins and Sardeshmukh 1984) applied to spherical harmonic coefficients to reduce ringing/sidelobe artifacts (Gibbs phenomenon) during spectral filtering.
  - **TRACK References**: `src/time_avg.c`, `src/spec_filt.c`

- [x] **Postprocessing (Track Metrics)**
  - **Description**: Implement track metrics including cyclone amplitude, cyclone frequency, track frequency, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA) on a 2D spatial grid (Yau and Chang 2020).
  - **Relevant Papers**: Yau and Chang (2020) "Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts."

- [x] **Eulerian Metrics & Weather Impacts**
  - **Description**: Compute Eulerian variance metrics (e.g., 24-h difference filter for EKE850, Var(SLP)) and weather impact indices (e.g., 95th percentile 10-m wind speed) to evaluate against Lagrangian statistics.
  - **Relevant Papers**: Yau and Chang (2020).
  - **Notes**: Implemented in `metrics/eulerian.py`.

- [x] **CORMAX Evaluation Framework**
  - **Description**: Add the Maximum one-point correlation (CORMAX) framework to find the highest correlation between weather impacts and storm track metrics within a localized spatial region (e.g., 60°x20° box).
  - **Relevant Papers**: Yau and Chang (2020).
  - **Notes**: Implemented in `metrics/cross_validation.py`.

- [x] **Spherical Statistics**
  - **Description**: Spherical kernel estimators for calculating track frequency/density, statistical distributions, and confidence intervals on a sphere.
  - **Notes**: Implemented a unified Numba-optimized kernel estimator supporting Constant (Yau & Chang 2020), Fisher (Hodges 1999), and Cressman (Simmonds 2026) kernels.

- [x] **Object Size Calculation (Reverse Flood Fill)**
  - **Description**: Calculate area/size of the identified features (storm objects) using a reverse flood fill algorithm.
  - **TRACK References**: `src/boundary_find.c` (`ofill` logic), `src/shape_setup.c`
  - **Notes**: Implemented both raw area (km^2) and precise intensity-weighted ellipse fitting.

- [x] **Extra Variables Sampling**
  - **Description**: Add support for "sampling" or attaching secondary variables (e.g., max winds, full-res MSLP minima) to the identified trajectories from external NetCDF datasets.
  - **Notes**: Implemented in `sample.py` with support for nearest, bilinear, area-average, and local max/min sampling.

- [ ] **Vorticity Tracking CLI**
  - **Description**: Standard tracking using relative vorticity as the primary field. Expose vorticity/divergence calculation in the CLI.
  - **TRACK References**: `src/compute_vorticity.c`, `src/compute_vorticity_fd.c`

- [x] **Ensemble & Dataset Utilities**
  - **Description**: Utilities to combine track files, match tracks between datasets (e.g., ensemble NWP members), and perform track-to-track intercomparison.
  - **TRACK References**: `utils/`, ensemble matching scripts (`GFS_SCRIPTS/eps_match.csh`, etc.)
  - **Notes**: Implemented `match_tracks` in `metrics/compare.py` mirroring the logic in TRACK's `trdist_eps.c`. Added `stormtracker compare` CLI.

- [ ] **Storm Lifecycle Compositing**
  - **Description**: Implement functionality to produce storm-centered composites (averaging environmental fields like moisture or wind speed relative to the moving cyclone center).
  - **Notes**: Requested by K. Hodges for diagnostic analysis.

- [ ] **Tropical Cyclone (TC) Specialization**
  - **Description**: Optimized configurations and filtering (e.g., T63) for tracking Tropical Cyclones.
  - **Relevant Papers**: Hodges et al. (2017) "How well are Tropical Cyclones represented in reanalysis data sets?"

- [x] **Gaussian Grid Support**
  - **Description**: Support for processing and regridding data directly from full Gaussian grids (e.g., N320) using `GL` geometry in `ducc0.sht.analysis_2d`.
  - **Notes**: Implemented auto-detection of `GL` geometry based on non-uniform latitude spacing.

- [ ] **Reduced Gaussian Grid Support**
  - **Description**: Native support for reduced Gaussian grids (e.g., native ERA5 N320 unstructured 1D arrays) using `ducc0.sht.analysis` with varying `nlon` parameters.

- [ ] **4D Feature-Tracking (STACKER)**
  - **Description**: Implementation of multi-dimensional tracking for a more comprehensive view of cyclone systems.
  - **Relevant Papers**: Lakkis et al. (2019) "A 4D feature-tracking algorithm."
