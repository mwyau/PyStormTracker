# TRACK Feature Porting TODO

This document tracks work derived from TRACK and related analysis workflows. Completed items describe implemented behavior, not proof of numerical identity with TRACK.

- **Verified**: The implemented scope is covered by repository tests.
- **To be verified**: Implementation, reference-data comparison, or external-dataset validation remains.

## Planned Features and TRACK Source Mapping

- [ ] **B-Spline Detection/Smoothing** — Partial
  - **Description**: Off-grid detection using a global spherical B-spline and steepest descent to produce smoother tracks, especially useful for lower resolution grids (e.g., CMIP).
  - **TRACK References**: `src/spline_smooth.c`, Dierckx library in `lib/src/` (e.g., `sphery.f`, `smoopy.f`)
  - **Progress**: `RectSphereBivariateSpline` is fit once per global frame and its value is stored at the quadratic center. Direct optimization of the B-spline center and TRACK smoothing behavior remain to be implemented and validated.
  - **Verification**: To be verified against TRACK after center optimization is implemented.

- [x] **Regional Model Support (DCT)**
  - **Description**: Support for tracking on regional models (e.g., WRF) utilizing Discrete Cosine Transforms (DCT) for spectral filtering and domain transformations.
  - **TRACK References**: `src/track.c`, `src/statspl.F`
  - **Relevant Papers**: MWRE 2002 (DCT implementation)
  - **Progress**: `DCTFilter` uses the `ducc0.fft.dct` type-II/type-III transform pair with a radial wave-number band and an $l(l+1)$ exponential coefficient taper. `TaperFilter` provides the separate spatial boundary taper. Automatic grid selection distinguishes periodic global longitude from regional longitude.
  - **Verification**: Verified for execution and global/regional selection by repository tests. Numerical comparison with a regional reference filter remains to be verified.

- [x] **Spectral Tapering**
  - **Description**: Apply a wave-number taper to spherical harmonic coefficients to reduce spectral ringing and sidelobes (Gibbs phenomenon) during filtering (Hoskins and Sardeshmukh 1984).
  - **TRACK References**: `src/time_avg.c`, `src/spec_filt.c`
  - **Progress**: Harmonic coefficients use a configurable high-wave-number taper. `TaperFilter` provides a separate spatial boundary taper.
  - **Verification**: Verified for execution in spectral and tracker tests. Coefficient-level taper behavior remains to be verified against a reference calculation.

- [ ] **SHTns Comparison Benchmark**
  - **Description**: Restore a standalone spherical harmonic transform benchmark script for reproducible accuracy and timing comparisons between the SHTns and ducc0 engines. SHTns is not a production transform backend or package dependency.
  - **Progress**: Historical comparison results are retained in `docs/spectral_accuracy.md`; the executable benchmark harness still needs to be restored.
  - **Verification**: To be verified by running both implementations on the same checked-in or versioned reference fields and recording environment, truncation, error metrics, and timings.

- [x] **Postprocessing (Track Metrics)**
  - **Description**: Implement track metrics including cyclone amplitude, cyclone frequency, track frequency, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA) on a 2D spatial grid (Yau and Chang 2020).
  - **Relevant Papers**: Yau and Chang (2020) "Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts."
  - **Verification**: Verified by Lagrangian metric unit tests.

- [x] **Eulerian Metrics & Weather Impacts**
  - **Description**: Compute Eulerian variance metrics (e.g., 24-h difference filter for EKE850, Var(SLP)) and weather impact indices (e.g., 95th percentile 10-m wind speed) to evaluate against Lagrangian statistics.
  - **Relevant Papers**: Yau and Chang (2020).
  - **Progress**: Implemented in `metrics/eulerian.py`.
  - **Verification**: Verified by Eulerian metric unit tests.

- [x] **CORMAX Evaluation Framework**
  - **Description**: Add the Maximum one-point correlation (CORMAX) framework to find the highest correlation between weather impacts and storm track metrics within a localized spatial region (e.g., 60°x20° box).
  - **Relevant Papers**: Yau and Chang (2020).
  - **Progress**: Implemented in `metrics/cross_validation.py`.
  - **Verification**: Verified by cross-validation and missing-data unit tests.

- [x] **Spherical Kernel Gridding**
  - **Description**: Spherical kernel estimators for gridded track frequency and density.
  - **Progress**: Numba kernels support constant, Fisher, and Cressman weighting.
  - **Verification**: Verified by weighting and gridded Lagrangian metric unit tests.

- [ ] **Statistical Distributions and Confidence Intervals**
  - **Description**: Add distribution and confidence-interval analysis for track and gridded metrics. This is separate from spherical kernel gridding.
  - **Verification**: To be verified after methods and reference cases are defined and implemented.

- [x] **Object Size Calculation (Reverse Flood Fill)**
  - **Description**: Calculate area/size of the identified features (storm objects) using a reverse flood fill algorithm.
  - **TRACK References**: `src/boundary_find.c` (`ofill` logic), `src/shape_setup.c`
  - **Progress**: Implemented grid-cell area accumulation and intensity-weighted ellipse diagnostics. Global longitude is unwrapped within each object; projected grids use planar kilometre coordinates.
  - **Verification**: Verified for spherical, projected, boundary, and longitude-seam cases. Direct TRACK object-property comparison remains to be verified.

- [x] **Extra Variables Sampling**
  - **Description**: Add support for "sampling" or attaching secondary variables (e.g., max winds, full-res MSLP minima) to the identified trajectories from external NetCDF datasets.
  - **Progress**: `sample.py` supports nearest, bilinear, radius mean, radius maximum, and radius minimum sampling.
  - **Verification**: Verified by sampling unit and CLI integration tests.

- [ ] **Vorticity Tracking CLI** — Partial
  - **Description**: Standard tracking using relative vorticity as the primary field. Expose vorticity/divergence calculation in the CLI.
  - **TRACK References**: `src/compute_vorticity.c`, `src/compute_vorticity_fd.c`
  - **Progress**: The tracking CLI accepts existing relative-vorticity fields. `preprocessing.kinematics` computes vorticity and divergence through the Python API; a CLI command that derives them from wind is still planned.
  - **Verification**: To be verified after the derivation CLI is implemented. Python kinematics are covered by NCL-reference integration tests.

- [ ] **Ensemble & Dataset Utilities** — Partial
  - **Description**: Utilities to combine track files, match tracks between datasets (e.g., ensemble NWP members), and perform track-to-track intercomparison.
  - **TRACK References**: `utils/`, ensemble matching scripts (`GFS_SCRIPTS/eps_match.csh`, etc.)
  - **Progress**: `match_tracks` implements spatiotemporal track matching based on TRACK's `trdist_eps.c`; the `stormtracker compare` command exposes it through the CLI.
  - **Verification**: Track matching is verified by unit and CLI integration tests. A general track-file combine operation remains to be implemented and verified.

- [ ] **Storm Lifecycle Compositing**
  - **Description**: Implement functionality to produce storm-centered composites (averaging environmental fields like moisture or wind speed relative to the moving cyclone center).
  - **Notes**: Requested by K. Hodges for diagnostic analysis.
  - **Verification**: To be verified after implementation.

- [ ] **Tropical Cyclone (TC) Specialization**
  - **Description**: Documented configurations and filtering (e.g., T63) for tracking tropical cyclones.
  - **Relevant Papers**: Hodges et al. (2017) "How well are Tropical Cyclones represented in reanalysis data sets?"
  - **Verification**: To be verified against a documented TC reference workflow after implementation.

- [ ] **Gaussian Grid Support** — Partial
  - **Description**: Support for processing and regridding data directly from full Gaussian grids (e.g., N320) using `GL` geometry in `ducc0.sht.analysis_2d`.
  - **Notes**: Implemented auto-detection of `GL` geometry based on non-uniform latitude spacing.
  - **Verification**: Geometry detection and GL synthesis paths have repository coverage. A versioned full-Gaussian input fixture remains to be verified end to end.

- [x] **Reduced Gaussian Grid Support**
  - **Description**: Support reduced Gaussian grids (for example ERA5 N320 one-dimensional fields) using the per-latitude longitude counts required by spherical harmonic analysis.
  - **Progress**: `DataLoader` reads `GRIB_pl` ring-size metadata. Filtering and regridding use `ducc0.sht.pseudo_analysis` with varying `nlon`; an optional real-N320 integration suite covers loading, filtering, regridding, and tracking.
  - **Verification**: To be verified with the real N320 dataset. Synthetic reduced-grid unit tests cover metadata, filtering, and regridding code paths.

- [ ] **4D Feature-Tracking (STACKER)**
  - **Description**: Implementation of multi-dimensional tracking for cyclone systems.
  - **Relevant Papers**: Lakkis et al. (2019) "A 4D feature-tracking algorithm."
  - **Verification**: To be verified after implementation.
