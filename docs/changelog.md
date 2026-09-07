# Changelog

## v0.7.0 - 2026-09-07

### Tracking and preprocessing

- Reconciled `HodgesTracker` with TRACK 1.5.4 across object detection, feature-point refinement, Modified Greedy Exchange (MGE) linking, adaptive constraints, missing-frame handling, track failure and splitting, segment splicing, and RSPLICE filtering.
- Added rectangular and spherical B-spline and quadratic feature-point refinement, together with storm-object area and intensity-weighted ellipse diagnostics.
- Unified preprocessing across Simple, Hodges, and HEALPix tracking, including spherical harmonic filtering, spectral and spatial tapering, polar and HEALPix regridding, regional discrete cosine transform filtering, and full and reduced Gaussian grids.
- Corrected cyclic-longitude sampling and periodic, regional, projected, and HEALPix boundary handling, and derived HEALPix transform bandwidth from source and target resolution.

### Data model, interfaces, and formats

- Replaced the procedural tracking interface with configured `SimpleTracker`, `HodgesTracker`, and `HealpixTracker` classes and the `stormtracker track`, `sample`, `compare`, and `convert` commands.
- Replaced nested mutable trajectories with packed immutable `Tracks` arrays and per-track views.
- Added TrackJSON 1.0 with typed `msgspec` models, generated JSON Schema, semantic validation, compact encoding, and preservation of variable, units, bounds, time, extrema mode, and preprocessing metadata.
- Centralized CF time and calendar handling, normalized recognized pressure and vorticity units, and improved IMILAST, TRACK tdump, format detection, conversion, and empty-track handling.

### Parallel execution and performance

- Added serial, Dask, and MPI execution for Simple, Hodges, and HEALPix tracking. Hodges execution separates `frame_workers`, `sht_threads`, and `mge_workers`, with parallel frame detection and MGE segment linking followed by deterministic splicing.
- Optimized rectangular detection, MGE preprocessing, and FITPACK rectangular spline systems while preserving the existing TRACK 1.5.4 comparison tests.
- Added progress reporting and logging for parallel tracking and expanded ARM64, minimum-dependency, and free-threaded Python 3.14 checks.

### Comparison and analysis

- Expanded trajectory comparison with temporal-overlap eligibility, geodesic separation, matched-candidate output, and assignment diagnostics, and added external-variable sampling along tracks.
- Added gridded cyclone amplitude, cyclone frequency, track frequency, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA), including hourly linear and PCHIP amplitude interpolation and constant, Cressman, linear, and quadratic spherical distance weights.
- Added 24-hour difference variance, eddy kinetic energy, high-wind percentile metrics, CORMAX, CCA/PCA truncation cross-validation, anomaly correlation coefficient, and fraction of variance explained calculations.

### Validation, testing, and distribution

- Completed full-year 2024 ERA5 mean sea-level pressure comparisons with TRACK 1.5.4 for F320 → T42 and F320 → F320, including raw and RSPLICE-filtered trajectory comparisons.
- Reorganized unit, integration, and parity tests with explicit `slow` and `data` markers and retained the bundled NCL/Spherepack T5-42 scalar spectral parity case.
- Pinned external integration and parity data to PyStormTracker-Data `v0.2.0-data` and added reduced-Gaussian ERA5 coverage.
- Added pre-commit checks for the uv lockfile, Ruff, ty, Markdown formatting, and common file errors; tightened mypy and CI checks; and moved package publishing to a workflow that runs after successful CI.
- Updated the scientific-method, architecture, testing, CLI, TrackJSON, benchmark, and contributor documentation.

---

## v0.5.0 - 2026-04-08

### Features

- **High-Precision Derivatives**: New kinematic-derivative functions for computing relative vorticity and divergence using spin-1 vector harmonics.
- **Planetary Constants**: Standardized Earth radius to 6,371,220 m across derivatives and tracking geometry.
- **pst-convert & JSON Support**: New utility for format conversion and a JSON format for track data.
- **Interactive Track Explorer**: Web-based visualization with filtering and time animation.
- **Hodges (TRACK) Implementation**: Added object detection, adaptive constraints, and Modified Greedy Exchange linking based on TRACK. Direct end-to-end TRACK validation remains ongoing.
- **Preprocessing & Performance**: Improved spherical harmonic filtering and backend auto-detection.
- **Remote Zarr Support**: Added support for remote Zarr datasets via HTTP, S3, and GS protocols.
- **Enhanced DataLoader**: Refactored `io.loader` to `io.data_loader` with automatic format detection for NetCDF, GRIB, and Zarr.
- **Dependency Errors**: Added error messages with installation instructions when optional dependencies (`cfgrib`, `zarr`) are missing.
- **Expanded Sample Data**: Integrated ERA5 UV850 sample datasets and Zarr-formatted alternatives in `utils.data`.

### Testing

- **NCL Validation**: New integration test suite validated against NCL 6.6.2 reference data.
- **Format Auto-detection Tests**: Added tests for NetCDF, GRIB, and Zarr auto-detection in `DataLoader`.

### CI/CD & Testing

- **Verification**: Added documentation builds and expanded test coverage.

### Maintenance

- **Spectral Backend Consolidation**: Evaluated `pyshtools`, SHTns, and `ducc0` for performance, accuracy, and portability. Selected `ducc0` as the production spherical harmonic transform backend. Historical SHTns comparisons and NCL kinematic validation are documented separately; JAX is not a current runtime backend.
- **Strict Typing**: Enabled strict `mypy` checks in core I/O modules and removed `Any` from the covered declarations.
- **Dependency Refinement**: Introduced a dedicated `zarr` optional dependency group and updated the `all` extra.

---

## v0.4.2 - 2026-03-19

### Performance

- Transitioned Dask backend to threaded scheduling for improved efficiency.
- Decoupled chunk processing to reduce memory overhead.
- Limited the default worker count to the available CPU count, with a maximum of four.

### Features

- Added `--chunk-size` (or `-c`) CLI argument to control processing granularity.
- Updated Docker publishing tag selection.

### Maintenance

- Updated CI concurrency rules to cancel redundant builds.

---

## v0.4.1 - 2026-03-18

> **Note:** First release available on [**Conda-Forge**](https://anaconda.org/conda-forge/pystormtracker).

### Testing

- Registered integration markers to resolve warnings in Conda-forge CI pipelines.

---

## v0.4.0 - 2026-03-14

### Architecture

- **Vectorized Data Model**: Refactored from Python-object trajectories to an array-backed structure.
- **JIT-Optimized Kernels**: Replaced core mathematical loops with Numba-compiled kernels for compiled execution.
- **Dask Integration**: Implemented Dask multiprocessing with tree reduction in this release. Later releases replaced tree reduction with Gather-then-Link.

### Features

- **GRIB Support**: Introduced support for GRIB files via the `cfgrib` engine.
- **Multi-Variable Tracking**: Support for tracking multiple variables per center using a flexible dictionary structure.

### Performance

- **Vectorized Linker**: Re-engineered the `SimpleLinker` using NumPy broadcasting for vectorized distance calculations.
- **Memory Efficiency**: Implemented `slots=True` for dataclasses and flat-array extraction for centers.

---

## v0.3.3 - 2026-03-10

### Security

- Integrated Trivy vulnerability scanning into the Docker build pipeline.
- Added SBOM (Software Bill of Materials) and provenance attestations to all releases.

### Maintenance

- Refined Docker image tagging and unified caching scopes.
- Simplified CI test matrix for faster verification.

---

## v0.3.2 - 2026-03-09

### CI/CD

- Added support for **ARM64** Docker images.
- Migrated project management to `uv` for deterministic builds and faster dependency resolution.
- Optimized Docker layer caching for faster verification.

### Maintenance

- Updated project homepage and refined repository metadata.

---

## v0.3.1 - 2026-03-08

### Maintenance

- Synchronized documentation versions and updated dependency lockfiles.

---

## v0.3.0 - 2026-03-08

### Features

- **IMILAST Export**: Added support for exporting cyclone trajectories in the standard IMILAST intercomparison format.
- **Automated Data Fetching**: Integrated `pooch` for automatic retrieval of ERA5 test datasets.

### Refactoring

- **Xarray Native**: Migrated detection pipeline to use Xarray for coordinate-aware processing.
- **Strict Typing**: Achieved 100% `mypy` compliance and dropped support for **Python 3.10** (enforced Python 3.11+ standards).

### Infrastructure

- Added Read the Docs documentation scaffolding.
- Implemented tiered integration testing (Short vs. Full variants).

---

## v0.2.2 - 2026-03-04

> **Note:** First release available on [**PyPI**](https://pypi.org/project/PyStormTracker/0.2.2/).

### Distribution

- Established automated publishing to **PyPI** via GitHub Actions.
- Added Zenodo DOI integration for scientific citation.

### Documentation

- Initial Read the Docs configuration.
- Standardized PyPI installation instructions in README.

---

## v0.2.1 - 2026-03-04

### Maintenance

- Metadata fixes and version synchronization.

---

## v0.2.0 - 2026-03-01

### Features

- **Dask Backend**: Introduced task-parallel execution with automatic worker detection.
- **CSV Output**: Transitioned from pickle to user-friendly CSV as the default output format.

### Architecture

- **Python 3 Migration**: Migration from Python 2.7 to Python 3.10+, including type hints and Python 3 syntax.
- **NetCDF4 Migration**: Switched from the legacy `Nio` library to `netCDF4` for NetCDF data handling.

### Refactoring

- Extracted core logic into `simple/` and `models/` modules for better maintainability.

---

## v0.0.2 - 2018-10-25

- Added `minmaxmode` support for ERA-Interim Mean Sea Level Pressure (MSL) and Vorticity (VO) tracking.

---

## v0.0.1 - 2016-01-11

- Initial release.
- Core cyclone tracking logic based on local extrema detection.
