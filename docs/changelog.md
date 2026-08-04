# Changelog

## v0.6.1.dev0
### Tracks and formats

- Replaced the mutable nested track representation with packed immutable `Tracks` arrays and lightweight per-track views.
- Added TrackJSON 1.0 with typed `msgspec` models, generated JSON Schema, semantic validation, compact encoding, and optional derived statistics.
- Preserved primary variable, extrema mode, units, spatial bounds, time metadata, and preprocessing history across tracking and serialization.
- Centralized CF time conversion and calendar validation using signed millisecond times and the proleptic Gregorian calendar.
- Normalized recognized pressure and vorticity units and their detection thresholds.
- Improved IMILAST, TRACK tdump, GeoJSON, conversion, format detection, and empty-track handling.
- Temporarily disabled the existing HTML explorer while its replacement is developed.

### Tracking and preprocessing

- Consolidated preprocessing across Simple, Hodges, and HEALPix trackers.
- Separated optional spectral filtering, spatial tapering, and projection or HEALPix regridding.
- Derived HEALPix transform bandwidth from source and target grid resolution instead of applying an implicit T42 truncation.
- Aligned Simple serial, Dask, and MPI preprocessing, units, thresholds, metadata, and output.
- Fixed cyclic-longitude sampling for signed and unsigned longitude grids, descending axes, duplicate endpoints, and antimeridian crossings.

### Comparison and analysis

- Reworked trajectory comparison using temporal overlap and mean geodesic separation.
- Added matched-candidate output and corrected pressure-unit handling across TrackJSON and IMILAST inputs.

### Testing and infrastructure

- Added committed TrackJSON reference data and expanded unit and integration coverage.
- Added automatic verified retrieval of N320 reduced-Gaussian test data and local Zarr extraction.
- Consolidated CI, package testing, publishing, and multi-architecture Docker workflows.
- Added minimum-dependency, Python 3.14 free-threaded, ARM64, schema, and dependency-review checks.
- Corrected Linux ARM64 `ducc0` compilation in the Docker build.

---

## v0.6.0.dev0
### Tracking and preprocessing

- Added the modular `stormtracker track`, `sample`, `compare`, and `convert` command structure. The tracking command is implemented by `pystormtracker.track.run_tracker`; no package-level `pystormtracker.track()` function is exported.
- Added algorithm-dependent tri-state filtering and subgrid-refinement controls. Omitted values use the direct tracker defaults: disabled for Simple and enabled for Hodges and HEALPix.
- Added shared local quadratic subgrid refinement for regular and projected grids and local quadratic refinement on the HEALPix neighbor graph.
- Added one spherical `RectSphereBivariateSpline` fit per periodic global Hodges frame and a `bspline_val` diagnostic in raw detections at each quadratic center. The spline does not currently optimize the center coordinate, and the current linker does not propagate the diagnostic to final tracks.
- Added polar stereographic and HEALPix preprocessing with explicit `lmax` propagation, configurable projected extent and resolution, and corrected band-pass handling.
- Added DCT filtering for regional grids, full-Gaussian geometry handling, and reduced-Gaussian metadata, pseudo-analysis, filtering, and regridding paths.
- Corrected periodic-global and nonperiodic regional or projected boundary handling.
- Added serial Hodges detection chunking followed by one global linking pass. Hodges and HEALPix remain serial-only; Simple supports serial, Dask, and MPI execution.
- Added storm-object area and intensity-weighted ellipse diagnostics to raw Hodges detections. The current Hodges linker retains only the primary tracked variable.

### Metrics and analysis

- Added gridded cyclone amplitude, cyclone frequency, track frequency, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA).
- Added constant, Fisher, Cressman, linear, and quadratic spherical distance weights.
- Added 24-hour difference variance, Eddy Kinetic Energy, and high-wind percentile metrics.
- Added CORMAX, CCA/PCA truncation cross-validation, anomaly correlation coefficient, and fraction-of-variance-explained calculations through the `metrics` optional dependency.
- Added spatiotemporal track matching and external-variable sampling.

### Formats and interfaces

- Added the unified CLI subcommands and finite-value validation for numerical arguments.
- Added JSON-based sampling and comparison workflows and expanded conversion to IMILAST, TRACK tdump, JSON, and HTML outputs.
- Added the `eof` optional dependency group for `xeofs`.

### Testing and CI

- Added unit and integration tests for refinement, Gaussian and reduced-Gaussian preprocessing, projected filtering, metrics, sampling, comparison, conversion, and modular CLI commands.
- Added `--run-slow` for slow integration and historical regression cases.
- Corrected the Docker Trivy image reference and included slow integration cases in the designated Linux CI run.

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
- **Format Auto-detection Tests**: Added comprehensive tests for NetCDF, GRIB, and Zarr auto-detection in `DataLoader`.

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
- Optimized Docker layer caching for faster image builds.

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
- **Python 3 Migration**: Migration from Python 2.7 to Python 3.10+, including type hints and modern syntax.
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
