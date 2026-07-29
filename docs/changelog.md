# Changelog

## v0.6.0-dev
### Tracking and preprocessing
- Added polar stereographic and HEALPix preprocessing paths with explicit `lmax` propagation.
- Added shared quadratic subgrid refinement. Simple defaults to disabled; Hodges and HEALPix default to enabled.
- Added algorithm-dependent `--filter` and `--subgrid-refine` boolean overrides.
- Corrected periodic-global versus nonperiodic regional/projected boundary handling.
- Changed Hodges time partitioning to gather detections before one linking pass.
- Added reduced-Gaussian metadata, filtering, regridding, and optional real-N320 integration coverage.

### Validation
- Added serial/Dask polar projection equality tests and coordinate-selection tests.
- Enabled the legacy Simple vorticity regression with its historical `1e-4` threshold. The production default remains `1e-5`.
- Added finite-value validation for numerical CLI arguments.

## v0.5.0 - 2026-04-08
### Features
- **Kinematic Derivatives**: Added relative vorticity and divergence calculations using spin-1 vector harmonics.
- **Planetary Constants**: Standardized Earth radius to 6,371,220 m across derivatives and tracking geometry.
- **Convert and JSON Support**: Added `stormtracker convert`, JSON track data, and HTML explorer output.
- **Interactive Track Explorer**: Added an HTML track view with filters and time animation.
- **Hodges Tracker**: Added object detection, adaptive constraints, and Modified Greedy Exchange linking based on TRACK.
- **Preprocessing**: Added spherical harmonic filtering and backend selection.
- **Remote Zarr Support**: Added support for remote Zarr datasets via HTTP, S3, and GS protocols.
- **DataLoader**: Refactored `io.loader` to `io.data_loader` with format detection for NetCDF, GRIB, and Zarr.
- **Dependency Errors**: Added installation guidance when optional dependencies (`cfgrib`, `zarr`) are missing.
- **Sample Data**: Added ERA5 UV850 sample datasets and Zarr alternatives.

### Testing
- **NCL Validation**: New integration test suite validated against NCL 6.6.2 reference data.
- **Format Detection Tests**: Added tests for NetCDF, GRIB, and Zarr detection in `DataLoader`.

### CI/CD & Testing
- **Verification**: Added documentation builds and test coverage.

### Maintenance
- **Spectral Backend**: Selected `ducc0` as the production SHT implementation. Historical SHTns comparisons are documented separately; SHTns and JAX are not current runtime backends.
- **Strict Typing**: Enabled strict `mypy` checks and removed `Any` from core declarations.
- **Dependency Refinement**: Introduced a dedicated `zarr` optional dependency group and updated the `all` extra.

---

## v0.4.2 - 2026-03-19
### Performance
- Transitioned Dask detection to threaded scheduling.
- Decoupled chunk processing to reduce memory overhead.
- Limited the default worker count to four.

### Features
- Added `--chunk-size` (or `-c`) CLI argument to control processing granularity.
- Updated Docker tag selection.

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
- **Vectorized Data Model**: Replaced Python-object trajectory storage with array-backed storage.
- **Numba Kernels**: Replaced core mathematical loops with Numba-compiled kernels.
- **Dask Integration**: Added Dask execution with tree reduction in this release; later releases replaced tree reduction with Gather-then-Link.

### Features
- **GRIB Support**: Introduced support for GRIB files via the `cfgrib` engine.
- **Multi-Variable Tracking**: Support for tracking multiple variables per center using a flexible dictionary structure.

### Performance
- **Vectorized Linker**: Reworked `SimpleLinker` using NumPy broadcasting.
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
- Updated Docker layer caching.

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
- **Xarray Input**: Migrated detection input to Xarray for coordinate-aware processing.
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
- **CSV Output**: Replaced pickle with CSV as the default output format in this release.

### Architecture
- **Python 3 Migration**: Migrated from Python 2.7 to Python 3.10+ with type annotations.
- **NetCDF4 Migration**: Replaced `Nio` with `netCDF4`.

### Refactoring
- Extracted core logic into `simple/` and `models/` modules for better maintainability.

---

## v0.0.2 - 2018-10-25
- Added `minmaxmode` support for ERA-Interim Mean Sea Level Pressure (MSL) and Vorticity (VO) tracking.

---

## v0.0.1 - 2016-01-11
- Initial release.
- Core cyclone tracking logic based on local extrema detection.
