# Changelog

## v0.5.0 - 2026-04-08
### Features
- **High-Precision Derivatives**: New `vodv` module for computing relative vorticity and divergence using spin-1 vector harmonics.
- **Planetary Constants**: Standardized Earth radius to 6,371,220 m across derivatives and tracking geometry.
- **pst-convert & JSON Support**: New utility for format conversion and a GPU-optimized JSON format for track data.
- **Interactive Track Explorer**: Web-based visualization with real-time filtering and animations.
- **Hodges (TRACK) Parity**: Full implementation of the Hodges algorithm with TRACK parity.
- **Preprocessing & Performance**: Improved spherical harmonic filtering and backend auto-detection.
- **Remote Zarr Support**: Added support for remote Zarr datasets via HTTP, S3, and GS protocols.
- **Enhanced DataLoader**: Refactored `io.loader` to `io.data_loader` with automatic format detection for NetCDF, GRIB, and Zarr.
- **Improved UX**: Added friendly error messages with installation instructions when optional dependencies (`cfgrib`, `zarr`) are missing.
- **Expanded Sample Data**: Integrated ERA5 UV850 sample datasets and Zarr-formatted alternatives in `utils.data`.

### Testing
- **NCL Validation**: New integration test suite validated against NCL 6.6.2 reference data.
- **Format Auto-detection Tests**: Added comprehensive tests for NetCDF, GRIB, and Zarr auto-detection in `DataLoader`.

### CI/CD & Testing
- **Verification**: Enhanced documentation builds and expanded test coverage.

### Maintenance
- **Spectral Backend Consolidation**: Evaluated `pyshtools`, `SHTns`, `ducc0`, and `jax` for performance, accuracy, and portability. Selected **ducc0** as the default backend due to its superior multi-frame performance (6.3x faster than SHTns), self-contained architecture (no external C dependencies), and near bit-wise parity with NCL for kinematics. Integrated **JAX** as an experimental alternative backend for GPU-accelerated spherical harmonic transforms.
- **Strict Typing**: Achieved 100% `mypy` compliance in core I/O modules and removed usage of `Any`.
- **Dependency Refinement**: Introduced a dedicated `zarr` optional dependency group and updated the `all` extra.

---

## v0.4.2 - 2026-03-19
### Performance
- Transitioned Dask backend to threaded scheduling for improved efficiency.
- Decoupled chunk processing to reduce memory overhead.
- Optimized default worker count to match CPU core availability (max 4).

### Features
- Added `--chunk-size` (or `-c`) CLI argument to control processing granularity.
- Refined Docker publishing logic with improved tag prioritization.

### Maintenance
- Enhanced CI/CD concurrency rules to reduce redundant build noise.

---

## v0.4.1 - 2026-03-18
> **Note:** First release available on [**Conda-Forge**](https://anaconda.org/conda-forge/pystormtracker).

### Testing
- Registered integration markers to resolve warnings in Conda-forge CI pipelines.

---

## v0.4.0 - 2026-03-14
### Architecture (Modernization)
- **Vectorized Data Model**: Massive refactor from Python-object trajectories to a high-performance, array-backed structure.
- **JIT-Optimized Kernels**: Replaced core mathematical loops with Numba-compiled kernels, achieving C-level performance.
- **Dask Integration**: Implemented end-to-end multi-processing with tree reduction for large-scale datasets.

### Features
- **GRIB Support**: Introduced support for GRIB files via the `cfgrib` engine.
- **Multi-Variable Tracking**: Support for tracking multiple variables per center using a flexible dictionary structure.

### Performance
- **Vectorized Linker**: Re-engineered the `SimpleLinker` using NumPy broadcasting for near-instant distance calculations.
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
- Optimized Docker layer caching for significantly faster image deployments.

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

### Architecture (Modernization)
- **Python 3 Migration**: Complete refactor from Python 2.7 to Python 3.10+, including type hints and modern syntax.
- **NetCDF4 Migration**: Switched from the legacy `Nio` library to `netCDF4` for robust data handling.

### Refactoring
- Extracted core logic into `simple/` and `models/` modules for better maintainability.

---

## v0.0.2 - 2018-10-25
- Added `minmaxmode` support for ERA-Interim Mean Sea Level Pressure (MSL) and Vorticity (VO) tracking.

---

## v0.0.1 - 2016-01-11
- Initial release.
- Core cyclone tracking logic based on local extrema detection.
