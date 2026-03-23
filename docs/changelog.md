# Changelog

## v0.5.0.dev
### Features
- **Hodges (TRACK) Algorithm**: Full implementation of the Hodges tracking algorithm with algorithmic parity to the original TRACK software. Includes object-based detection (CCL), spherical cost functions, adaptive constraints, and recursive MGE optimization.
- **New Preprocessing Module**: Introduced a dedicated module for spherical harmonic filtering and tapering. Initially implemented with `pyshtools` and subsequently migrated to the high-performance `shtns` library for improved scalability.
- **Interactive Visualization**: Added a Jupyter Notebook and **Storm Track Explorer** for interactive trajectory analysis and visualization.
- **Backend Auto-Detection**: Improved logic for automatically selecting the most efficient execution backend (Serial, Dask, or MPI) based on the environment.

### Performance
- **SHTns Optimization**: Implemented thread-local caching and explicit thread control for spherical harmonic transforms. This resolves performance bottlenecks and prevents CPU oversubscription in multi-process workloads.
- **Architecture-Aware Builds**: Configured CI to use generic x86-64 and ARM64 CFLAGS, resolving "Illegal instruction" errors on certain hardware while maintaining high performance.

### CI/CD & Testing
- **New Documentation Build**: Added a verification step to the CI pipeline to ensure documentation builds successfully before merging.
- **Improved Linting**: Migrated to `ruff-action v3` for faster automated code quality checks.
- **Expanded Test Coverage**: Added comprehensive unit tests for I/O, utility functions, and preprocessing modules.

### Maintenance
- **Repository Organization**: Centralized core documentation in the `docs/` directory and cleaned up legacy metadata and unused data files.
- **Dependency Management**: Refined optional dependency groups (`hodges`, `pre`, `viz`).

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
