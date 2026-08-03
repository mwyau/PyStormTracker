# PyStormTracker Architecture

This document describes the data model, tracker interfaces, preprocessing, execution paths, testing, and release workflows in PyStormTracker.

## 1. Architecture Principles

The architecture has four main features:

1. **Unified API (`Tracker` protocol):** A structural interface allows the CLI and Python API to use `SimpleTracker`, `HodgesTracker`, and `HealpixTracker` through common orchestration code.
2. **Centralized constants:** Standard constants such as radius of earth, and detection thresholds, are defined in `models/constants.py` and shared by trackers.
3. **Vectorization and Numba JIT:** Numerical operations use NumPy broadcasting and cached, GIL-free Numba kernels where suitable. Detection and linking kernels avoid Python loops where practical.
4. **Gather-then-Link:** Simple Dask and MPI execution distribute detection, gather raw detections in time order, and run one linking pass.

## 2. Core Components

### 2.1 Array-Backed Data Models (`Tracks`, `Track`, `Center`)

The data models use contiguous array-backed storage:

- **`Tracks`:** The immutable central container holding per-track `ids` and
  half-open `offsets`, plus aligned one-dimensional `times`, `lats`, `lons`,
  and meteorological variable arrays. For example, `ids=[10, 20, 35]` and
  `offsets=[0, 4, 7, 12]` represent points `[0:4]`, `[4:7]`, and `[7:12]`.
- **`Track`:** A lightweight view storing its parent and packed track position;
  its `point_slice` is derived directly from adjacent offsets.
- **`Center`:** A dataclass used during cyclone-center detection.

Mutation occurs in `TracksBuilder`. Finalized arrays are native-endian,
C-contiguous, and read-only. Point columns are aligned by position; a
point-level ID array is available only through the explicit
`Tracks.point_track_ids()` derived operation.

This layout avoids one persistent Python object per center, reduces allocation overhead, and supports NumPy selection, broadcasting, and serialization between workers.

### 2.2 Shared `DataLoader`

Data loading is encapsulated in `DataLoader` (`io/data_loader.py`). It handles:

- **Formats:** NetCDF through `h5netcdf` or `netCDF4`, GRIB through `cfgrib`, and Zarr through `zarr`.
- **Remote data:** Zarr stores over HTTP, S3, and Google Cloud Storage through `fsspec` when the optional Zarr dependencies are installed.
- **Variable and coordinate mapping:** Common field aliases such as `msl` and `slp`, and latitude, longitude, and time coordinate aliases.
- **Grid metadata:** Regular latitude-longitude, full Gaussian, reduced Gaussian, projected `x/y`, and HEALPix coordinates, including metadata required by spherical harmonic transforms and map projections.

### 2.3 Heuristic Tracking (`SimpleTracker`)

The Simple tracker constructs great-circle distance matrices from clamped unit-vector dot products using NumPy broadcasting. Candidate centers are sorted lexicographically before deterministic greedy nearest-neighbor matching between consecutive time steps.

### 2.4 Optimization-Based Tracking (`HodgesTracker`)

`HodgesTracker` implements methods based on TRACK (Hodges 1994, 1995, 1999):

- **Object-based detection:** Thresholding, connected-component labeling, object filtering, and local-extrema detection.
- **Modified Greedy Exchange (MGE):** Iterative exchange of points between tracks to reduce the total cost function.
- **Spherical cost function:** Penalties for changes in tangent direction and displacement magnitude between consecutive track segments.
- **Adaptive constraints:** Regional and displacement-dependent maximum-displacement and smoothness limits.
- **Sub-grid refinement:** Local quadratic fitting around eligible extrema. A spherical spline value may also be evaluated on periodic global grids.

The detailed algorithmic requirements and TRACK comparison scope are documented in [Hodges Tracking](hodges.md).

### 2.5 Parallel Pipeline (Gather-then-Link)

Simple parallel execution uses the following sequence:

1. Assigned time chunks are distributed across Dask or MPI workers.
2. Each worker runs frame-local detection kernels and returns raw coordinate arrays.
3. The main process gathers detections in global time order.
4. One sequential linking pass constructs the tracks.

Detection can be partitioned without making link decisions at chunk boundaries. Centralized linking avoids merging independently linked chunks. Repository integration tests compare complete serial, Dask, and MPI Simple outputs using versioned test data.

Hodges supports serial chunked detection followed by one linking pass. Hodges and HEALPix do not currently provide Dask or MPI tracking.

## 3. Command-Line Interface Architecture

PyStormTracker uses `argparse` subcommands.

### 3.1 Modular Router Pattern

`cli.py` is a thin entry point. It delegates argument definition and execution to focused modules:

- **`track.py`:** Detection and trajectory construction.
- **`sample.py`:** Sampling secondary variables, such as precipitation, along tracks.
- **`compare.py`:** Intercomparison and matching of track datasets.
- **`convert.py`:** Output conversion and supported visualization generation.

### 3.2 Decoupled Analysis Commands

Tracking, secondary-variable sampling, track comparison, and format conversion are separate commands. A track file can be reused for sampling or comparison without rerunning detection and trajectory linking on the original field.

## 4. The `Tracker` Protocol

The `Tracker` protocol is defined in `src/pystormtracker/models/tracker.py` and provides a common interface for tracking algorithms:

```python
import pystormtracker as pst

tracker = pst.SimpleTracker()

tracks = tracker.track(
    infile="era5_msl.nc",
    varname="msl",
    start_time="2025-01-01",
    backend="dask",
)

tracks.write("output.txt", format="imilast")
```

Tracker implementations retain algorithm-specific defaults while accepting shared orchestration arguments. The high-level API and CLI pass algorithm-specific options through `**kwargs` where required.

## 5. Testing and Validation

The test suite has several levels:

- **Unit tests:** Models, numerical kernels, geometry, parsing, and isolated behavior.
- **Integration tests:** Data loading, complete tracker paths, optional dependencies, parallel backends, and output formats.
- **Slow integration tests:** Larger scientific comparisons and more expensive end-to-end cases.
- **Package smoke tests:** Wheel and source-distribution builds installed with standard `pip` in clean environments.
- **Container smoke tests:** CLI startup, package import, `cfgrib` self-check, and vulnerability scanning of the built image.

`uv` manages development and CI environments. Package smoke tests use `pip` because they test the installed distribution as users receive it.

The minimum-direct-dependency test resolves with `uv sync --resolution lowest-direct` and runs with `uv run --no-sync` so the environment is not replaced by the normal lockfile resolution.

Parallel results must be deterministic for the tested scope. Serial, Dask, and MPI Simple outputs are compared as complete track results rather than only by counts or summary statistics.

## 6. Continuous Integration and Publishing

### 6.1 CI Triggers and Duplicate Runs

The `CI` workflow runs for:

- pull requests, including each new commit pushed to an open pull request;
- pushes to `main`;
- version tags matching the broad `v*` trigger;
- manual workflow dispatch.

Ordinary feature-branch pushes do not run CI independently. This prevents a branch push and its pull request from running the same CI suite twice. Concurrency cancellation stops superseded pull-request and `main` runs. Tag runs are not cancelled.

### 6.2 Pull-Request and Full Test Suites

Pull requests execute a representative suite:

- code-quality and type checks;
- Python 3.14 tests on Ubuntu and macOS, Python 3.13
  tests on Windows, Python 3.11 minimum-direct-dependency unit tests, and
  Ubuntu AMD64 integration tests without the slow marker;
- wheel installation, documentation, and dependency review;
- a local AMD64 Docker build, smoke test, and vulnerability scan after the
  non-Docker suite succeeds.

Pushes to `main`, version tags, and manual CI runs execute the full suite:

- supported Python versions, including minimum-direct-dependency and free-threaded tests;
- Ubuntu AMD64 and ARM64, Ubuntu 24.04 and 26.04 compatibility, Windows AMD64, and macOS ARM64 integration tests. Windows 2025 uses Python 3.13 because `eccodes` is not available with Python 3.14;
- slow integration tests;
- wheel and source-distribution installation tests;
- native AMD64 and ARM64 test-image builds, except that manual CI runs omit
  Docker work.

Pull-request Docker jobs do not receive registry credentials or push images.
CI-originated native test images are built only after the full non-Docker suite
succeeds.

### 6.3 Tested Docker Images

The reusable `Docker Build` workflow runs local AMD64 validation for CI pull
requests after their non-Docker suite succeeds. For `main` and version-tag CI
runs, it instead builds native test images on:

- `ubuntu-26.04` for `linux/amd64`;
- `ubuntu-26.04-arm` for `linux/arm64`.

Each platform image is pushed to the private repository
`docker.io/mwyau/pystormtracker` under a `<seven-character-commit>-<architecture>`
tag. CI pulls and tests the exact pushed digest. After both platform jobs succeed,
CI creates one private multi-platform test manifest tagged
`<seven-character-commit>`.

Pull-request Docker builds are local and do not receive registry credentials.
The `Docker Build` workflow also supports manual dispatch for a selected ref; it
builds, tests, and pushes the same private multi-platform test manifest. Manual
CI dispatch runs the non-Docker full suite only.

### 6.4 Docker Publishing

After the test manifest is created, CI calls the reusable `Docker Publish`
workflow to promote it without rebuilding. The source commit's seven-character SHA
identifies the test image. Publication applies the following tags:

| Source | `mwyau/pystormtracker` | `xddd/pystormtracker` |
| --- | --- | --- |
| `main` | seven-character SHA, `<seven-character-SHA>` | `edge`, `<seven-character-SHA>` |
| stable tag `v0.6.0` | seven-character SHA, `<seven-character-SHA>` | `0.6.0`, `0.6`, `latest`, `<seven-character-SHA>` |
| manual private promotion | seven-character SHA, `<seven-character-SHA>` | none |

The completed Docker Hub manifests are copied to the corresponding GHCR repositories without rebuilding. Public `edge` and release manifests are attested on Docker Hub only; GHCR copies are not separately attested.

Manual dispatch can promote an existing tested image tag through the `private`, `edge`, or `release` channel.

The publisher contains no recovery build path. If a private test image has been
deleted, manually dispatch `Docker Build` for the corresponding ref to rebuild and
retest it before publication. This keeps image construction and testing in the
Docker build workflow and keeps the publishing workflow limited to validation,
tagging, copying, and attestation.

Release tags are validated with an explicit stable-version expression equivalent to `vX.Y.Z`. The broad workflow trigger is not treated as version validation.

### 6.5 Python Publishing and Dependency Updates

CI validates that a version tag points to a commit reachable from `main` and that
the package version matches the tag without its leading `v`. It then calls the
reusable Python publishing workflow with the distributions built and package-tested
in the same CI run. The reusable workflow has no manual-dispatch trigger.

Stable tags matching `vX.Y.Z` publish to PyPI after the protected `pypi`
environment approves the job. Development tags matching `vX.Y.Z.devN` publish to
TestPyPI after the protected `testpypi` environment approves the job. Development
tags do not publish Docker images. Docker `edge` follows the newest successful
`main` CI run, and Docker `latest` changes only during a successful stable-tag
release. Docker republishing does not republish the Python package.

Dependabot checks `uv`, GitHub Actions, and Docker dependencies daily. Minor and patch Python updates and GitHub Actions updates may be grouped to reduce pull-request volume. Major Python updates and Docker image updates remain separate for diagnosis.

## 7. Planned Architecture Work

Current planned work includes:

- **Xarray generalized ufuncs:** Spectral filtering and kinematic derivatives use `xr.apply_ufunc(..., dask="parallelized")`; detection still uses explicit time partitioning.
- **Lazy evaluation and thread topology:** Reduce eager frame loading and define `ducc0` and Numba thread counts when Dask threads or MPI processes distribute work.
- **Spatial indexing:** Evaluate `scipy.spatial.cKDTree` or another spherical candidate index to reduce the $O(N \times M)$ candidate-search cost in dense Simple linking workloads.
- **Additional backends:** Hodges and HEALPix parallel tracking require Gather-then-Link implementations and serial-parallel equality tests.

For planned features, see the [Roadmap](roadmap.md).

## 8. Performance Benchmarks

The benchmark page records Simple tracker timings for versions `v0.3.3` and `v0.4.0`.

Detailed execution timings (breaking down Detection, Linking, Export, and I/O Overhead) across Serial, Dask, and MPI backends for both standard and high-resolution ERA5 datasets are available in the [Benchmark Report](benchmark.md).

## Appendix: Evolution from Legacy Architecture

| Feature | Legacy Architecture (v0.0.2) | Current Architecture (v0.4.0+) |
| --- | --- | --- |
| **Data storage** | Nested lists of `Center` and `Track` objects. | Immutable packed C-contiguous NumPy columns with IDs and offsets. |
| **Parallelism** | Threaded tree reduction. | Parallel detection followed by centralized linking. |
| **Linking strategy** | Tree reduction across chunks. | Ordered detection gathering and one linking pass with serial-equality tests. |
| **Linker** | $O(N^2)$ nested Python loops. | Vectorized great-circle distance matrices and deterministic greedy matching. |
| **Algorithms** | Simple only. | Simple, Hodges-based, and HEALPix trackers. |
| **I/O** | Many small lazy-loaded chunks. | Xarray reads coordinated through `DataLoader`. |
