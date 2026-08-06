# PyStormTracker Roadmap

This document records in progress and implemented engineering and scientific work.

Status terms are used as follows:

- **In progress**: A subset is implemented, or reference-data and external-dataset validation remains.
- **Implemented**: The stated scope exists in the repository and has tests for that scope.

## 1. Performance and scalability

### 1.1 Prevent CPU oversubscription

**Current state:** Simple Dask uses threads and Simple MPI uses processes. Distributed preprocessing paths limit some `ducc0` thread counts, but the policy for `ducc0`, Numba, Dask, and MPI is not centralized.

**Planned work:** Define thread limits for each backend and add tests or benchmarks that record the effective thread configuration.

### 1.2 Improve `SimpleLinker` candidate search

**Current state:** `SimpleLinker` evaluates a vectorized great-circle distance matrix. Its candidate search remains $O(NM)$ for $N$ existing trajectories and $M$ detections at the next time step.

**Planned work:** Evaluate a spherical candidate index such as `scipy.spatial.cKDTree` while preserving deterministic matching and the current distance definition.

### 1.3 Manage memory pressure by chunking — 🚧 In progress

Simple Dask and MPI partition detection work and gather detections before one linking pass. Hodges can partition detection into serial time chunks and also performs one linking pass after gathering. Several preprocessing paths still load complete selected arrays into memory.

### 1.4 Array-backed data model — ✅ Implemented

Tracks, coordinates, times, identifiers, and variables are stored in flat NumPy arrays. `Track` objects provide indexed views into this storage.

### 1.5 Numba kernels — ✅ Implemented

Core local-extrema, Laplacian, connected-component labeling (CCL), geometric, quadratic feature-point interpolation, object-property, and Modified Greedy Exchange (MGE) operations are implemented as cached Numba kernels where applicable.

### 1.6 GPU preprocessing and detection

No JAX or other GPU execution backend is present in the current package dependencies. A future implementation requires a separate optional dependency group and numerical comparison against the CPU implementation.

## 2. CI, testing, and benchmarks

### 2.1 Performance regression testing

Historical benchmark results are documented, but CI does not enforce performance limits. A suitable CI benchmark requires a deterministic test dataset, repeated measurements, and criteria that account for timing variability.

### 2.2 SHTns comparison benchmark

**Description:** Restore a standalone spherical harmonic transform benchmark for reproducible accuracy and timing comparisons between SHTns and `ducc0`. SHTns is not a production transform backend or package dependency.

**Progress:** Historical comparison results remain in `docs/spectral_accuracy.md`. The executable comparison harness has not been restored.

**Verification:** Both implementations should be run on the same versioned scalar-filter and kinematic-derivative fields. The result must record grid geometry, spectral truncation, normalization, compiler and library versions, thread count, hardware, error metrics, and timings.

### 2.3 Dependency audit

Add a scheduled CI job using `uv sync --resolution lowest-direct` and the relevant test suites to check whether direct minimum versions in `pyproject.toml` remain valid.

### 2.4 Tiered testing — ✅ Implemented

Unit tests run by default. Integration tests require `--run-integration`; slow regression and full-duration backend-parity cases additionally require `--run-slow`. `--run-all` disables these collection filters.

## 3. Architecture

### 3.1 Xarray generalized ufunc integration — 🚧 In progress

Spectral filtering and kinematic calculations use `xarray.apply_ufunc` in applicable paths. Detection uses explicit Xarray loading followed by NumPy and Numba kernels. Further use of `apply_ufunc` should not change serial, Dask, or MPI results.

### 3.2 Distributed backends — 🚧 In progress

Simple supports serial, Dask, and MPI detection with Gather-then-Link orchestration. Hodges and HEALPix are serial-only and reject unsupported backend selections. Hodges applies `min_lifetime` after linking. HEALPix currently stores the same constructor value without applying lifetime pruning.

### 3.3 CLI and tracker protocol — ✅ Implemented

The `stormtracker` command has `track`, `sample`, `compare`, and `convert` subcommands. `SimpleTracker`, `HodgesTracker`, and `HealpixTracker` implement the common tracker interface. The high-level CLI implementation is in `pystormtracker.track.run_tracker`; no package-level `pystormtracker.track()` function is currently exported.

### 3.4 Remote data support — ✅ Implemented

`DataLoader` supports remote Zarr datasets over HTTP, S3, and Google Cloud Storage when the Zarr optional dependencies are installed.

## 4. Distribution and dependencies

### 4.1 Modular dependencies — ✅ Implemented

Optional extras cover MPI, GRIB, NetCDF4, Zarr, metrics, documentation, and visualization. The Hodges and HEALPix implementations and `ducc0` are core package components.

### 4.2 Conda-forge distribution — ✅ Implemented

PyStormTracker is distributed through `conda-forge` in addition to PyPI.

## 5. Scientific and technical feature roadmap

### 5.1 B-spline detection and smoothing — 🚧 In progress

**Description:** Off-grid detection using a global spherical B-spline and steepest-descent optimization can provide smoother center coordinates, particularly on lower-resolution grids such as CMIP output.

**TRACK references:** `src/spline_smooth.c`; Dierckx routines in `lib/src/`, including `sphery.f` and `smoopy.f`.

**Progress:** A `RectSphereBivariateSpline` is fitted once for each periodic global Hodges frame. Center coordinates are obtained from the local $3\\times3$ quadratic stationary-point fit. The spherical spline is evaluated at that quadratic coordinate and returned as `bspline_val` in the raw detection dictionary; it does not currently determine the center coordinate. The current Hodges linker retains only the primary tracked variable in the final `Tracks` object.

**Remaining work:** Implement direct optimization on the spherical B-spline surface and the corresponding TRACK smoothing behavior. Propagate selected refinement diagnostics through linking when they are intended as public trajectory variables.

**Verification:** Compare optimized coordinates, values, and resulting trajectories with TRACK after center optimization is implemented.

### 5.2 Regional model support with the discrete cosine transform — 🚧 In progress

**Description:** Support tracking on limited-area model fields, including WRF output, with discrete cosine transform (DCT) spectral filtering and nonperiodic domain boundaries.

**TRACK references:** `src/track.c`, `src/statspl.F`.

**Relevant paper:** **Denis, B., J. Côté, and R. Laprise**, 2002: Spectral Decomposition of Two-Dimensional Atmospheric Fields on Limited-Area Domains Using the Discrete Cosine Transform (DCT). *Mon. Wea. Rev.*, **130**, 1812–1829, [doi:10.1175/1520-0493(2002)130\<1812:SDOTDA>2.0.CO;2](https://doi.org/10.1175/1520-0493%282002%29130%3C1812%3ASDOTDA%3E2.0.CO%3B2).

**Progress:** `DCTFilter` uses the `ducc0.fft.dct` type-II/type-III transform pair. It applies a radial effective-wave-number band and an $l(l+1)$ exponential coefficient taper. `TaperFilter` supplies a separate spatial boundary taper. Automatic selection distinguishes periodic global longitude from regional longitude.

**Verification:** Repository tests cover execution and global-versus-regional selection. Numerical comparison with a limited-area reference calculation remains.

### 5.3 Spectral tapering — ✅ Implemented

**Description:** Apply a wave-number taper to spherical harmonic coefficients to reduce spectral ringing and sidelobes at the truncation boundary.

**Relevant paper:** **Sardeshmukh, P. D., and B. I. Hoskins**, 1984: Spatial Smoothing on the Sphere. *Mon. Wea. Rev.*, **112**, 2524–2529, [doi:10.1175/1520-0493(1984)112\<2524:SSOTS>2.0.CO;2](https://doi.org/10.1175/1520-0493%281984%29112%3C2524%3ASSOTS%3E2.0.CO%3B2).

**TRACK references:** `src/time_avg.c`, `src/spec_filt.c`.

**Progress:** Spherical harmonic and DCT coefficients use a configurable high-wave-number exponential taper. `TaperFilter` is a separate spatial boundary taper.

**Verification:** Spectral and tracker tests cover execution. A coefficient-level comparison with an independently calculated reference remains.

### 5.4 Postprocessing track metrics — ✅ Implemented

**Description:** Compute cyclone amplitude, cyclone frequency, track frequency, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA) on a two-dimensional latitude-longitude grid.

**Relevant paper:** **Yau, A. M. W., and E. K. M. Chang**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *J. Climate*, **33**, 10169–10186, [doi:10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1).

**Progress:** `metrics.tracks.compute_track_metrics` supports monthly or aggregate output and constant, Fisher, Cressman, linear, and quadratic spatial weighting.

**Verification:** Lagrangian metric and weighting unit tests cover the implemented estimators.

### 5.5 Eulerian metrics and weather-impact indices — ✅ Implemented

**Description:** Compute Eulerian variance measures using a 24-hour difference filter, Eddy Kinetic Energy (EKE), and high-wind percentile indices for comparison with Lagrangian track statistics.

**Relevant paper:** [Yau and Chang (2020)](https://doi.org/10.1175/JCLI-D-20-0393.1), cited in Section 5.4.

**Progress:** `metrics/eulerian.py` implements the 24-hour difference variance, EKE, and wind-speed percentile calculations.

**Verification:** Eulerian metric unit tests cover the implemented calculations.

### 5.6 CORMAX and CCA/PCA evaluation — ✅ Implemented

**Description:** CORMAX is the maximum one-point correlation between a weather-impact field and a storm-track metric within a local search region, such as a $60^\\circ\\times20^\\circ$ box. Canonical correlation analysis (CCA), optionally preceded by principal component analysis (PCA), is used to evaluate field relationships and truncation sensitivity.

**Relevant paper:** [Yau and Chang (2020)](https://doi.org/10.1175/JCLI-D-20-0393.1), cited in Section 5.4.

**Progress:** `metrics.cross_validation` implements CORMAX, leave-$n$-out CCA truncation testing, anomaly correlation coefficient (ACC), fraction of variance explained (FVE), and full-data CCA model training through the optional `xeofs` dependency.

**Verification:** Cross-validation and missing-data tests cover the implemented scope.

### 5.7 Spherical kernel gridding — ✅ Implemented

**Description:** Apply spherical weighting functions to gridded cyclone and track statistics.

**Relevant paper:** **Hodges, K. I.**, 1999: Extension of Spherical Nonparametric Estimators to Nonisotropic Kernels: An Oceanographic Application. *Mon. Wea. Rev.*, **127**, 214–227, [doi:10.1175/1520-0493(1999)127\<0214:EOSNET>2.0.CO;2](https://doi.org/10.1175/1520-0493%281999%29127%3C0214%3AEOSNET%3E2.0.CO%3B2).

**Progress:** Numba kernels support constant-radius, Fisher exponential, Cressman rational, linear, and quadratic weights. The current implementation uses isotropic distance-based kernels.

**Verification:** Weighting and gridded Lagrangian metric unit tests cover the implemented functions. Nonisotropic kernel estimation is not currently implemented.

### 5.8 Statistical distributions and confidence intervals

**Description:** Add distribution estimates, confidence intervals, and significance tests for trajectory and gridded statistics. This work is separate from spatial kernel gridding.

**Relevant paper:** **Hodges, K. I.**, 2008: Confidence Intervals and Significance Tests for Spherical Data Derived from Feature Tracking. *Mon. Wea. Rev.*, **136**, 1758–1777, [doi:10.1175/2007MWR2299.1](https://doi.org/10.1175/2007MWR2299.1).

**Verification:** Define statistical methods and versioned reference cases before implementation.

### 5.9 Object size and morphological properties — 🚧 In progress

**Description:** Calculate the area and intensity-weighted ellipse properties of each thresholded storm object.

**TRACK references:** `src/boundary_find.c`, including the `ofill` logic; `src/shape_setup.c`.

**Progress:** The Hodges detector accumulates grid-cell area over each CCL-labeled object and computes fitted area, major axis, minor axis, and orientation. Longitude is unwrapped within global objects; projected grids use planar kilometre coordinates. These diagnostics are present in raw detections but are not propagated by the current Hodges linker to the final `Tracks` object.

**Remaining work and verification:** Propagate selected object diagnostics through linking, then compare object masks and properties directly with TRACK. Existing tests cover spherical, projected, boundary, and longitude-seam cases.

### 5.10 Sampling secondary variables — ✅ Implemented

**Description:** Attach secondary variables, such as maximum wind or full-resolution minimum sea-level pressure, to trajectory centers from external NetCDF datasets.

**Progress:** `sample.py` supports nearest-neighbor and bilinear interpolation and radius-based mean, maximum, and minimum sampling.

**Verification:** Sampling unit tests and CLI integration tests cover the implemented methods.

### 5.11 Vorticity tracking CLI — 🚧 In progress

**Description:** Track an existing relative-vorticity field and expose derivation of vorticity and divergence from wind components through the CLI.

**TRACK references:** `src/compute_vorticity.c`, `src/compute_vorticity_fd.c`.

**Progress:** The tracking CLI accepts existing relative-vorticity fields. `preprocessing.kinematics` computes vorticity and divergence through the Python API. A CLI subcommand that derives these fields from wind has not been implemented.

**Verification:** Python kinematic calculations have NCL-reference integration tests. The derivation CLI requires separate tests after implementation.

### 5.12 Ensemble and dataset utilities — 🚧 In progress

**Description:** Combine track files, match trajectories between datasets or ensemble members, and perform track-to-track intercomparison.

**TRACK references:** TRACK `utils/ENSEMBLE/toverlap.c`, `trdist.c`, and
`compare_ensemble2.c`, plus ensemble matching scripts including
`GFS_SCRIPTS/eps_match.csh`.

**Progress:** `compare_tracks` and `stormtracker compare` apply whole-overlap
mean geodetic separation and overlap-fraction eligibility to exact concurrent
track sections. Each reference independently selects its closest eligible
candidate, matching the source utility's directed selection. A general
track-file combination operation has not been implemented.

**Verification:** Unit and CLI tests cover eligibility, selection, and reported
lifecycle and intensity statistics. A paired N320/0.25-degree Hodges integration
comparison covers the reduced-Gaussian vorticity workflow.

### 5.13 Storm lifecycle compositing

**Description:** Produce storm-centered composites of environmental fields, such as moisture or wind, relative to a moving cyclone center.

**Note:** Requested by K. Hodges for diagnostic analysis.

**Verification:** Define the coordinate transform, interpolation, lifecycle normalization, and reference cases before implementation.

### 5.14 Tropical cyclone specialization

**Description:** Document and test configurations, including T63 filtering, for tropical cyclone tracking.

**Relevant paper:** **Hodges, K., A. Cobb, and P. L. Vidale**, 2017: How Well Are Tropical Cyclones Represented in Reanalysis Datasets? *J. Climate*, **30**, 5243–5264, [doi:10.1175/JCLI-D-16-0557.1](https://doi.org/10.1175/JCLI-D-16-0557.1).

**Verification:** Compare with a documented tropical-cyclone reference workflow after the configuration is implemented.

### 5.15 Full Gaussian grids — 🚧 In progress

**Description:** Process and regrid fields on full Gaussian grids, including N320, using Gauss-Legendre (`GL`) geometry in `ducc0.sht.analysis_2d`.

**Progress:** `DataLoader` identifies Gaussian latitude spacing, and the SHT and regridding paths support `GL` geometry.

**Verification:** Geometry-detection and `GL` synthesis paths have repository coverage. A versioned full-Gaussian input test dataset remains to be tested end to end.

### 5.16 Reduced Gaussian grids — 🚧 In progress

**Description:** Process one-dimensional reduced Gaussian fields, such as ERA5 N320 data, using the number of longitude points on each latitude ring.

**Progress:** `DataLoader` reads `GRIB_pl` ring-size metadata. Filtering and regridding use `ducc0.sht.pseudo_analysis` with per-ring `nphi`, longitude origin, and ring offsets. Synthetic tests cover metadata, filtering, and regridding paths. Integration tests download versioned N320 mean sea level pressure and relative-vorticity GRIB test datasets, then cover loading, filtering, regridding, and tracking.

**Verification:** Repository integration tests exercise end-to-end loading,
filtering, regridding, detection, and Hodges tracking on versioned N320
test datasets. The vorticity coverage compares N320 trajectories with the paired
0.25-degree input after common-grid preprocessing; it is not a TRACK-parity or
external-validation claim.

### 5.17 Four-dimensional feature tracking (STACKER)

**Description:** Implement multidimensional feature tracking for cyclone systems.

**Relevant paper:** **Lakkis, S. G., P. Canziani, A. Yuchechen, L. Rocamora, A. Caferri, K. Hodges, and A. O'Neill**, 2019: A 4D Feature-Tracking Algorithm: A Multidimensional View of Cyclone Systems. *Quart. J. Roy. Meteor. Soc.*, **145**, 395–417, [doi:10.1002/qj.3436](https://doi.org/10.1002/qj.3436).

**Verification:** Define the multidimensional feature representation and reference workflow before implementation.
