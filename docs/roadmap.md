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

Simple, Hodges, and HEALPix partition detection work and gather or splice
results before deterministic linking. Several preprocessing paths still load
complete selected arrays into memory.

### 1.4 Array-backed data model — ✅ Implemented

Tracks, coordinates, times, identifiers, and variables are stored in flat NumPy arrays. `Track` objects provide indexed views into this storage.

### 1.5 Numba kernels — ✅ Implemented

Core local-extrema, Laplacian, connected-component labeling (CCL), geometric, quadratic feature-point interpolation, object-property, and Modified Greedy Exchange (MGE) operations are implemented as cached Numba kernels where applicable.

### 1.6 GPU preprocessing and detection

No JAX or other GPU execution backend is present in the current package dependencies. A future implementation requires a separate optional dependency group and numerical comparison against the CPU implementation.

## 2. CI, testing, and benchmarks

### 2.1 Performance regression testing

The generic benchmark runner is available, but CI does not enforce performance
limits. A suitable CI benchmark requires a deterministic test dataset, repeated
measurements, and criteria that account for timing variability.

### 2.2 External reference-data comparisons

The completed 2024 TRACK 1.5.4 comparison covers F320 → T42 and F320 → F320
using full-year 2024 ERA5 MSLP. It includes runtime measurements, raw trajectory
comparison, and RSPLICE-filtered trajectory comparison. The full-year filtered
one-to-one F1 is 0.997 for both cases.

The versioned external-data contract now exists: `tests/utils.py` pins
`PyStormTracker-Data` at `v0.2.0-data`, and release-backed assets are available
for supported ERA5 inputs, including UV850 at 0.25° and 2.5°. Individual parity
cases still require their exact versioned input and reference products. The
NCL/Spherepack kinematics comparison remains deferred because the pinned Data
release does not yet contain the required NCL-generated VODV reference fields.
The small bundled NCL T5-42 spectral output for the committed December MSL
frame remains the ordinary repository-level bounded parity test:
`tests/data/ncl/era5_msl_2025-12-01_0000_2.5x2.5_t5-42.nc`.

### 2.3 Dependency audit

Add a scheduled CI job using `uv sync --resolution lowest-direct` and the relevant test suites to check whether direct minimum versions in `pyproject.toml` remain valid.

### 2.4 Tiered testing — ✅ Implemented

Unit tests run by default from `tests/unit`. Integration and parity tests are
selected explicitly by directory; `slow` selects runtime cost and
`data` selects external contracts. Full-duration cases are
marked `slow` and are not part of the normal development suite.

## 3. Architecture

### 3.1 Xarray generalized ufunc integration — ✅ Implemented

Spectral filtering, regridding, and kinematic calculations use
`xarray.apply_ufunc` where appropriate to preserve xarray/Dask execution.
Detection intentionally operates on complete two-dimensional NumPy frames
passed to Numba kernels. This is the supported execution boundary rather than
an incomplete conversion of all kernels to `apply_ufunc`.

### 3.2 Distributed backends — ✅ Implemented

Simple, Hodges, and HEALPix support serial, Dask, and MPI execution. Dask
parallelizes independent frame work; Hodges and HEALPix additionally
parallelize MGE segments before deterministic ordered splicing. Integration
tests verify representative serial/Dask equivalence and MPI execution. Larger
worker-count and dataset-size scaling belongs to benchmarking rather than the
implementation status.

### 3.3 CLI and tracker protocol — ✅ Implemented

`pystormtracker.cli.main()` creates the top-level parser, registers the
`track`, `sample`, `compare`, and `convert` subcommands, and dispatches through
`args.func`. `pystormtracker.track.main(args)` owns setup and execution for the
`track` subcommand. `SimpleTracker`, `HodgesTracker`, and `HealpixTracker`
implement the common tracker interface.

### 3.4 Remote data support — ✅ Implemented

`DataLoader` supports remote Zarr datasets over HTTP, S3, and Google Cloud Storage when the Zarr optional dependencies are installed.

## 4. Distribution and dependencies

### 4.1 Modular dependencies — ✅ Implemented

Package extras cover EOF/CCA (`eof`), GRIB (`grib`), MPI (`mpi`), NetCDF4
(`netcdf4`), and Zarr (`zarr`); the `all` extra combines these groups.
Documentation, test, and visualization dependencies are development groups,
not package extras. The Hodges and HEALPix implementations and `ducc0` are core
package components.

### 4.2 Conda-forge distribution — ✅ Implemented

PyStormTracker is distributed through `conda-forge` in addition to PyPI.

## 5. Scientific and technical feature roadmap

### 5.1 B-spline detection and smoothing — ✅ Implemented for the supported configuration

**Description:** Off-grid detection using a global spherical B-spline and
derivative-based local optimization can provide smoother center coordinates,
particularly on lower-resolution grids such as CMIP output.

**TRACK references:** `src/spline_smooth.c`; Dierckx routines in `lib/src/`, including `sphery.f` and `smoopy.f`.

**Progress:** `feature_refinement="spherical_bspline"` now fits one periodic
global `RectSphereBivariateSpline` per Hodges frame and uses its analytical
first derivatives to determine the center coordinate. `feature_refinement="bspline"`
implements Dierckx SMOOPY `RectBivariateSpline` with native TRACK GDFP optimization, boundary
restarts, and duplicate suppression. The `bspline_feature_points`
source reference case covers synthetic extrema, including seam and high-latitude
cases, against TRACK's direct `sphery`/GDFP path. The Hodges linker preserves
the primary variable, raw grid value, and documented numeric object diagnostics
in final `Tracks`; refinement status remains detector-level diagnostics.

**Known differences:** TRACK's smoothing and polar-continuity choices are
interactive; PyStormTracker explicitly supports interpolating (`0`) smoothing
without added polar constraints. Numeric object diagnostics are preserved in
final tracks, while string refinement status remains detector-level
diagnostics rather than a trajectory variable.

**Verification:** The direct source reference case compares optimized coordinates
and values. Real-frame and trajectory probes remain additional validation work;
neither is a field-level spectral-filter comparison.

### 5.2 Regional model support with the discrete cosine transform — 🚧 In progress

**Description:** Support tracking on limited-area model fields, including WRF output, with discrete cosine transform (DCT) spectral filtering and nonperiodic domain boundaries.

**TRACK references:** `src/track.c`, `src/dfct.c` (discrete cosine transform), `src/limited_area_filter.c`.

**Relevant paper:** **Denis, B., J. Côté, and R. Laprise**, 2002: Spectral Decomposition of Two-Dimensional Atmospheric Fields on Limited-Area Domains Using the Discrete Cosine Transform (DCT). *Mon. Wea. Rev.*, **130**, 1812–1829, [doi:10.1175/1520-0493(2002)130\<1812:SDOTDA>2.0.CO;2](https://doi.org/10.1175/1520-0493%282002%29130%3C1812%3ASDOTDA%3E2.0.CO%3B2).

**Progress:** `DCTFilter` uses the `ducc0.fft.dct` type-II/type-III transform pair. It applies a radial effective-wave-number band and an $l(l+1)$ exponential coefficient taper. `TaperFilter` supplies a separate spatial boundary taper. Automatic selection distinguishes periodic global longitude from regional longitude.

**Verification:** Repository tests cover execution and global-versus-regional selection. Numerical comparison with a limited-area reference calculation remains.

### 5.3 Spectral tapering — ✅ Implemented

**Description:** Apply a wave-number taper to spherical harmonic coefficients to reduce spectral ringing and sidelobes at the truncation boundary.

**Relevant paper:** **Sardeshmukh, P. D., and B. I. Hoskins**, 1984: Spatial Smoothing on the Sphere. *Mon. Wea. Rev.*, **112**, 2524–2529, [doi:10.1175/1520-0493(1984)112\<2524:SSOTS>2.0.CO;2](https://doi.org/10.1175/1520-0493%281984%29112%3C2524%3ASSOTS%3E2.0.CO%3B2).

**TRACK references:** `src/spectral_filter.c` (filter kernel with Hoskins taper), and `src/spec_filt.c` (interactive wrapper).

**Progress:** Spherical harmonic and DCT coefficients use a configurable high-wave-number exponential taper. `TaperFilter` is a separate spatial boundary taper.

**Verification:** Spectral and tracker tests cover execution. A coefficient-level comparison with an independently calculated reference remains.

### 5.4 Postprocessing track metrics — ✅ Implemented

**Description:** Compute established cyclone amplitude, cyclone frequency, and track frequency statistics, ACA following Guo et al. (2017), and ATA introduced by Yau and Chang (2020) on a two-dimensional latitude-longitude grid. ATA now resamples each packed track to hourly values by default using local coordinate-linear latitude, shortest-wrapped longitude, and linear amplitude interpolation. The explicit `linear_pchip` PyStormTracker extension uses the same position interpolation with shape-preserving PCHIP amplitude interpolation.

**Relevant papers:** **Yau, A. M. W., and E. K. M. Chang**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *J. Climate*, **33**, 10169–10186, [doi:10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1), for ATA and the evaluation framework; Guo, Shinoda, Lin, and Chang (2017), “Variations of Northern Hemisphere Storm Track and Extratropical Cyclone Activity Associated with the Madden--Julian Oscillation,” *Journal of Climate*, 30(13), 4799--4818, https://doi.org/10.1175/JCLI-D-16-0513.1, for the ACA lineage.

**Progress:** `metrics.lagrangian.compute_track_metrics` supports monthly or aggregate output, constant-radius, Cressman, linear, and quadratic spatial weighting, and the two ATA interpolation modes `linear` and `linear_pchip`. The published paper specifies linear temporal interpolation of track positions but does not specify its coordinate geometry; the default is documented as the closest literal/simple implementation rather than as the uniquely implied geometry. PCHIP uses SciPy's `PchipInterpolator` implementation and is not attributed to Yau and Chang.

**Verification:** Lagrangian metric and weighting unit tests cover the implemented estimators and ATA interpolation cadence, knot preservation, fast-track encounter regression, antimeridian invariance, identical position interpolation between modes, and PCHIP shape preservation. A continuous cap/segment ATA formulation remains separate future work and is not part of the current implementation.

### 5.5 Eulerian metrics and weather-impact indices — ✅ Implemented

**Description:** Compute Eulerian variance measures using a 24-hour difference filter, Eddy Kinetic Energy (EKE), and high-wind percentile indices for comparison with Lagrangian track statistics.

**Relevant papers:** [Yau and Chang (2020)](https://doi.org/10.1175/JCLI-D-20-0393.1) for the weather-impact evaluation configuration; Wallace, Lim, and Blackmon (1988), “Relationship between Cyclone Tracks, Anticyclone Tracks and Baroclinic Waveguides,” *Journal of the Atmospheric Sciences*, 45(3), 439--462, https://doi.org/10.1175/1520-0469(1988)045\<0439:RBCTAT>2.0.CO;2, for the simple 24-hour difference-filter lineage.

**Progress:** `metrics/eulerian.py` implements the 24-hour difference variance, EKE, and wind-speed percentile calculations.

**Verification:** Eulerian metric unit tests cover the implemented calculations.

### 5.6 CORMAX and CCA/PCA evaluation — ✅ Implemented

**Description:** CORMAX is the maximum positive one-point correlation between a weather-impact field and a storm-track metric within a local search region, such as a $60^\\circ\\times20^\\circ$ box. PyStormTracker currently requires the two fields to use the same grid and time coordinates. Canonical correlation analysis (CCA), optionally preceded by principal component analysis (PCA), is used to evaluate field relationships and truncation sensitivity.

**Relevant paper:** [Yau and Chang (2020)](https://doi.org/10.1175/JCLI-D-20-0393.1), cited in Section 5.4.

**Progress:** `metrics.cross_validation` implements positive-only CORMAX with periodic longitude search and nonperiodic latitude shifts, leave-$n$-out CCA truncation testing, anomaly correlation coefficient (ACC), domain FVE as `1 - mean(MSE) / mean(VAR)`, and full-data CCA model training through the optional `xeofs` dependency. Dimension order is normalized, exact coordinate matches are required, and fields on different grids are rejected.

**Verification:** Cross-validation tests cover the domain-FVE aggregation, strict time/grid validation, dimension-order normalization, positive-only CORMAX, longitude wrapping, missing correlations, and all-negative search windows.

### 5.7 Spherical kernel gridding — ✅ Implemented

**Description:** Apply spherical weighting functions to gridded cyclone and track statistics.

**Relevant papers:** Hodges (1996), “Spherical Nonparametric Estimators Applied to the UGAMP Model Integration for AMIP,” *Monthly Weather Review*, 124(12), 2914--2932, https://doi.org/10.1175/1520-0493(1996)124\<2914:SNEATT>2.0.CO;2, provides relevant meteorological spherical nonparametric-estimation lineage. Cressman weighting follows Cressman (1959), “An Operational Objective Analysis System,” *Monthly Weather Review*, 87(10), 367--374, https://doi.org/10.1175/1520-0493(1959)087\<0367:AOOAS>2.0.CO;2. Linear and quadratic compact kernels are PyStormTracker generalizations.

**Progress:** Numba kernels support constant-radius, Cressman rational, linear, and quadratic weights. The current implementation uses isotropic distance-based kernels.

**Verification:** Weighting and gridded Lagrangian metric unit tests cover the implemented functions. Nonisotropic kernel estimation is not currently implemented.

### 5.8 Statistical distributions and confidence intervals

**Description:** Add distribution estimates, confidence intervals, and significance tests for trajectory and gridded statistics. This work is separate from spatial kernel gridding.

**Relevant paper:** **Hodges, K. I.**, 2008: Confidence Intervals and Significance Tests for Spherical Data Derived from Feature Tracking. *Mon. Wea. Rev.*, **136**, 1758–1777, [doi:10.1175/2007MWR2299.1](https://doi.org/10.1175/2007MWR2299.1).

**Verification:** Define statistical methods and versioned reference cases before implementation.

### 5.9 Object size and morphological properties — 🚧 In progress

**Description:** Calculate the area and intensity-weighted ellipse properties of each thresholded storm object.

**TRACK references:** `src/boundary_find.c`, including the `ofill` logic; `src/shape_setup.c`.

**Progress:** The Hodges detector accumulates grid-cell area over each CCL-labeled object and computes intensity-weighted second-moment fitted area, major axis, minor axis, and orientation. Longitude is unwrapped within global objects; projected grids use planar kilometre coordinates. The linker propagates these aligned diagnostics to final `Tracks` as `object_gridcell_area_km2` and `object_moment_*` variables.

**Remaining work and verification:** The moment-based values are intentional
PyStormTracker extensions, not TRACK's optional feature-centred anisotropy and
area workflow. Compare object masks and properties directly with a configured
TRACK anisotropy/area case before making any TRACK-equivalence claim. Existing
tests cover propagation plus spherical, projected, boundary, and
longitude-seam extension behavior.

### 5.10 Sampling secondary variables — ✅ Implemented

**Description:** Attach secondary variables, such as maximum wind or full-resolution minimum sea-level pressure, to trajectory centers from external NetCDF datasets.

**Progress:** `sample.py` supports nearest-neighbor and bilinear interpolation and radius-based mean, maximum, and minimum sampling.

**Verification:** Sampling unit tests and CLI integration tests cover the implemented methods.

### 5.11 Vorticity tracking CLI — 🚧 In progress

**Description:** Track an existing relative-vorticity field and expose derivation of vorticity and divergence from wind components through the CLI.

**TRACK references:** `src/compute_vorticity.c`, `src/compute_vorticity_fd.c`.

**Progress:** The tracking CLI accepts existing relative-vorticity fields. `preprocessing.kinematics` computes vorticity and divergence through the Python API. A CLI subcommand that derives these fields from wind has not been implemented.

**Verification:** Python kinematic calculations have constructed-input unit
coverage. The main repository retains the compact, bundled NCL T5-42 spectral
parity case for the committed December frame. Broader NCL/Spherepack kinematics
parity remains deferred until the required NCL-generated VODV reference fields
are added to the pinned external-data contract.

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
lifecycle and intensity statistics. A versioned external-data integration test
additionally compares Hodges VO850 trajectories from ERA5 N320 and 0.25°
regular-grid inputs. General track-file combination remains future work.

### 5.13 Storm lifecycle compositing

**Description:** Produce storm-centered composites of environmental fields, such as moisture or wind, relative to a moving cyclone center.

**Note:** Requested by K. Hodges for diagnostic analysis.

**Verification:** Define the coordinate transform, interpolation, lifecycle normalization, and reference cases before implementation.

### 5.14 Tropical cyclone specialization

**Description:** Document and test configurations, including T63 filtering, for tropical cyclone tracking.

**Relevant paper:** **Hodges, K., A. Cobb, and P. L. Vidale**, 2017: How Well Are Tropical Cyclones Represented in Reanalysis Datasets? *J. Climate*, **30**, 5243–5264, [doi:10.1175/JCLI-D-16-0557.1](https://doi.org/10.1175/JCLI-D-16-0557.1).

**Verification:** Compare with a documented tropical-cyclone reference workflow after the configuration is implemented.

### 5.15 Full Gaussian grids — 🚧 In progress

**Description:** Process and regrid fields on full Gaussian (Gauss-Legendre)
latitude-longitude grids using `GL` geometry in `ducc0.sht.analysis_2d`.

**Progress:** `DataLoader` identifies Gaussian latitude spacing, and the SHT and regridding paths support `GL` geometry.

**Verification:** Geometry-detection and `GL` synthesis paths have repository coverage. A versioned full-Gaussian input test dataset remains to be tested end to end.

### 5.16 Reduced Gaussian grids — ✅ Implemented

**Description:** Process one-dimensional reduced Gaussian fields, such as ERA5 N320 data, using the number of longitude points on each latitude ring.

**Progress:** `DataLoader` reads `GRIB_pl` ring-size metadata. Filtering and
regridding use `ducc0.sht.pseudo_analysis` with per-ring `nphi`, longitude
origin, and ring offsets.

**Verification:** Synthetic tests cover reduced-Gaussian metadata and numerical
paths. Versioned `v0.2.0-data` ERA5 N320 integration coverage exercises real
MSLP and VO850 loading, filtering/regridding, tracking, and the paired
N320-versus-0.25° Hodges vorticity comparison. The paired trajectory
comparison remains marked `data`/`slow`; that affects routine CI selection,
not implementation status.

### 5.17 Four-dimensional feature tracking (STACKER)

**Description:** Implement multidimensional feature tracking for cyclone systems.

**Relevant paper:** **Lakkis, S. G., P. Canziani, A. Yuchechen, L. Rocamora, A. Caferri, K. Hodges, and A. O'Neill**, 2019: A 4D Feature-Tracking Algorithm: A Multidimensional View of Cyclone Systems. *Quart. J. Roy. Meteor. Soc.*, **145**, 395–417, [doi:10.1002/qj.3436](https://doi.org/10.1002/qj.3436).

**Verification:** Define the multidimensional feature representation and reference workflow before implementation.
