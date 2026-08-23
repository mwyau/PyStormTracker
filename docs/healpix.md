# HEALPix Support: Regridding and Tracking

This document describes spectral regridding to HEALPix and object detection on a one-dimensional HEALPix grid.

## 1. Overview

HEALPix represents a global field as equal-area, iso-latitude pixels with a
neighbor graph. The underlying grid is described by Górski et al. (2005),
“HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of
Data Distributed on the Sphere,” *The Astrophysical Journal*, 622(2),
759--771, https://doi.org/10.1086/427976. The one-dimensional topology avoids
the pole singularity and latitude-dependent cell areas of a regular
latitude-longitude mesh.

PyStormTracker's threshold -> connected-object -> local-feature -> refinement
pipeline is a separate adaptation of Hodges-style detection to HEALPix
topology. The HEALPix paper establishes the grid, not this detector or its
TRACK compatibility behavior.

## 2. Spectral Regridding (`SpectralRegridder`)

`SpectralRegridder` uses `ducc0.sht` for spherical harmonic analysis of an input grid and synthesis on a target grid.

- **Supported Inputs**: Clenshaw-Curtis (CC) and Gauss-Legendre (GL).
- **Supported Outputs**: CC, GL, and HEALPix.
- **Spectral Logic**:
  - **Analysis**: Extracts spherical harmonic coefficients ($a\_{lm}$) from 2D grids using `ducc0.sht.analysis_2d`.
  - **Synthesis**: Projects coefficients onto the target grid. For HEALPix, it uses `ducc0.sht.synthesis` with `geometry` parameters derived from `ducc0.healpix.Healpix_Base.sht_info()`.
  - **Spectral Truncation**: Supports explicit $L\_{max}$ and $M\_{max}$ band limits. If omitted, the truncation is inferred from the input longitude count.

## 3. HEALPix Tracking Algorithm (`HealpixTracker`)

`HealpixTracker` implements the `Tracker` protocol for HEALPix pixel arrays.
Serial, threaded Dask, and MPI backends are supported; Dask distributes frame
detection and segment MGE tasks before deterministic segment splicing.

### 3.1. 1D Graph Topology

Unlike 2D meshes where neighbors are found via index offsets, HEALPix neighbors are determined via a precomputed adjacency list.

- During initialization, the `HealpixDetector` generates a **neighbor table** of shape `(8, N_pixels)` using `ducc0.healpix.Healpix_Base.neighbors()`.
- All detection kernels operate on this 1D graph, using the table for topological lookups.

### 3.2. Connected Component Labeling (CCL)

The tracker groups adjacent pixels into objects with the Numba kernel `_label_healpix_connected_components`.

- **Algorithm**: Iterative label propagation over the 1D graph until convergence.
- **Constraints**: Supports `object_threshold` filtering and
  `min_object_grid_points` object-size constraints. The default MSL object
  threshold is `0.0` and the default relative-vorticity threshold is
  `1e-5`; these defaults are owned by the HEALPix package.

### 3.3. Quadratic Feature-point Refinement

HEALPix quadratic refinement uses the candidate pixel's neighbor ring as a
fixed, candidate-local domain and shares the intrinsic spherical-quadratic
refinement core with the regular-grid path:

1. It maps valid neighbor centers to the tangent plane at the detected center
   with the exact spherical logarithm map. Missing topology slots are ignored,
   while fewer than five valid finite samples fail explicitly.
1. It fits a center-anchored five-parameter quadratic in normalized tangent
   coordinates with SVD least squares, checks rank/conditioning, and requires
   the curvature sign appropriate to the detected minimum or maximum.
1. It accepts only stationary points inside the fixed local tangent box derived
   from the candidate's neighbor ring, maps accepted points back with the exact
   sphere exponential map, and returns the detected center on failure.

The refinement records explicit internal failure statuses for invalid
neighborhoods, singular or ill-conditioned fits, wrong curvature, and
outside-locality solutions. Longitude wrapping and signed versus unsigned
coordinate representations therefore do not change the physical result. The
intrinsic spherical quadratic combination is a PyStormTracker extension built
from established sphere log/exp maps and local least-squares fitting; it is not
a TRACK algorithm.

## 4. Engineering Standards

- **Dependencies**: HEALPix operations use the existing `ducc0` dependency; `healpy` is not required.
- **Numba JIT**: Graph traversal and connected-component kernels are compiled
  with `cache=True` and `nogil=True`; intrinsic quadratic fitting is vectorized
  through the shared spherical refinement core.
- **Defaults**: HEALPix tracking does not apply optional spectral filtering
  unless both `lmin` and `lmax` are supplied. `taper_points` is independent.
  Regridding derives a finite transform bandwidth from the source grid and
  target `nside`; that bandwidth is not an optional filter. Quadratic
  feature-point interpolation remains enabled by default. The
  `min_track_points` constructor value is applied after linking.
- **Tracker protocol and output**: `HealpixTracker` implements the common
  `Tracker` protocol and returns the array-backed `Tracks` model. Its spectral
  taper default is `0.1`; this differs from the Hodges tracker default of `1.0`.

## 5. Usage Example

```python
import xarray as xr

from pystormtracker import HealpixTracker
from pystormtracker.preprocessing import SpectralRegridder

# Regrid one ERA5 frame (CC) to HEALPix (Nside=64)
ds = xr.open_dataset("data.nc")
field = ds["msl"]
frame = field.isel({field.dims[0]: 0})
regridder = SpectralRegridder()
da_hp = regridder.to_healpix(frame, nside=64)

# Track the first eight time steps after automatic HEALPix conversion
tracker = HealpixTracker(lmin=5, lmax=42)
tracks = tracker.track(
    data=field.isel({field.dims[0]: slice(0, 8)}),
    variable="msl",
    detection_mode="min",
)
```

The first part demonstrates one-frame regridding; the second passes a regular
three-dimensional latitude-longitude field to `HealpixTracker.track()`, which
requests T5-42 filtering and HEALPix conversion. If `lmin` and `lmax` are
omitted, only the transform needed for HEALPix conversion is
performed. For an already regridded time-by-cell field, preprocessing must be
performed before tracking.
