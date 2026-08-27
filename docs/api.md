# Python API Reference

PyStormTracker provides tracker classes, packed trajectory models, scientific
preprocessing, feature-point refinement, track I/O, and analysis functions.
Tracker instances combine algorithm, preprocessing, and execution settings.
The `.track()` method accepts a file path or xarray object and returns an
immutable `Tracks` object. `Tracks.write()` serializes the result.

## Overview

```python
import pystormtracker as pst

tracker = pst.HodgesTracker(
    lmin=5,
    lmax=42,
    min_object_grid_points=1,
    feature_refinement="spherical_bspline",
    backend="dask",
    frame_workers=4,
    sht_threads=4,
    mge_workers=16,
)

tracks = tracker.track(
    "input.nc",
    "vo",
    start_time="2000-01-01",
    end_time="2000-01-31",
    detection_mode="max",
)

tracks.write("tracks.trackjson")
```

For `HodgesTracker`, `frame_workers` controls concurrent frame tasks,
`sht_threads` controls DUCC0 threads per active spherical-harmonic transform,
and `mge_workers` controls concurrent MGE segment tasks. `segment_frames`
controls the scientific temporal segment length independently of these
execution controls.

## Package API

The package root provides the main tracking classes, result models, and track
I/O functions:

- `pystormtracker.Center`
- `pystormtracker.HealpixTracker`
- `pystormtracker.HodgesTracker`
- `pystormtracker.SimpleTracker`
- `pystormtracker.Track`
- `pystormtracker.Tracker`
- `pystormtracker.Tracks`
- `pystormtracker.load_tracks`
- `pystormtracker.save_tracks`

Algorithm-specific convenience classes are available from their subpackages:

```python
from pystormtracker.healpix import HealpixDetector, HealpixTracker
from pystormtracker.hodges import HodgesTracker
from pystormtracker.simple import SimpleDetector, SimpleLinker, SimpleTracker
```

## Tracker classes

```{eval-rst}
.. automodule:: pystormtracker.simple.tracker
   :members: SimpleTracker
   :show-inheritance:

.. automodule:: pystormtracker.hodges.tracker
   :members: HodgesTracker
   :show-inheritance:

.. automodule:: pystormtracker.healpix.tracker
   :members: HealpixTracker
   :show-inheritance:
```

## Models

The `pystormtracker.models` package contains domain and result types:
`Center`, `DetectionMode`, `ProcessingStep`, `Projection`,
`ResolvedDetectionMode`, `SpatialBounds`, `Track`, `Tracker`, `Tracks`, and
`TracksMetadata`.

### Tracks and metadata

```{eval-rst}
.. automodule:: pystormtracker.models.tracks
   :members: DetectionMode, ProcessingStep, ResolvedDetectionMode, Track, Tracks, TracksMetadata
   :show-inheritance:
```

### Storm centers

```{eval-rst}
.. automodule:: pystormtracker.models.center
   :members: Center
   :show-inheritance:
```

### Geographic domain types

```{eval-rst}
.. automodule:: pystormtracker.models.geo
   :members: Projection, SpatialBounds
   :show-inheritance:
```

### Tracker protocol

```{eval-rst}
.. automodule:: pystormtracker.models.tracker
   :members: Tracker
```

## I/O

Generic format handling is provided by `DataLoader`, `SUPPORTED_FORMATS`,
`SupportedFormat`, `infer_format`, `load_tracks`, and `save_tracks`.

```{eval-rst}
.. automodule:: pystormtracker.io.data_loader
   :members: DataLoader

.. automodule:: pystormtracker.io.format
   :members: SUPPORTED_FORMATS, SupportedFormat, infer_format, load_tracks, save_tracks
```

TrackJSON-specific functions and wire structures are provided by
`pystormtracker.io.trackjson`; see the [TrackJSON reference](trackjson.md).
TRACK text I/O is provided by `pystormtracker.io.track`.

## Preprocessing

The preprocessing package provides filters, regridding, spatial tapering, and
spherical wind diagnostics. Tracker constructors use `lmin` and `lmax` for an
optional spectral filter and `taper_points` for spatial tapering. Projection or
HEALPix conversion can require a transform bandwidth even when no optional
filter is requested.
`HodgesTracker` uses a `spectral_taper` default of `1.0`; `HealpixTracker` uses
`0.1`. Hodges missing-frame handling uses `missing_frame_parameters` and
`time_step` when the input cadence contains known gaps.

### Vorticity and divergence

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.kinematics
   :members: compute_vorticity_divergence
```

### Spectral filtering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.spectral
   :members: DCTFilter, SHTFilter
```

### Regridding

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.regrid
   :members: SpectralRegridder
```

### Spatial tapering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.taper
   :members: BoundaryTaper
```

## Feature refinement

Feature-point refinement is selected with the `feature_refinement` tracker
option. `SimpleTracker` and `HealpixTracker` support `grid` and `quadratic`.
`HodgesTracker` also supports `spherical_quadratic`, `bspline`, and
`spherical_bspline`; its default is `bspline`.

The rectangular `bspline` method uses the TRACK/SMOOPY-compatible surface and
coordinate-space GDFP optimization. `spherical_bspline` fits a global spline
on the sphere and constrains optimization to the candidate neighborhood.

The refinement package provides operations for constructing spline surfaces and
locating feature points:

- `build_bspline_surface` and `build_spherical_bspline_surface` construct
  frame-level surfaces and return named results containing `surface` and
  `status`.
- `refine_bspline_feature_point` and
  `refine_spherical_bspline_feature_point` return a
  `BsplineRefinementResult` with latitude, longitude, value, and status.
- `refine_quadratic_feature_point` and `refine_quadratic_feature_points` return
  refined coordinates and values for regular-grid candidates.
- `refine_spherical_quadratic_feature_points` returns a
  `SphericalQuadraticRefinementBatch` containing refined coordinates, values,
  status codes, and numerical diagnostics. Use
  `spherical_quadratic_status_name` to decode a status code.

```{eval-rst}
.. automodule:: pystormtracker.refinement.bspline
   :members: BsplineRefinementResult, BsplineSurface, BsplineSurfaceResult, SphericalBsplineSurface, SphericalBsplineSurfaceResult, build_bspline_surface, build_spherical_bspline_surface, refine_bspline_feature_point, refine_spherical_bspline_feature_point

.. automodule:: pystormtracker.refinement.quadratic
   :members: SphericalQuadraticRefinementBatch, refine_quadratic_feature_point, refine_quadratic_feature_points, refine_spherical_quadratic_feature_points, spherical_quadratic_status_name
```

## Metrics and post-processing

### Secondary-variable sampling

```{eval-rst}
.. automodule:: pystormtracker.sample
   :members: sample_tracks
```

### Track comparison

```{eval-rst}
.. automodule:: pystormtracker.metrics.compare
   :members: compare_tracks, TrackComparisonConfig, TrackComparison, TrackMatch, TrackProperties, IntensityDifference
```

`compare_tracks` returns lifecycle and intensity summaries together with the
track matches.

### Eulerian metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.eulerian
   :members: compute_eke, compute_high_wind_index, compute_variance_metric
```

### Lagrangian metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.lagrangian
   :members: compute_track_metrics
```

### CORMAX and CCA cross-validation

```{eval-rst}
.. automodule:: pystormtracker.metrics.cross_validation
   :members: compute_cormax, find_best_cca_truncation, train_cca_model
```

### Hodges splice filtering

```{eval-rst}
.. automodule:: pystormtracker.hodges.rsplice
   :members: filter_rsplice
```
