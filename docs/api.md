# Python API Reference

PyStormTracker provides an object-oriented Python API. Tracker instances encapsulate algorithm, preprocessing, and execution configuration. The `.track()` method executes tracking on an input file or xarray dataset and returns an immutable `Tracks` container, which provides `.write()` for output serialization.

## Overview

```python
import pystormtracker as pst

# 1. Instantiate a configured tracker
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

# 2. Run tracking on input data
tracks = tracker.track(
    "input.nc",
    "vo",
    start_time="2000-01-01",
    end_time="2000-01-31",
    detection_mode="max",
)

# 3. Write results
tracks.write("tracks.trackjson")
```

### API Division

- **Constructor**: Specifies algorithm parameters, spatial/spectral preprocessing filters, projections, and backend/concurrency settings.
- **`track()`**: Specifies input data (`data`, `variable`), time window (`start_time`, `end_time`), extremum mode (`detection_mode`), thresholds (`feature_threshold` for Simple, `object_threshold` for Hodges and HEALPix), and reader engine (`engine`).
- **`Tracks.write()`**: Handles output serialization to supported formats (`json`, `track`, `imilast`).

For `HodgesTracker`, `frame_workers` controls concurrent frame tasks,
`sht_threads` controls DUCC0 threads per active spherical-harmonic transform,
and `mge_workers` controls concurrent MGE segment tasks. `segment_frames`
remains the scientific temporal segment length and is independent of these
controls.

## Core Exports

The package root exports only the public tracker classes, domain entities, and format functions:

- `pst.Tracker` (Protocol)
- `pst.SimpleTracker`
- `pst.HodgesTracker`
- `pst.HealpixTracker`
- `pst.Center`
- `pst.Track`
- `pst.Tracks`
- `pst.load_tracks`
- `pst.save_tracks`

Domain value types remain available under `pystormtracker.models`:

- `pystormtracker.models.Center`
- `pystormtracker.models.CenterFrame`
- `pystormtracker.models.SpatialBounds`
- `pystormtracker.models.ProcessingStep`
- `pystormtracker.models.TracksMetadata`

## Trackers

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

## Core Models

### Tracks

```{eval-rst}
.. automodule:: pystormtracker.models.tracks
   :members: Track, Tracks
   :show-inheritance:
```

### Storm Centers

```{eval-rst}
.. automodule:: pystormtracker.models.center
   :members: Center
   :show-inheritance:
```

## Data Loader

```{eval-rst}
.. automodule:: pystormtracker.io.data_loader
   :members: DataLoader
```

## Preprocessing

Trackers accept preprocessing options in their constructors: `lmin` and `lmax` request an optional spectral filter when supplied together, while `taper_points` controls spatial tapering independently. `HodgesTracker` accepts `spectral_taper` with source-compatible default `1.0`; `HealpixTracker` owns a separate default of `0.1`. Hodges' default `feature_refinement="bspline"` uses TRACK/SMOOPY-compatible rectangular B-spline refinement with coordinate-space GDFP optimization. The advanced experimental `"spherical_bspline"` option uses a global spherical B-spline with a candidate-local feasible region on eligible periodic latitude-longitude frames. `"quadratic"` and `"spherical_quadratic"` provide local polynomial subgrid refinement, and `"grid"` disables subgrid refinement. `missing_frame_parameters` and `time_step` model known input-time gaps.

### Kinematics (vorticity and divergence)

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.kinematics
   :members:
```

### Spectral Filtering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.spectral
   :members:
```

### Regridding

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.regrid
   :members:
```

### Feature Refinement

```{eval-rst}
.. automodule:: pystormtracker.refinement
   :members:
```

### Spatial Tapering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.taper
   :members:
```

## Post-processing and Analysis

### Secondary-variable Sampling

```{eval-rst}
.. automodule:: pystormtracker.sample
   :members: sample_tracks
```

### Track Matching

```{eval-rst}
.. automodule:: pystormtracker.metrics.compare
   :members: compare_tracks, TrackComparisonConfig, TrackComparison, TrackMatch
```

### Spherical Weighting Kernels

```{eval-rst}
.. automodule:: pystormtracker.metrics.weighting
   :members:
```

### Eulerian Track Metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.eulerian
   :members:
```

### Lagrangian Track Metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.lagrangian
   :members: compute_track_metrics
```

### Hodges Splice Filtering

```{eval-rst}
.. automodule:: pystormtracker.hodges.rsplice
   :members: filter_rsplice
```

### CORMAX and CCA/PCA Cross-validation

```{eval-rst}
.. automodule:: pystormtracker.metrics.cross_validation
   :members:
```
