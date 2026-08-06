# Python API Reference

PyStormTracker provides an object-oriented Python API. Tracker instances encapsulate algorithm, preprocessing, and execution configuration. The `.track()` method executes tracking on an input file or xarray dataset and returns an immutable `Tracks` container, which provides `.write()` for output serialization.

## Overview

```python
import pystormtracker as pst

# 1. Instantiate a configured tracker
tracker = pst.HodgesTracker(
    filter_lmin=5,
    filter_lmax=42,
    min_grid_points=1,
    feature_point_method="quadratic",
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
tracks.write("tracks.json")
```

### API Division

- **Constructor**: Specifies algorithm parameters, spatial/spectral preprocessing filters, projections, and backend/concurrency settings.
- **`track()`**: Specifies input data (`data`, `variable`), time window (`start_time`, `end_time`), extremum mode (`detection_mode`), intensity threshold (`intensity_threshold`), and reader engine (`engine`).
- **`Tracks.write()`**: Handles output serialization to supported formats (TrackJSON, IMILAST, NetCDF, TRACK ASCII, etc.).

## Core Exports

The package root exports only the public tracker classes and trajectory container models:

- `pst.Tracker` (Protocol)
- `pst.SimpleTracker`
- `pst.HodgesTracker`
- `pst.HealpixTracker`
- `pst.Track`
- `pst.Tracks`

Domain value types remain available under `pystormtracker.models`:

- `pystormtracker.models.Center`
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

Trackers accept preprocessing options in their constructors: `filter_lmin` and `filter_lmax` request an optional spectral filter when supplied together, while `taper_points` controls spatial tapering independently.

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

### Quadratic Feature-point Interpolation

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.refinement
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
.. automodule:: pystormtracker.metrics.tracks
   :members: compute_track_metrics
```

### CORMAX and CCA/PCA Cross-validation

```{eval-rst}
.. automodule:: pystormtracker.metrics.cross_validation
   :members:
```
