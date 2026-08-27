# Python API Reference

PyStormTracker provides an object-oriented Python API. Tracker instances
encapsulate algorithm, preprocessing, and execution configuration. The
`.track()` method executes tracking on an input file or xarray dataset and
returns an immutable `Tracks` container, which provides `.write()` for output
serialization.

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

## API policy

PyStormTracker is a pre-1.0 project with a deliberately small compatibility
surface. Python importability alone does not make a name supported. The
`__all__` declarations, package re-exports, and the explicitly documented
members below define the intended API.

### Primary API

The preferred interface is the package root:

```python
import pystormtracker as pst
```

The root exports exactly these nine names:

- `pst.Center`
- `pst.HealpixTracker`
- `pst.HodgesTracker`
- `pst.SimpleTracker`
- `pst.Track`
- `pst.Tracker`
- `pst.Tracks`
- `pst.load_tracks`
- `pst.save_tracks`

`pystormtracker.__version__` remains normally accessible, but is not part of
the root `__all__` compatibility list.

### Advanced supported APIs

The following package and direct-module APIs are supported when explicitly
documented here or in the linked methodology pages:

- `pystormtracker.models`: domain types `Center`, `DetectionMode`,
  `ProcessingStep`, `Projection`, `ResolvedDetectionMode`, `SpatialBounds`,
  `Track`, `Tracker`, `Tracks`, and `TracksMetadata`.
- `pystormtracker.io`: generic `DataLoader`, `SUPPORTED_FORMATS`,
  `SupportedFormat`, `infer_format`, `load_tracks`, and `save_tracks`.
- `pystormtracker.preprocessing`: `BoundaryTaper`, `DCTFilter`, `SHTFilter`,
  `SpectralRegridder`, and `compute_vorticity_divergence`.
- `pystormtracker.metrics`: the high-level metric functions and the explicit
  comparison API documented below.
- `pystormtracker.sample.sample_tracks`.
- `pystormtracker.hodges.rsplice.filter_rsplice`.

TRACK-specific `read_track`, `write_track`, and `TrackNumericTime` remain
available from `pystormtracker.io.track` for deliberate advanced use. The
provisional TrackJSON wire structs and encoder remain in
`pystormtracker.io.trackjson`, but are not re-exported by generic
`pystormtracker.io`.

### Implementation modules

Detector, linker, MGE, segment, execution-helper, refinement, CLI-routing,
schema-generation, and TrackJSON wire-structure internals are not a general
compatibility surface. Feature-point refinement is selected through tracker
configuration, for example `feature_refinement="grid"`,
`"quadratic"`, `"spherical_quadratic"`, `"bspline"`, or
`"spherical_bspline"`; the low-level refinement structures remain in their
implementation modules.

Modules not listed here may be importable but are implementation details and
are not part of the supported compatibility surface. This project makes no
semantic-version stability claim for unlisted names while it remains pre-1.0.

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

## Domain models

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

`CenterFrame` is an internal frame-processing transport type. It is not a
public model or part of the Tracker return contract.

## Generic I/O

```{eval-rst}
.. automodule:: pystormtracker.io.data_loader
   :members: DataLoader

.. automodule:: pystormtracker.io.format
   :members: SUPPORTED_FORMATS, SupportedFormat, infer_format, load_tracks, save_tracks
```

Use `load_tracks` and `save_tracks` for ordinary format handling. TrackJSON/1.0
is provisional; its typed wire structures and direct encoder are documented in
the [TrackJSON reference](trackjson.md), not promoted to the generic I/O
namespace.

## Scientific preprocessing

Tracker constructors accept preprocessing options: `lmin` and `lmax` request an
optional spectral filter when supplied together, while `taper_points` controls
spatial tapering independently. `HodgesTracker` accepts `spectral_taper` with
source-compatible default `1.0`; `HealpixTracker` owns a separate default of
`0.1`. The Hodges default `feature_refinement="bspline"` uses the
TRACK/SMOOPY-compatible rectangular B-spline path with coordinate-space GDFP
optimization. The advanced experimental `"spherical_bspline"` option uses a
global spherical B-spline with a candidate-local feasible region on eligible
periodic latitude-longitude frames. `"quadratic"` and
`"spherical_quadratic"` provide local polynomial alternatives, and `"grid"`
disables subgrid refinement. `missing_frame_parameters` and `time_step` model
known input-time gaps for Hodges tracking.

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

The HEALPix example uses the supported package import:

```python
from pystormtracker.preprocessing import SpectralRegridder
```

## Metrics and post-processing

### Secondary-variable sampling

```{eval-rst}
.. automodule:: pystormtracker.sample
   :members: sample_tracks
```

### Track comparison

The comparison result uses `TrackProperties` for lifecycle and intensity
summaries and `IntensityDifference` for pointwise candidate-minus-reference
statistics. These result structures are part of the documented comparison
module, but are not re-exported through `pystormtracker.metrics`.

```{eval-rst}
.. automodule:: pystormtracker.metrics.compare
   :members: compare_tracks, TrackComparisonConfig, TrackComparison, TrackMatch, TrackProperties, IntensityDifference
```

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

Spherical weighting helpers are implementation details of the scientific metric
workflows and are not documented as standalone convenience APIs.

### Hodges splice filtering

```{eval-rst}
.. automodule:: pystormtracker.hodges.rsplice
   :members: filter_rsplice
```
