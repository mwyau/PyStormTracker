# API Reference

This page lists the principal Python modules and public classes used by PyStormTracker. The command-line orchestration function is available as `pystormtracker.track.run_tracker`.

## Tracking orchestration

```{eval-rst}
.. automodule:: pystormtracker.track
   :members: run_tracker
```

## Trackers

```{eval-rst}
.. automodule:: pystormtracker.simple.tracker
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pystormtracker.hodges.tracker
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pystormtracker.healpix.tracker
   :members:
   :undoc-members:
   :show-inheritance:
```

## Detectors

```{eval-rst}
.. automodule:: pystormtracker.simple.detector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pystormtracker.hodges.detector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pystormtracker.healpix.detector
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core models

### Tracks

```{eval-rst}
.. automodule:: pystormtracker.models.tracks
   :members:
   :undoc-members:
   :show-inheritance:
```

### Storm centers

```{eval-rst}
.. automodule:: pystormtracker.models.center
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data loader

```{eval-rst}
.. automodule:: pystormtracker.io.data_loader
   :members: DataLoader
```

## Preprocessing

### Kinematics (vorticity and divergence)

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.kinematics
   :members:
```

### Spectral filtering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.spectral
   :members:
```

### Regridding

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.regrid
   :members:
```

### Sub-grid refinement

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.refinement
   :members:
```

### Spatial tapering

```{eval-rst}
.. automodule:: pystormtracker.preprocessing.taper
   :members:
```

## Post-processing and analysis

### Secondary-variable sampling

```{eval-rst}
.. automodule:: pystormtracker.sample
   :members: sample_tracks
```

### Track matching

```{eval-rst}
.. automodule:: pystormtracker.metrics.compare
   :members: compare_tracks, TrackComparisonConfig, TrackComparison, TrackMatch
```

### Spherical weighting kernels

```{eval-rst}
.. automodule:: pystormtracker.metrics.weighting
   :members:
```

### Eulerian track metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.eulerian
   :members:
```

### Lagrangian track metrics

```{eval-rst}
.. automodule:: pystormtracker.metrics.tracks
   :members: compute_track_metrics
```

### CORMAX and CCA/PCA cross-validation

```{eval-rst}
.. automodule:: pystormtracker.metrics.cross_validation
   :members:
```
