# PyStormTracker

[![CI](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml/badge.svg)](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/pystormtracker/badge/?version=latest)](https://pystormtracker.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/mwyau/PyStormTracker/graph/badge.svg?token=JmTabGA3cq)](https://codecov.io/github/mwyau/PyStormTracker)
[![PyPI version](https://img.shields.io/pypi/v/PyStormTracker)](https://pypi.org/project/PyStormTracker/)
[![TestPyPI Version](https://img.shields.io/pypi/v/PyStormTracker?pypiBaseUrl=https://test.pypi.org&label=testpypi)](https://test.pypi.org/project/PyStormTracker/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pystormtracker)](https://anaconda.org/channels/conda-forge/packages/pystormtracker/overview)
[![GitHub License](https://img.shields.io/github/license/mwyau/PyStormTracker)](https://github.com/mwyau/PyStormTracker/blob/main/LICENSE)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/PyStormTracker)](https://pypi.org/project/PyStormTracker/)
[![Docker](https://img.shields.io/badge/docker-xddd%2Fpystormtracker-blue?logo=docker)](https://hub.docker.com/r/xddd/pystormtracker)
[![GHCR](https://img.shields.io/badge/ghcr.io-xddd%2Fpystormtracker-blue?logo=github)](https://github.com/orgs/xddd/packages/container/package/pystormtracker)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18764813-blue.svg)](https://doi.org/10.5281/zenodo.18764813)

**PyStormTracker** is a Python package for cyclone trajectory analysis. It provides cyclone detection, trajectory construction, and track-based analysis for meteorological and climate datasets. The package includes a Numba Simple Tracker implementation with high-level concept lineage from **Yau and Chang (2020)** and TRACK-compatible algorithms with scientific lineage from **Hodges (1994, 1995, 1999)**. The project was initially developed at the **National Center for Atmospheric Research (NCAR)** during the **2015 SIParCS** program.

## Features

- Simple, TRACK-compatible Hodges, and HEALPix tracking.
- NetCDF, GRIB, Zarr, TrackJSON, TRACK tdump, and IMILAST support across regular, Gaussian, reduced-Gaussian, projected, and HEALPix grids.
- Spectral preprocessing, feature refinement, track comparison, variable sampling, and storm-track metrics.
- Serial, Dask, and MPI execution with Numba-accelerated numerical kernels.

## Quick start

PyStormTracker requires **Python 3.12+**.

```bash
pip install PyStormTracker

stormtracker track \
    -i data.nc \
    --variable msl \
    -o tracks.trackjson \
    -m min \
    -a hodges
```

Python API:

```python
import pystormtracker as pst

tracker = pst.HodgesTracker()
tracks = tracker.track(data="data.nc", variable="msl", detection_mode="min")
```

See the [Quickstart](docs/quickstart.md) for installation options, CLI and Python usage, formats, sample data, and development commands.

## Documentation

[Documentation](https://pystormtracker.readthedocs.io/) · [Quickstart](docs/quickstart.md) · [CLI](docs/cli.md) · [API](docs/api.md) · [Architecture](docs/architecture.md) · [Hodges / TRACK](docs/hodges.md) · [HEALPix](docs/healpix.md) · [TrackJSON](docs/trackjson.md) · [Benchmarks](docs/benchmark.md)

## Citations

If you use PyStormTracker in research, please cite the software:

- **Yau, A. M. W.**, 2026: *PyStormTracker: A High-Performance Cyclone Tracker in Python*. Zenodo, [doi:10.5281/zenodo.18764813](https://doi.org/10.5281/zenodo.18764813).

PyStormTracker was originally presented as:

- **Yau, A. M. W., K. Paul, and J. Dennis**, 2016: PyStormTracker: A Parallel Object-Oriented Cyclone Tracker in Python. *96th American Meteorological Society Annual Meeting*, New Orleans, LA. Zenodo, [doi:10.5281/zenodo.18868625](https://doi.org/10.5281/zenodo.18868625).

For methods used by the Simple tracker and storm-track analysis:

- **Yau, A. M. W., and E. K. M. Chang**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *J. Climate*, **33**, 10169–10186, [doi:10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1).

For the feature-identification and trajectory-linking methods implemented by `HodgesTracker`:

- **Hodges, K. I.**, 1994: A General Method for Tracking Analysis and Its Application to Meteorological Data. *Mon. Wea. Rev.*, **122**, 2573–2586, [doi:10.1175/1520-0493(1994)122\<2573:AGMFTA>2.0.CO;2](https://doi.org/10.1175/1520-0493%281994%29122%3C2573%3AAGMFTA%3E2.0.CO%3B2).
- **Hodges, K. I.**, 1995: Feature Tracking on the Unit Sphere. *Mon. Wea. Rev.*, **123**, 3458–3465, [doi:10.1175/1520-0493(1995)123\<3458:FTOTUS>2.0.CO;2](https://doi.org/10.1175/1520-0493%281995%29123%3C3458%3AFTOTUS%3E2.0.CO%3B2).
- **Hodges, K. I.**, 1999: Adaptive Constraints for Feature Tracking. *Mon. Wea. Rev.*, **127**, 1362–1373, [doi:10.1175/1520-0493(1999)127\<1362:ACFFT>2.0.CO;2](https://doi.org/10.1175/1520-0493%281999%29127%3C1362%3AACFFT%3E2.0.CO%3B2).

PyStormTracker uses `ducc0` for spherical-harmonic transforms and related spherical numerical operations:

- **Reinecke, M.**, 2020: DUCC: Distinctly Useful Code Collection. *Astrophysics Source Code Library*, record [ascl:2008.023](https://ascl.net/2008.023).

For the IMILAST cyclone-tracking intercomparison:

- **Neu, U., et al.**, 2013: IMILAST: A Community Effort to Intercompare Extratropical Cyclone Detection and Tracking Algorithms. *Bull. Amer. Meteor. Soc.*, **94**, 529–547, [doi:10.1175/BAMS-D-11-00154.1](https://doi.org/10.1175/BAMS-D-11-00154.1).

Additional method-specific references are given in the relevant documentation.

## License

PyStormTracker is distributed under the [BSD 3-Clause License](LICENSE).
