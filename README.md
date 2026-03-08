# PyStormTracker

[![DOI](https://zenodo.org/badge/36328800.svg)](https://doi.org/10.5281/zenodo.18764813)
[![PyPI version](https://img.shields.io/pypi/v/PyStormTracker)](https://pypi.org/project/PyStormTracker/)
[![Docker](https://img.shields.io/badge/docker-xddd%2Fpystormtracker-blue?logo=docker)](https://hub.docker.com/r/xddd/pystormtracker)
[![GHCR](https://img.shields.io/badge/ghcr.io-XDDD%2Fpystormtracker-blue?logo=github)](https://github.com/orgs/XDDD/packages/container/package/pystormtracker)
[![CI](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml/badge.svg)](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/mwyau/PyStormTracker/graph/badge.svg?token=JmTabGA3cq)](https://codecov.io/github/mwyau/PyStormTracker)
![License](https://img.shields.io/pypi/l/PyStormTracker)
![Python versions](https://img.shields.io/pypi/pyversions/PyStormTracker)

PyStormTracker provides the implementation of the "Simple Tracker" algorithm used for cyclone trajectory analysis in **Yau and Chang (2020)**. It was originally developed at the **National Center for Atmospheric Research (NCAR)** as part of the **2015 Summer Internships in Parallel Computational Science (SIParCS)** program, utilizing a task-parallel strategy with temporal decomposition and a tree reduction algorithm to process large climate datasets.

## Features

- **Modern Python Support**: Strictly targets **Python 3.11+** with comprehensive type hints and 100% strict `mypy` compliance.
- **Xarray Integrated**: Fully migrated to `xarray` and `h5netcdf` for robust, high-performance coordinate-aware I/O and lazy data loading.
- **Parallel Backends**:
  - **Dask (Default)**: Automatically scales to all available CPU cores on local machines.
  - **MPI**: Supports distributed execution via `mpi4py`.
  - **Serial**: Standard sequential execution for smaller datasets or debugging.
- **Robust Detection**: Optimized $O(N)$ extrema filtering and robust handling of masked/missing data.
- **CI/CD Integrated**: Automated linting, type-checking, and code coverage reporting via GitHub Actions.
- **Standardized Output**: Results are exported to the IMILAST intercomparison format (.txt) with readable datetime strings and formatted numeric values.

## Technical Methodology

PyStormTracker treats meteorological fields as 2D images and leverages `scipy.ndimage` for robust feature detection:

- **Local Extrema Detection**: Uses an optimized sliding window filter to identify local minima (cyclones) or maxima (anticyclones/vorticity).
- **Intensity & Refinement**: Applies the **Laplacian operator** (`laplace`) to measure the "sharpness" of the field at each detected center point. This is used to resolve duplicates and ensure only the most physically intense point is kept when multiple adjacent pixels are flagged.
- **Spherical Continuity**: Utilizes `mode='wrap'` for all filters to correctly handle periodic boundaries across the Prime Meridian, enabling seamless tracking across the entire globe.
- **Heuristic Linking**: Implements a nearest-neighbor linking strategy to connect detected centers into trajectories across successive time steps.

## Installation

### Prerequisites
- Python 3.11+
- (Optional) OpenMPI for MPI support.

### From PyPI (Recommended)
You can install the latest stable version of PyStormTracker directly from PyPI:
```bash
pip install PyStormTracker
```

### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/mwyau/PyStormTracker.git
   cd PyStormTracker
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

Once installed, you can use the `stormtracker` command directly:

```bash
stormtracker -i era5_msl_2.5x2.5.nc -v msl -o my_tracks
```

### Command Line Arguments

| Argument | Short | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required.** Path to the input NetCDF file. |
| `--var` | `-v` | **Required.** Variable name to track (e.g., `msl`, `vo`). |
| `--output` | `-o` | **Required.** Path to the output track file (appends `.txt` if missing). |
| `--num" | `-n` | Number of time steps to process. |
| `--mode` | `-m` | `min` (default) for low pressure, `max` for vorticity/high pressure. |
| `--backend` | `-b` | `dask` (default), `serial`, or `mpi`. |
| `--workers` | `-w` | Number of Dask workers (defaults to CPU core count). |

## Development

### Setup
It is recommended to use a virtual environment or conda for development:
```bash
conda create -n pst python=3.11
conda activate pst
pip install -e .[dev]
```

### Quality Control
Before submitting a pull request, please ensure all checks pass:

**Linting & Formatting:**
```bash
ruff check . --fix
ruff format .
```

**Type Checking:**
```bash
mypy src/
```

### Tiered Testing
To keep development cycles fast, testing is tiered:
- **Fast Tests**: Default local runs (skips slow tests).
- **Slow Tests**: ONLY long-running integration/regression tests.
- **Full Suite**: Everything (used in CI).

**Run fast unit tests only (Default):**
```bash
pytest
```

**Run ONLY slow/integration tests:**
```bash
pytest --run-slow
```

**Run everything:**
```bash
pytest --run-all
```

## Citations

If you use this software in your research, please cite the following:

- **Yau, A. M. W., and E. K. M. Chang**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *J. Climate*, **33**, 10169–10186, https://doi.org/10.1175/JCLI-D-20-0393.1.   

- **Yau, A. M. W.**, 2026: mwyau/PyStormTracker. *Zenodo*, https://doi.org/10.5281/zenodo.18764813.

## References

 - **Yau, A. M. W., K. Paul and J. Dennis**, 2016: PyStormTracker: A Parallel Object-Oriented Cyclone Tracker in Python. *96th American Meteorological Society Annual Meeting*, New Orleans, LA. *Zenodo*, https://doi.org/10.5281/zenodo.18868625.
 - **Neu, U., et al.**, 2013: IMILAST: A Community Effort to Intercompare Extratropical Cyclone Detection and Tracking Algorithms. *Bull. Amer. Meteor. Soc.*, **94**, 529–547, https://doi.org/10.1175/BAMS-D-11-00154.1.
 - **IMILAST Intercomparison Protocol**: https://proclim.scnat.ch/en/activities/project_imilast/intercomparison
 - **IMILAST Data Download**: https://proclim.scnat.ch/en/activities/project_imilast/data_download

## License

This project is licensed under the terms found in the `LICENSE` file.
