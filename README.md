# PyStormTracker

[![CI](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml/badge.svg)](https://github.com/mwyau/PyStormTracker/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/pystormtracker/badge/?version=latest)](https://pystormtracker.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/mwyau/PyStormTracker/graph/badge.svg?token=JmTabGA3cq)](https://codecov.io/github/mwyau/PyStormTracker)
[![PyPI version](https://img.shields.io/pypi/v/PyStormTracker)](https://pypi.org/project/PyStormTracker/)
![Python versions](https://img.shields.io/pypi/pyversions/PyStormTracker)
![License](https://img.shields.io/pypi/l/PyStormTracker)
[![DOI](https://zenodo.org/badge/36328800.svg)](https://doi.org/10.5281/zenodo.18764813)

**PyStormTracker** is a high-performance Python package for cyclone trajectory analysis. It implements the "Simple Tracker" algorithm and provides a scalable framework for processing large-scale climate datasets like ERA5.

The project is currently being expanded to include a Python port of the adaptive constraints tracking algorithm from **Hodges (1999)** and Accumulated Track Activity metrics.

## Features

- **High-Performance Architecture**: Uses an **Array-Backed** data model to eliminate Python object overhead and ensure zero-copy serialization during parallel execution.
- **JIT-Optimized Kernels**: Core mathematical filters are implemented in **Numba**, running at raw C speeds while releasing the GIL for true multi-process execution.
- **Xarray Native**: Seamlessly handles NetCDF and GRIB formats with coordinate-aware processing and robust variable alias handling (e.g., `msl`/`slp`, `lon`/`longitude`).
- **Scalable Backends**: 
  - **Serial (Default)**: Standard sequential execution.
  - **Dask**: Multi-process tree-reduction for local or distributed scaling.
  - **MPI**: High-performance distributed execution via `mpi4py`.
- **Typed & Modern**: Built for **Python 3.11+** with strict type safety and `mypy` compliance.
- **Interoperable**: Full support for the standard **IMILAST** intercomparison format.

## Installation

```bash
# Using uv (recommended)
uv add PyStormTracker

# Using pip
pip install PyStormTracker
```

## Usage

### Command Line Interface

Once installed, you can use the `stormtracker` command directly:

```bash
stormtracker -i data.nc -v msl -o my_tracks
```

#### Command Line Arguments

| Argument | Short | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required.** Path to the input NetCDF/GRIB file. |
| `--var` | `-v` | **Required.** Variable name to track (e.g., `msl`, `vo`). |
| `--output` | `-o` | **Required.** Path to the output track file (appends `.txt` if missing). |
| `--num` | `-n` | Number of time steps to process. |
| `--mode` | `-m` | `min` (default) for low pressure, `max` for vorticity/high pressure. |
| `--backend` | `-b` | `serial` (default), `dask`, or `mpi`. |
| `--workers` | `-w` | Number of Dask workers (defaults to CPU core count). |
| `--engine` | `-e` | Xarray engine (e.g., `h5netcdf`, `netcdf4`, `cfgrib`). |

### Python API

You can easily integrate PyStormTracker into your own scripts or Jupyter Notebooks:

```python
import pystormtracker as pst

# 1. Instantiate the tracker (defaults to Serial backend)
tracker = pst.SimpleTracker()

# 2. Run the tracking algorithm. Returns an array-backed Tracks object.
tracks = tracker.track(
    infile="data.nc", 
    varname="msl", 
    mode="min",
    start_time="2025-01-01",   # Optional: limit by start date
    end_time="2025-01-31",     # Optional: limit by end date
    backend="dask",            # Optional: use 'serial', 'dask', or 'mpi'
    n_workers=4
)

# 3. Analyze the results programmatically
for track in tracks:
    if len(track) >= 8:
        print(f"Track {track.track_id} lived for {len(track)} steps.")

# 4. Export results
tracks.write("output.txt", format="imilast")
```

## Sample Data

Sample datasets for testing and benchmarking are hosted in the [PyStormTracker-Data](https://github.com/mwyau/PyStormTracker-Data) repository.

## Citations

If you use this software in your research, please cite the following:

- **Yau, A. M. W. and Chang, E. K. M.**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. *J. Climate*, **33**, 10169–10186.
- **Hodges, K. I.**, 1999: Adaptive Constraints for Feature Tracking. *Mon. Wea. Rev.*, **127**, 1362–1373.

## License

BSD-3-Clause
