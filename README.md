# PyStormTracker

[![DOI](https://zenodo.org/badge/36328800.svg)](https://doi.org/10.5281/zenodo.18764813)

PyStormTracker provides the implementation of the "Simple Tracker" algorithm used for cyclone trajectory analysis in **Yau and Chang (2020)**. It was originally developed at the **National Center for Atmospheric Research (NCAR)** as part of the **2015 Summer Internships in Parallel Computational Science (SIParCS)** program, utilizing a task-parallel strategy with temporal decomposition and a tree reduction algorithm to process large climate datasets.

## Features

- **Modern Python 3 Support**: Fully migrated from Python 2 with comprehensive type hints.
- **Flexible Data Support**: Works with `netCDF4` and handles various coordinate naming conventions (`lat`/`lon` vs `latitude`/`longitude`).
- **Parallel Backends**:
  - **Dask (Default)**: Automatically scales to all available CPU cores on local machines.
  - **MPI**: Supports distributed execution via `mpi4py`.
  - **Serial**: Standard sequential execution for smaller datasets or debugging.
- **Robust Detection**: Handles masked/missing data correctly and includes automated unit/integration tests.
- **User-Friendly Output**: Results are exported directly to CSV with readable datetime strings and formatted numeric values.

## Technical Methodology

PyStormTracker treats meteorological fields as 2D images and leverages `scipy.ndimage` for robust feature detection:

- **Local Extrema Detection**: Uses `generic_filter` with a sliding window (default 5x5) to identify local minima (cyclones) or maxima (anticyclones/vorticity).
- **Intensity & Refinement**: Applies the **Laplacian operator** (`laplace`) to measure the "sharpness" of the field at each detected center point. This is used to resolve duplicates and ensure only the most physically intense point is kept when multiple adjacent pixels are flagged.
- **Spherical Continuity**: Utilizes `mode='wrap'` for all filters to correctly handle periodic boundaries across the Prime Meridian, enabling seamless tracking across the entire globe.
- **Heuristic Linking**: Implements a nearest-neighbor linking strategy to connect detected centers into trajectories across successive time steps.

## Installation

### Prerequisites
- Python 3.10+
- (Optional) MS-MPI or OpenMPI for MPI support.

### Setup
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
stormtracker -i data/slp.2012.nc -v slp -o my_tracks -n 120
```

### Command Line Arguments

| Argument | Short | Description |
| :--- | :--- | :--- |
| `--input` | `-i` | **Required.** Path to the input NetCDF file. |
| `--var` | `-v` | **Required.** Variable name to track (e.g., `slp`, `vo`). |
| `--output` | `-o` | **Required.** Path to the output CSV file (appends `.csv` if missing). |
| `--num` | `-n` | Number of time steps to process. |
| `--mode` | `-m` | `min` (default) for low pressure, `max` for vorticity/high pressure. |
| `--backend` | `-b` | `dask` (default), `serial`, or `mpi`. |
| `--workers` | `-w` | Number of Dask workers (defaults to CPU core count). |

### Examples

**Run with Dask (Auto-detected cores):**
```bash
stormtracker -i input.nc -v slp -o tracks
```

**Run with MPI (Distributed):**
```bash
mpiexec -n 4 stormtracker -i input.nc -v slp -o tracks -b mpi
```

## Project Structure

- `src/pystormtracker/models/`: Data structures (`Center`, `Grid`, `Tracks`).
- `src/pystormtracker/simple/`: Implementation of the Simple Tracker logic (`SimpleDetector`, `SimpleLinker`).
- `src/pystormtracker/stormtracker.py`: CLI orchestration and parallel backends.

## Testing

Run the full test suite (unit and integration tests) using `pytest`:

```bash
pytest
```

## Citations

If you use this software in your research, please cite the following:

- **Yau, A. M. W., and E. K. M. Chang, 2020**: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *Journal of Climate*, **33**, 8413–8462, [https://doi.org/10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1).

- **Yau, A. M. W., K. Paul, and J. Dennis, 2016**: PyStormTracker: A Parallel Object-Oriented Cyclone Tracker in Python. *96th AMS Annual Meeting*, New Orleans, LA. [Abstract](https://ams.confex.com/ams/96Annual/webprogram/Paper290042.html), [Poster PDF](https://www.mwyau.com/pystormtracker.pdf). Originally developed during NCAR SIParCS 2015.

## License

This project is licensed under the terms found in the `LICENSE` file.
