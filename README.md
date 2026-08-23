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

- **Vectorized, array-backed data model**: Stores track coordinates, times, identifiers, and variables in contiguous NumPy arrays. `Track` objects are views over this storage rather than independent collections of center objects.
- **Numba JIT-compiled kernels**: Detection, Laplacian intensity, great-circle geometry, connected-component labeling (CCL), quadratic feature-point interpolation, and Modified Greedy Exchange (MGE) kernels use cached, GIL-free Numba functions.
- **Multiple Algorithms**:
  - **Simple**: Fast, local-extrema detection and deterministic nearest-neighbor linking.
  - **TRACK compatibility (`HodgesTracker`)**: Thresholded object detection with connected-component labeling (CCL), spherical cost functions, adaptive constraints, and iterative Modified Greedy Exchange (MGE) linking based on TRACK. See the [Hodges implementation documentation](docs/hodges.md).
  - **HEALPix**: Thresholded object detection on a one-dimensional HEALPix neighbor graph, followed by the Hodges MGE linker.
- **Coordinate-aware Xarray input**: `DataLoader` opens NetCDF, GRIB, and Zarr data, resolves common variable and coordinate aliases, and identifies regular latitude-longitude, full Gaussian, reduced-Gaussian, projected, and HEALPix grids.
- **Execution Backends**:
  - **Simple**: Serial, threaded Dask detection, and MPI detection. Parallel paths gather detections before one linking pass.
  - **Hodges and HEALPix**: Serial, threaded Dask, and MPI execution with deterministic segment splicing.
- **Typed Implementation**: Built for **Python 3.12+** with strict type safety and `mypy` compliance.
- **Formats and analysis**: Reads IMILAST and TrackJSON track data; writes IMILAST, TRACK tdump, and TrackJSON. Analysis functions include secondary-variable sampling, track matching, gridded cyclone and track metrics, Eulerian variance and wind indices, CORMAX, and CCA/PCA truncation cross-validation.

## Technical Methodology

The trackers apply the following stages:

- **Preprocessing**: Optional spherical harmonic transform (SHT) filtering on global grids, discrete cosine transform (DCT) filtering on regional grids, Sardeshmukh & Hoskins spectral tapering, and regridding to polar stereographic or HEALPix coordinates. Full and reduced Gaussian grids are handled through `ducc0` geometry metadata.
- **Detection**: The Simple tracker applies a sliding-window local-extrema filter and uses the discrete Laplacian magnitude to select among adjacent extrema. Hodges and HEALPix use thresholding, connected-component labeling, object filtering, and local-extrema detection.
- **Feature-point location**: `HodgesTracker` supports `"grid"`, `"quadratic"`, `"spherical_quadratic"`, `"bspline"`, and `"spherical_bspline"`, and defaults to `"bspline"`: a TRACK SMOOPY-compatible rectangular B-spline with coordinate-space GDFP optimization. `"spherical_quadratic"` and `"spherical_bspline"` are advanced experimental spherical options. `SimpleTracker` supports `"grid"` and `"quadratic"` and defaults to `"grid"`; `HealpixTracker` supports the same two choices and defaults to `"quadratic"`. The supported TRACK source-reference baseline is [TRACK 1.5.4](https://gitlab.act.reading.ac.uk/track/track/-/tree/TRACK-1.5.4).
- **Linking**: Simple uses deterministic nearest-neighbor linking with a vectorized great-circle distance matrix. Hodges and HEALPix use Modified Greedy Exchange with spherical displacement and smoothness constraints.

## Documentation

Full documentation, including API references and advanced usage examples, is available at [pystormtracker.readthedocs.io](https://pystormtracker.readthedocs.io/).

## Installation

### Prerequisites

- **Python 3.12+**
- **Message Passing Interface (MPI)**:
  - **Linux/macOS**: `OpenMPI` is recommended and included as a development dependency.
  - **Windows**: Use `winget install -e --id Microsoft.msmpi` (recommended) or [MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
- **Spherical Harmonic Transform (SHT) engine**:
  - `ducc0` provides scalar and spin-weighted spherical harmonic transforms, reduced-grid synthesis, and HEALPix geometry.
- **Free-threaded Python**:
  - Python 3.14 free-threaded support is experimental. CI excludes `eof`, `grib`, and `zarr`: `xeofs` requires a `statsmodels` source build on 3.14t, while `eccodeslib` has no compatible free-threaded distribution for GRIB support.

### From PyPI

You can install the latest stable version of PyStormTracker directly from PyPI:

Using `pip`:

```bash
# Standard installation
pip install PyStormTracker

# With optional components
pip install "PyStormTracker[mpi]"     # Includes mpi4py for distributed execution
pip install "PyStormTracker[grib]"    # Includes GRIB support
pip install "PyStormTracker[netcdf4]" # Includes NetCDF4 backend
pip install "PyStormTracker[zarr]"    # Includes Zarr support (with remote HTTP/S3/GS)
pip install "PyStormTracker[eof]"     # Includes xeofs for CCA/PCA analysis
pip install "PyStormTracker[all]"     # Includes non-visualization optional components
```

Using `uv`:

```bash
# For use as a CLI tool
uv tool install "PyStormTracker[mpi]"

# For use as a library in your project
uv add "PyStormTracker[mpi]"
```

### From Conda-Forge

You can also install PyStormTracker from `conda-forge`:

Using `mamba`:

```bash
mamba install -c conda-forge pystormtracker
```

Using `conda`:

```bash
conda install -c conda-forge pystormtracker
```

### From Source

Install with `uv`:

```bash
git clone https://github.com/mwyau/PyStormTracker.git
cd PyStormTracker
uv sync
```

## Usage

### Command Line Interface

Once installed, the `stormtracker` command provides separate subcommands for tracking, sampling, comparison, and conversion:

#### 1. Track Features

Run the core storm tracking algorithm (e.g., tracking cyclones in MSLP):

```bash
stormtracker track -i data.nc --variable msl -o tracks.trackjson -m min -a hodges -f json
```

#### 2. Sample Variables

Extract external variables (e.g., precipitation) along existing tracks:

```bash
# Calculate mean precipitation within a 500km radius of storm centers
stormtracker sample -i tracks.trackjson -d precip.nc --variable pr -o tracks_enriched.trackjson --method mean --radius 500
```

#### 3. Match and Intercompare

Compare tracks from different datasets or ensemble members:

```bash
# Match tracks from two sources with a 2.0 degree mean distance threshold
stormtracker compare -r era5.trackjson -c gfs.trackjson -s 2.0 -l 0.6 --variable vo -m max --json
```

#### 4. Convert

Convert between supported trajectory formats:

```bash
stormtracker convert -i tracks.trackjson -o tracks.imilast -F imilast
```

#### CLI Argument Reference

Use `stormtracker <command> --help` for detailed argument lists. `-v` and
`-vv` select INFO and DEBUG logging before or after any subcommand; `-V`
prints the version. `--variable` is long-only on every subcommand. Key options
for the `track` command include:

| Argument               | Short | Description                                                                                                                                                                                                                                                                                                                                                                                                                      |
| :--------------------- | :---- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input`              | `-i`  | Path to the input NetCDF/GRIB file.                                                                                                                                                                                                                                                                                                                                                                                              |
| `--variable`           |       | Variable name for the command (e.g., `msl`, `vo`); no short alias.                                                                                                                                                                                                                                                                                                                                                               |
| `-v`, `--verbose`      |       | Increase operational logging (`-v`: INFO, `-vv`: DEBUG).                                                                                                                                                                                                                                                                                                                                                                         |
| `--output`             | `-o`  | Path to the output track file.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `--algorithm`          | `-a`  | `simple` (default), `hodges`, or `healpix`.                                                                                                                                                                                                                                                                                                                                                                                      |
| `--format`             | `-f`  | Output format: `auto`, `json`, `track`, or `imilast`; recognized extensions are inferred automatically.                                                                                                                                                                                                                                                                                                                          |
| `--detection-mode`     | `-m`  | `auto` (default), `min`, or `max`; known aliases resolve automatically.                                                                                                                                                                                                                                                                                                                                                          |
| `--backend`            | `-b`  | `serial`, `dask`, or `mpi`. Local Dask is supported by the implemented tracker paths; availability of MPI depends on the selected tracker and installed MPI support.                                                                                                                                                                                                                                                             |
| `--workers`            | `-w`  | Number of parallel workers.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `--lmin`, `--lmax`     |       | Optional spectral filter bounds. Supply both to apply a filter; omit both to leave the native field unchanged.                                                                                                                                                                                                                                                                                                                   |
| `--taper-points`       |       | Independent spatial taper width; zero disables tapering.                                                                                                                                                                                                                                                                                                                                                                         |
| `--spectral-taper`     |       | Hodges/HEALPix high-wave-number coefficient taper; defaults are `1.0` for Hodges and `0.1` for HEALPix.                                                                                                                                                                                                                                                                                                                          |
| `--nside`              |       | Target HEALPix resolution; omitted values are derived from the source grid.                                                                                                                                                                                                                                                                                                                                                      |
| `--feature-refinement` |       | Tracker-dependent feature-point location method. Hodges accepts `grid`, `quadratic`, `spherical_quadratic`, `bspline`, and `spherical_bspline`; `bspline` is the default TRACK/SMOOPY-compatible rectangular B-spline path, while the two spherical methods are advanced experimental options. Simple accepts `grid` and `quadratic` and defaults to `grid`; HEALPix accepts `grid` and `quadratic` and defaults to `quadratic`. |
| `--no-progress`        |       | Disable the interactive Hodges Dask progress display. It is otherwise enabled when standard error is a terminal.                                                                                                                                                                                                                                                                                                                 |

### Python API

The trackers can also be called directly:

```python
import pystormtracker as pst

tracker = pst.HodgesTracker()

tracks = tracker.track(data="data.nc", variable="vo", detection_mode="max")
```

### Analyze the results programmatically

```python
for track in tracks:
    if len(track) >= 8:
        print(f"Track {track.track_id} lived for {len(track)} steps.")
```

### Export results

```python
tracks.write("output.txt", format="imilast")
```

## Sample Data

The checkout retains one ordinary integration input:
`tests/data/era5/era5_msl_2025-12_2.5x2.5.nc`. Specialized GRIB,
reduced-Gaussian, and broader reference datasets are owned by the pinned
[PyStormTracker-Data](https://github.com/mwyau/PyStormTracker-Data) contract.
Small references use direct raw-Git paths, large files use direct Release
filenames, and Git-tracked Zarr stores use their raw store paths. The one
bundled numerical parity reference is
`tests/data/ncl/era5_msl_2025-12-01_0000_2.5x2.5_t5-42.nc`; its generation
methodology is maintained in `PyStormTracker-Validation`.

## Development

### Setup

Using `uv` to set up your development environment:

```bash
# Install dependencies and sync virtual environment
uv sync --all-extras
```

### Quality Control

Run automated checks using `uv run`:

**Linting & Formatting:**

```bash
uv run ruff check . --fix
uv run ruff format .
```

**Type Checking:**

```bash
uv run mypy
```

### Tiered Testing

To keep development cycles fast, testing is tiered:

- **Fast Tests**: Default local unit runs.
- **Integration Tests**: Select `tests/integration/` explicitly; use `-m "not slow and not data"` for current local coverage.
- **Parity Tests**: `tests/parity/` contains strict end-to-end trajectory comparisons and the one bundled NCL/Spherepack T5-42 spectral numerical-parity case. Historical trajectory and broader reference cases use the pinned Data-repository paths when their external assets are available.
- **Scientific Validation**: TRACK source stages, MGE replay, and reconciliation material live in the sibling `PyStormTracker-Validation` repository; broader NCL/Spherepack reference data live in `PyStormTracker-Data`.
- **Full Suite**: Select `tests/unit tests/integration tests/parity`.

**Run fast unit tests only (Default):**

```bash
uv run pytest
```

**Run non-slow integration tests:**

```bash
uv run pytest tests/integration -m "not slow and not data"
```

**Run everything:**

```bash
uv run pytest tests/unit tests/integration tests/parity
```

## Citations

If you use this software in your research, please cite the following:

- **Yau, A. M. W.**, 2026: *mwyau/PyStormTracker*. Zenodo, [doi:10.5281/zenodo.18764813](https://doi.org/10.5281/zenodo.18764813).

- **Yau, A. M. W., and E. K. M. Chang**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. Part I: Frameworks for Evaluation and Accumulated Track Activity. *J. Climate*, **33**, 10169–10186, [doi:10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1).

## References

- **Reinecke, M.**, 2020: DUCC: Distinctly Useful Code Collection. *Astrophysics Source Code Library*, record [ascl:2008.023](https://ascl.net/2008.023), [https://gitlab.mpcdf.mpg.de/mtr/ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc).

- **Fritsch, F. N., and J. Butland**, 1984: A Method for Constructing Local Monotone Piecewise Cubic Interpolants. *SIAM Journal on Scientific and Statistical Computing*, **5**(2), 300–304, [doi:10.1137/0905021](https://doi.org/10.1137/0905021). This is the direct numerical-method reference for the PCHIP amplitude extension, implemented with SciPy's [`PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

- **Yau, A. M. W., K. Paul, and J. Dennis**, 2016: PyStormTracker: A Parallel Object-Oriented Cyclone Tracker in Python. *96th American Meteorological Society Annual Meeting*, New Orleans, LA. Zenodo, [doi:10.5281/zenodo.18868625](https://doi.org/10.5281/zenodo.18868625).

- **Neu, U., et al.**, 2013: IMILAST: A Community Effort to Intercompare Extratropical Cyclone Detection and Tracking Algorithms. *Bull. Amer. Meteor. Soc.*, **94**, 529–547, [doi:10.1175/BAMS-D-11-00154.1](https://doi.org/10.1175/BAMS-D-11-00154.1).

  - IMILAST Intercomparison Protocol: [https://proclim.scnat.ch/en/activities/project_imilast/intercomparison](https://proclim.scnat.ch/en/activities/project_imilast/intercomparison)
  - IMILAST Data Download: [https://proclim.scnat.ch/en/activities/project_imilast/data_download](https://proclim.scnat.ch/en/activities/project_imilast/data_download)

- **Blender, R., and M. Schubert**, 2000: Cyclone Tracking in Different Spatial and Temporal Resolutions. *Mon. Wea. Rev.*, **128(2)**, 377–384, [doi:10.1175/1520-0493(2000)128\<0377:CTIDSA>2.0.CO;2](https://doi.org/10.1175/1520-0493%282000%29128%3C0377%3ACTIDSA%3E2.0.CO%3B2).

- **Górski, K. M., et al.**, 2005: HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere. *Astrophysical Journal*, **622(2)**, 759–771, [doi:10.1086/427976](https://doi.org/10.1086/427976).

- **Hodges, K. I.**, 1999: Adaptive Constraints for Feature Tracking. *Mon. Wea. Rev.*, **127**, 1362–1373, [doi:10.1175/1520-0493(1999)127\<1362:ACFFT>2.0.CO;2](https://doi.org/10.1175/1520-0493%281999%29127%3C1362%3AACFFT%3E2.0.CO%3B2).

- **Hodges, K. I.**, 1995: Feature Tracking on the Unit Sphere. *Mon. Wea. Rev.*, **123**, 3458–3465, [doi:10.1175/1520-0493(1995)123\<3458:FTOTUS>2.0.CO;2](https://doi.org/10.1175/1520-0493%281995%29123%3C3458%3AFTOTUS%3E2.0.CO%3B2).

- **Hodges, K. I.**, 1994: A General Method for Tracking Analysis and Its Application to Meteorological Data. *Mon. Wea. Rev.*, **122**, 2573–2586, [doi:10.1175/1520-0493(1994)122\<2573:AGMFTA>2.0.CO;2](https://doi.org/10.1175/1520-0493%281994%29122%3C2573%3AAGMFTA%3E2.0.CO%3B2).

## License

This project is licensed under the BSD-3-Clause terms found in the `LICENSE` file.
