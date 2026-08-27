# Quickstart

This guide contains the installation, usage, sample-data, and development material moved out of the repository landing page. For scientific details, see the [architecture](architecture.md), [Hodges/TRACK implementation](hodges.md), [HEALPix support](healpix.md), and [TrackJSON](trackjson.md) documentation.

## Installation

### Prerequisites

- **Python 3.12+**
- **Message Passing Interface (MPI)**:
  - **Linux/macOS**: OpenMPI is recommended and included as a development dependency.
  - **Windows**: use `winget install -e --id Microsoft.msmpi` or install [MS-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
- **Spherical harmonic transforms** are provided by `ducc0`, including scalar and spin-weighted transforms, reduced-grid synthesis, and HEALPix geometry.
- **Free-threaded Python 3.14** support is experimental. CI currently excludes `eof`, `grib`, and `zarr` in that configuration because of upstream dependency availability.

### PyPI

```bash
pip install PyStormTracker
```

Optional components can be installed with extras:

```bash
pip install "PyStormTracker[mpi]"     # mpi4py
pip install "PyStormTracker[grib]"    # GRIB support
pip install "PyStormTracker[netcdf4]" # NetCDF4 backend
pip install "PyStormTracker[zarr]"    # Zarr and remote stores
pip install "PyStormTracker[eof]"     # xeofs for CCA/PCA analysis
pip install "PyStormTracker[all]"     # non-visualization optional components
```

With `uv`:

```bash
# CLI tool
uv tool install "PyStormTracker[mpi]"

# Library dependency
uv add "PyStormTracker[mpi]"
```

### Conda-forge

```bash
mamba install -c conda-forge pystormtracker
```

or:

```bash
conda install -c conda-forge pystormtracker
```

### From source

```bash
git clone https://github.com/mwyau/PyStormTracker.git
cd PyStormTracker
uv sync
```

## Command-line usage

The `stormtracker` command provides tracking, sampling, comparison, and conversion subcommands.

### Track features

Track mean sea-level pressure minima with the Hodges tracker:

```bash
stormtracker track \
    -i data.nc \
    --variable msl \
    -o tracks.trackjson \
    -m min \
    -a hodges
```

### Sample variables along tracks

For example, calculate mean precipitation within 500 km of existing storm centers:

```bash
stormtracker sample \
    -i tracks.trackjson \
    -d precip.nc \
    --variable pr \
    -o tracks_enriched.trackjson \
    --method mean \
    --radius 500
```

### Compare track datasets

```bash
stormtracker compare \
    -r era5.trackjson \
    -c gfs.trackjson \
    -s 2.0 \
    -l 0.6 \
    --variable vo \
    -m max \
    --json
```

### Convert trajectory formats

```bash
stormtracker convert \
    -i tracks.trackjson \
    -o tracks.imilast \
    -F imilast
```

Use `stormtracker <command> --help` for the full argument reference. `-v` and `-vv` select INFO and DEBUG logging; `-V` prints the version.

Important `track` options include:

| Argument               | Short | Description                                                                              |
| ---------------------- | ----- | ---------------------------------------------------------------------------------------- |
| `--input`              | `-i`  | Input NetCDF/GRIB file.                                                                  |
| `--variable`           |       | Variable name such as `msl` or `vo`.                                                     |
| `--output`             | `-o`  | Output trajectory file.                                                                  |
| `--algorithm`          | `-a`  | `simple`, `hodges`, or `healpix`.                                                        |
| `--format`             | `-f`  | `auto`, `json`, `track`, or `imilast`; recognized extensions are inferred automatically. |
| `--detection-mode`     | `-m`  | `auto`, `min`, or `max`.                                                                 |
| `--backend`            | `-b`  | `serial`, `dask`, or `mpi`.                                                              |
| `--workers`            | `-w`  | Number of parallel workers where applicable.                                             |
| `--lmin`, `--lmax`     |       | Optional spectral-filter bounds; supply both to filter.                                  |
| `--taper-points`       |       | Independent spatial taper width; zero disables tapering.                                 |
| `--spectral-taper`     |       | Hodges/HEALPix high-wavenumber coefficient taper.                                        |
| `--nside`              |       | Target HEALPix resolution; omitted values are derived from the source grid.              |
| `--feature-refinement` |       | Tracker-dependent feature-point location method.                                         |
| `--no-progress`        |       | Disable the interactive Hodges Dask progress display.                                    |

See the [CLI reference](cli.md) for all commands and options.

## Python API

```python
import pystormtracker as pst

tracker = pst.HodgesTracker()

tracks = tracker.track(
    data="data.nc",
    variable="vo",
    detection_mode="max",
)
```

`Tracks` is iterable:

```python
for track in tracks:
    if len(track) >= 8:
        print(f"Track {track.track_id} lived for {len(track)} steps.")
```

Write trajectories in a supported format:

```python
tracks.write("output.txt", format="imilast")
```

See the [API reference](api.md) for the full Python interface.

## How tracking is organized

PyStormTracker separates preprocessing, feature detection and refinement, and trajectory linking.

- **Preprocessing** can apply spherical-harmonic filtering on global grids, DCT filtering on regional grids, spectral tapering, and projection/regridding to polar or HEALPix coordinates.
- **SimpleTracker** uses local-extrema detection and deterministic nearest-neighbor linking.
- **HodgesTracker** uses thresholded objects, local extrema, optional sub-grid refinement, and Modified Greedy Exchange trajectory linking. The default `bspline` refinement follows the reconciled rectangular TRACK/SMOOPY workflow.
- **HealpixTracker** performs object detection on HEALPix topology and uses the Hodges MGE linker.
- Serial, Dask, and MPI execution are supported by the implemented tracker paths.

The authoritative implementation details are in [Architecture](architecture.md), [TRACK Implementation and Comparison](hodges.md), and [HEALPix Support](healpix.md).

## Formats and analysis

PyStormTracker reads IMILAST and TrackJSON trajectory data and writes IMILAST, TRACK tdump, and TrackJSON. Analysis functionality includes variable sampling, trajectory comparison, gridded cyclone and track metrics, Eulerian variance and wind indices, CORMAX, and CCA/PCA truncation cross-validation.

TrackJSON is the native compact array-oriented format; see [TrackJSON](trackjson.md).

## Sample and reference data

The checkout retains one ordinary integration input:

```text
tests/data/era5/era5_msl_2025-12_2.5x2.5.nc
```

Specialized GRIB, reduced-Gaussian, and broader reference datasets are owned by the pinned [PyStormTracker-Data](https://github.com/mwyau/PyStormTracker-Data) contract. Small references use raw Git paths, large files use Release assets, and Git-tracked Zarr stores use their raw store paths.

The bundled NCL T5-42 spectral numerical-parity reference is:

```text
tests/data/ncl/era5_msl_2025-12-01_0000_2.5x2.5_t5-42.nc
```

Broader NCL/Spherepack kinematics parity remains deferred because the pinned
`PyStormTracker-Data` release does not yet contain the required NCL-generated
VODV reference fields.

## Development

Set up the complete development environment with:

```bash
uv sync --all-extras
```

Run the fast default unit suite:

```bash
uv run pytest
```

Run code-quality checks with:

```bash
uv run prek run --all-files
uv run mypy
uv run python scripts/generate_trackjson_schema.py --check
```

Testing is tiered:

- **Unit** tests are the default fast offline suite.
- **Integration** tests exercise current PyStormTracker components together on real data.
- **Parity** tests compare current package behavior with static external or historical results, including the bundled NCL T5-42 spectral numerical-parity case.
- **Scientific validation** and TRACK source-stage reconciliation are outside
  the package test suite.

Examples:

```bash
# Unit tests
uv run pytest

# Non-slow, bundled integration tests
uv run pytest tests/integration -m "not slow and not data"

# Non-slow, bundled parity tests
uv run pytest tests/parity -m "not slow and not data"
```

See [CONTRIBUTING.md](https://github.com/mwyau/PyStormTracker/blob/main/CONTRIBUTING.md)
and the [testing guide](development/testing.md) for contributor workflows and
test taxonomy.
