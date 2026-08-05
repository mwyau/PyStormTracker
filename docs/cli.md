# Command-Line Interface Reference

PyStormTracker uses one `stormtracker` entry point with four subcommands:

```text
stormtracker track
stormtracker sample
stormtracker compare
stormtracker convert
```

Use `stormtracker <command> --help` for the complete parser-generated option list.

## `stormtracker track`

Runs feature detection and trajectory linking.

```bash
stormtracker track -i input.nc -v vo -o tracks.trackjson -m max -a hodges
```

### Required arguments

| Option           | Description                                                |
| :--------------- | :--------------------------------------------------------- |
| `-i`, `--input`  | Input NetCDF, GRIB, or other Xarray-readable dataset path. |
| `-v`, `--var`    | Variable to track, for example `msl` or `vo`.              |
| `-o`, `--output` | Output track file.                                         |

### Tracking and preprocessing

| Option                                    | Description                                                                                                                                       |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-a`, `--algorithm`                       | `simple` or `hodges`; default `simple`.                                                                                                           |
| `-f`, `--format`                          | `auto`, `trackjson`, `imilast`, or `hodges`; default `auto`. Recognized extensions select the format; a suffix-less output defaults to TrackJSON. |
| `-m`, `--mode`                            | `auto`, `min`, or `max`; default `auto` (infers `min` for MSL/pressure fields, `max` for vorticity and other standard fields).                    |
| `--map-proj`                              | `global`, `nh_stereo`, `sh_stereo`, or `healpix`. Selecting `healpix` uses `HealpixTracker` regardless of `--algorithm`.                          |
| `--resolution`                            | Polar stereographic grid spacing in kilometres; default `100`.                                                                                    |
| `--extent`                                | Polar stereographic bounds as `xmin,xmax,ymin,ymax` in kilometres.                                                                                |
| `-t`, `--threshold`                       | Detection threshold. When omitted, the tracker selects the variable-specific default.                                                             |
| `-n`, `--num`                             | Process the first specified number of time steps.                                                                                                 |
| `--lmin`                                  | Lower bound of an optional spectral filter; supply with `--lmax`.                                                                                 |
| `--lmax`                                  | Upper bound of an optional spectral filter; supply with `--lmin`.                                                                                 |
| `--taper-points`                          | Independent spatial taper width; zero disables tapering.                                                                                          |
| `--nside`                                 | Target HEALPix resolution; omitted values are derived from the source grid.                                                                       |
| `--subgrid-refine`, `--no-subgrid-refine` | Explicitly enable or disable sub-grid refinement.                                                                                                 |

Supplying both `--lmin` and `--lmax` applies the requested optional spectral
filter. Supplying only one is an error. When both are omitted, no optional
spectral filter is applied. `--taper-points` is independent and applies a
spatial taper when greater than zero. Projection and HEALPix conversion may
still use a finite transform bandwidth derived from the source and target
grids; that transform is regridding work, not optional filtering.

The production default relative-vorticity threshold is `1e-5 s^-1`. The `1e-4 s^-1` value used by legacy regression test datasets is test-specific.

### Backends and input processing

| Option               | Description                                                                                                                                        |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-b`, `--backend`    | `serial`, `dask`, or `mpi`. If omitted, an MPI environment is detected first; otherwise specifying workers selects Dask; otherwise serial is used. |
| `-w`, `--workers`    | Number of workers. For Dask, this also selects the Dask backend when `--backend` is omitted.                                                       |
| `-c`, `--chunk-size` | Number of detection time steps per chunk.                                                                                                          |
| `--overlap`          | Compatibility option retained from the former chunk-merging interface. Gather-then-Link does not require overlapping chunks.                       |
| `-e`, `--engine`     | Xarray engine: `h5netcdf`, `netcdf4`, or `cfgrib`.                                                                                                 |

Simple supports serial, threaded Dask detection, and MPI detection. Dask and MPI workers return raw detections, which are sorted by time and linked once. Hodges supports serial execution, including serial chunked detection followed by one MGE linking pass. HEALPix supports serial execution. Selecting Dask or MPI for Hodges or HEALPix raises `NotImplementedError`.

### Hodges and HEALPix linking parameters

| Option           | Description                                                                                                                                 |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| `--min-points`   | Minimum number of grid points in a thresholded object.                                                                                      |
| `--taper-points` | Number of points in the spatial boundary taper.                                                                                             |
| `--w1`, `--w2`   | Direction and displacement-magnitude weights in the MGE cost function.                                                                      |
| `--dmax`         | Maximum displacement in degrees before regional adjustment.                                                                                 |
| `--phimax`       | Smoothness or phantom-point penalty.                                                                                                        |
| `--iterations`   | Maximum number of MGE forward/backward passes.                                                                                              |
| `--min-lifetime` | Minimum number of time steps retained by `HodgesTracker`. `HealpixTracker` currently accepts the value but does not apply lifetime pruning. |
| `--max-missing`  | Maximum number of consecutive missing frames.                                                                                               |
| `--zone-file`    | TRACK-style regional `dmax` file.                                                                                                           |
| `--zones`        | JSON rows of `[lon_min, lon_max, lat_min, lat_max, dmax]`.                                                                                  |
| `--adapt-file`   | TRACK-style adaptive-smoothness file.                                                                                                       |
| `--adapt-params` | JSON `2 x 4` array of displacement thresholds and smoothness values.                                                                        |

## `stormtracker sample`

Samples a variable from an Xarray-readable dataset at existing track coordinates. The command reads and writes TrackJSON files.

```bash
stormtracker sample \
  -i tracks.trackjson \
  -d precipitation.nc \
  -v pr \
  -o tracks_with_pr.trackjson \
  -m mean \
  -r 500
```

| Option           | Description                                                                                            |
| :--------------- | :----------------------------------------------------------------------------------------------------- |
| `-i`, `--input`  | Input TrackJSON file.                                                                                  |
| `-d`, `--data`   | Dataset containing the sampled variable.                                                               |
| `-v`, `--var`    | Variable name in the dataset.                                                                          |
| `-o`, `--output` | Output TrackJSON file.                                                                                 |
| `-m`, `--method` | `nearest`, `bilinear`, `mean`, `max`, or `min`; default `nearest`.                                     |
| `-r`, `--radius` | Radius in kilometres for `mean`, `max`, and `min`. These methods require a positive radius in the CLI. |
| `--name`         | Output variable name stored in the track data.                                                         |
| `-e`, `--engine` | Optional Xarray input engine.                                                                          |

For radius methods, candidate grid cells are selected by a latitude-longitude bounding box and retained when their great-circle distance is within the requested radius.

## `stormtracker compare`

Compares candidate tracks with reference tracks using temporal overlap and
whole-overlap mean great-circle separation. Each reference track selects its
closest eligible candidate independently, so a candidate may be selected for
multiple references. Input files are selected by extension: `.trackjson` or
`.json` for TrackJSON and `.txt` or `.dat` for IMILAST.

```bash
stormtracker compare -r era5.trackjson -c model.trackjson -s 2 -l 0.6 -v vo -o comparison.json -j
```

| Option                | Description                                                                             |
| :-------------------- | :-------------------------------------------------------------------------------------- |
| `-r`, `--ref`         | Reference track file.                                                                   |
| `-c`, `--cand`        | Candidate track file.                                                                   |
| `-s`, `--max-sep`     | Maximum mean geodetic separation in degrees; default `2`.                               |
| `-l`, `--min-overlap` | Minimum overlap ratio, defined as `2 * overlap / (n_ref + n_candidate)`; default `0.6`. |
| `-v`, `--var`         | Optional common variable for intensity-difference statistics.                           |
| `-m`, `--mode`        | Extremum mode (`auto`, `min`, `max`); default `auto`.                                   |
| `-o`, `--out`         | Write the complete comparison report as JSON.                                           |
| `-M`, `--matched-out` | Write candidates selected by at least one reference as JSON.                            |
| `-j`, `--json`        | Print the full report JSON to standard output.                                          |

The report includes assigned and unassigned IDs, overlap, mean/percentile
separation, lifecycle and path metrics, and optional intensity bias, MAE, RMSE,
and correlation. The Hodges documentation records the source comparison method
and its common-cadence requirement.

## `stormtracker convert`

Converts track files. HTML output is retained for compatibility while the
explorer is being redesigned and currently emits a static placeholder.

```bash
# IMILAST to TrackJSON (the text header is inspected automatically)
stormtracker convert -i tracks.txt -o tracks.trackjson

# Standalone HTML explorer
stormtracker convert -i tracks.trackjson -o explorer.html

```

| Option               | Description                                                                               |
| :------------------- | :---------------------------------------------------------------------------------------- |
| `-i`, `--input`      | Input path.                                                                               |
| `-o`, `--out`        | Output path.                                                                              |
| `-f`, `--in-format`  | `auto`, `trackjson`, `imilast`, or `hodges`; default `auto`.                              |
| `-F`, `--out-format` | `auto`, `trackjson`, `imilast`, `hodges`, or `html`; default `auto`.                      |
| `-v`, `--var`        | Override the explicit primary variable name (e.g., `msl` or `vo`).                        |
| `--unit`             | Unit required when a variable rename cannot establish a physical unit.                    |
| `--mode`             | `auto`, `min`, or `max`; default `auto`. The final variable name resolves automatic mode. |

The native format is TrackJSON. `.json` and `.trackjson` select TrackJSON;
`.hodges`, `.track`, and `.tdump` select Hodges; `.txt` and `.dat` select
IMILAST for output and are inspected before selecting an input format. An
explicit format overrides the extension. HTML output is a temporary static
placeholder; no data script is produced. See the [TrackJSON v1.0
specification](trackjson.md) for the wire structure, bounds, and statistics.
