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

| Option             | Description                                                |
| :----------------- | :--------------------------------------------------------- |
| `-i`, `--input`    | Input NetCDF, GRIB, or other Xarray-readable dataset path. |
| `-v`, `--variable` | Variable to track, for example `msl` or `vo`.              |
| `-o`, `--output`   | Output track file.                                         |

### Tracking and preprocessing

| Option                           | Description                                                                                                                                       |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-a`, `--algorithm`              | `simple` or `hodges`; default `simple`.                                                                                                           |
| `-f`, `--format`                 | `auto`, `trackjson`, `imilast`, or `hodges`; default `auto`. Recognized extensions select the format; a suffix-less output defaults to TrackJSON. |
| `-m`, `--detection-mode`         | `auto`, `min`, or `max`; default `auto` (infers `min` for MSL/pressure fields, `max` for vorticity and other standard fields).                    |
| `-p`, `--projection`             | `global`, `nh_stereo`, `sh_stereo`, or `healpix`. Selecting `healpix` uses `HealpixTracker` regardless of `--algorithm`.                          |
| `-r`, `--stereo-grid-spacing-km` | Polar stereographic grid spacing in kilometres; default `100.0`.                                                                                  |
| `--extent`                       | Polar stereographic bounds as `xmin,xmax,ymin,ymax` in kilometres.                                                                                |
| `-t`, `--intensity-threshold`    | Detection threshold. When omitted, the tracker selects the variable-specific default.                                                             |
| `-n`, `--num`                    | Process the first specified number of time steps.                                                                                                 |
| `--filter-lmin`                  | Lower bound of an optional spectral filter; supply with `--filter-lmax`.                                                                          |
| `--filter-lmax`                  | Upper bound of an optional spectral filter; supply with `--filter-lmin`.                                                                          |
| `--taper-points`                 | Independent spatial taper width; zero disables tapering.                                                                                          |
| `--nside`                        | Target HEALPix resolution; omitted values are derived from the source grid.                                                                       |
| `--search-window-size`           | Extrema search window size; default `5`. Must be a positive odd integer.                                                                          |
| `--feature-point-method`         | `grid` (grid extrema) or `quadratic` (quadratic feature-point refinement).                                                                        |

Supplying both `--filter-lmin` and `--filter-lmax` applies the requested optional spectral
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
| `-e`, `--engine`     | Xarray engine: `h5netcdf`, `netcdf4`, or `cfgrib`.                                                                                                 |

Simple supports serial, threaded Dask detection, and MPI detection. Dask and MPI workers return raw detections, which are sorted by time and linked once. Hodges supports serial execution, including serial chunked detection followed by one MGE linking pass. HEALPix supports serial execution. Selecting Dask or MPI for Hodges or HEALPix raises `NotImplementedError`.

### Hodges and HEALPix linking parameters

| Option                       | Description                                                            |
| :--------------------------- | :--------------------------------------------------------------------- |
| `--min-grid-points`          | Minimum number of grid points in a thresholded object.                 |
| `--w1`, `--w2`               | Direction and displacement-magnitude weights in the MGE cost function. |
| `--dmax`                     | Maximum displacement in degrees before regional adjustment.            |
| `--phimax`                   | Smoothness or phantom-point penalty.                                   |
| `--min-lifetime-steps`       | Minimum number of time steps retained by `HodgesTracker`.              |
| `--max-missing-steps`        | Maximum number of consecutive missing frames.                          |
| `--dmax-zone-file`           | TRACK-style regional `dmax` file.                                      |
| `--dmax-zones`               | JSON rows of `[lon_min, lon_max, lat_min, lat_max, dmax]`.             |
| `--adaptive-smoothness-file` | TRACK-style adaptive-smoothness file.                                  |
| `--adaptive-smoothness`      | JSON `2 x 4` array of displacement thresholds and smoothness values.   |

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

| Option             | Description                                                                                           |
| :----------------- | :---------------------------------------------------------------------------------------------------- |
| `-i`, `--input`    | Input TrackJSON trajectory file path.                                                                 |
| `-d`, `--data`     | Input NetCDF data file path to sample from.                                                           |
| `-v`, `--variable` | Variable name in the NetCDF file to sample.                                                           |
| `-o`, `--output`   | Output TrackJSON file path.                                                                           |
| `-m`, `--method`   | Sampling method: `nearest`, `bilinear`, `mean`, `max`, or `min`. Default `nearest`.                   |
| `-r`, `--radius`   | Radius in km for spatial aggregation methods (`mean`, `max`, `min`). Must be positive when specified. |
| `--name`           | Output variable name stored in the track metadata. Defaults to the sampled variable name.             |
| `-e`, `--engine`   | Xarray engine for reading the NetCDF file: `h5netcdf`, `netcdf4`, or `cfgrib`.                        |

## `stormtracker compare`

Compares reference storm tracks with candidate tracks using temporal overlap and mean geodesic separation.

```bash
stormtracker compare -r reference.json -c candidate.json -s 2.0 -l 0.6 -v vo -m max -j
```

| Option                        | Description                                                              |
| :---------------------------- | :----------------------------------------------------------------------- |
| `-r`, `--reference`           | Reference track file path (TrackJSON or IMILAST).                        |
| `-c`, `--candidate`           | Candidate track file path (TrackJSON or IMILAST).                        |
| `-s`, `--max-mean-separation` | Maximum mean great-circle separation in degrees. Default `2.0`.          |
| `-l`, `--min-overlap`         | Minimum temporal overlap fraction. Default `0.6`.                        |
| `-v`, `--variable`            | Common trajectory variable used for intensity statistics.                |
| `-m`, `--detection-mode`      | Peak intensity extremum mode (`auto`, `min`, `max`). Default `auto`.     |
| `-o`, `--output-report`       | Write full comparison report as JSON to the specified path.              |
| `-M`, `--matched-output`      | Write candidate tracks selected by at least one reference track to JSON. |
| `-j`, `--json`                | Output report in JSON format to stdout.                                  |

## `stormtracker convert`

Converts trajectory files between supported formats or generates a static HTML explorer placeholder.

```bash
stormtracker convert -i tracks.trackjson -o tracks.nc -F netcdf
```

| Option                   | Description                                                                                 |
| :----------------------- | :------------------------------------------------------------------------------------------ |
| `-i`, `--input`          | Input trajectory file path.                                                                 |
| `-o`, `--output`         | Output trajectory file path.                                                                |
| `-f`, `--in-format`      | Input format (`auto`, `trackjson`, `imilast`, `hodges`, `netcdf`). Default `auto`.          |
| `-F`, `--out-format`     | Output format (`auto`, `trackjson`, `imilast`, `hodges`, `netcdf`, `html`). Default `auto`. |
| `-v`, `--variable`       | Override or set primary variable name.                                                      |
| `--unit`                 | Unit for a renamed or ambiguous variable.                                                   |
| `-m`, `--detection-mode` | Extremum mode for the primary variable (`auto`, `min`, `max`). Default `auto`.              |
