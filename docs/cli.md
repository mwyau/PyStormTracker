# Command-Line Interface Reference

PyStormTracker uses one `stormtracker` entry point with four subcommands:

```text
stormtracker track
stormtracker sample
stormtracker compare
stormtracker convert
```

Use `stormtracker <command> --help` for the complete parser-generated option list.
The shared short-option contract is `-v` for INFO logging, `-vv` for DEBUG,
and `-V` for the version. These options work before or after a subcommand.
`--variable NAME` is the variable option for every command; it has no short
alias.

## `stormtracker track`

Runs feature detection and trajectory linking.

```bash
stormtracker -v track -i input.nc --variable vo -o tracks.trackjson -m max -a hodges
```

### Required arguments

| Option           | Description                                                |
| :--------------- | :--------------------------------------------------------- |
| `-i`, `--input`  | Input NetCDF, GRIB, or other Xarray-readable dataset path. |
| `--variable`     | Variable to track, for example `msl` or `vo`.              |
| `-o`, `--output` | Output track file.                                         |

### Tracking and preprocessing

| Option                           | Description                                                                                                                                                                                                                                                                                             |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-a`, `--algorithm`              | `simple`, `hodges`, or `healpix`; default `simple`.                                                                                                                                                                                                                                                     |
| `-f`, `--format`                 | `auto`, `json`, `track`, or `imilast`; default `auto`. Recognized extensions select the format; a suffix-less output defaults to TrackJSON.                                                                                                                                                             |
| `-m`, `--detection-mode`         | `auto`, `min`, or `max`; default `auto` (infers `min` for MSL/pressure fields, `max` for vorticity and other standard fields).                                                                                                                                                                          |
| `-p`, `--projection`             | `global`, `nh_stereo`, or `sh_stereo`.                                                                                                                                                                                                                                                                  |
| `-r`, `--stereo-grid-spacing-km` | Polar stereographic grid spacing in kilometres; default `100.0`.                                                                                                                                                                                                                                        |
| `--extent`                       | Polar stereographic bounds as `xmin,xmax,ymin,ymax` in kilometres.                                                                                                                                                                                                                                      |
| `--object-threshold`             | Object segmentation threshold for Hodges and HEALPix trackers.                                                                                                                                                                                                                                          |
| `--feature-threshold`            | Feature detection threshold for SimpleTracker.                                                                                                                                                                                                                                                          |
| `-n`, `--n-frames`               | Process the first specified number of time steps.                                                                                                                                                                                                                                                       |
| `--lmin`                         | Lower bound of an optional spectral filter; supply with `--lmax`.                                                                                                                                                                                                                                       |
| `--lmax`                         | Upper bound of an optional spectral filter; supply with `--lmin`.                                                                                                                                                                                                                                       |
| `--taper-points`                 | Independent spatial taper width; zero disables tapering.                                                                                                                                                                                                                                                |
| `--spectral-taper`               | Hodges/HEALPix spectral coefficient taper in `(0, 1]`; default `1.0` for Hodges (no taper), `0.1` for HEALPix.                                                                                                                                                                                          |
| `--nside`                        | Target HEALPix resolution; omitted values are derived from the source grid.                                                                                                                                                                                                                             |
| `--search-window-size`           | Extrema search window size; Simple uses `5` by default. Must be a positive odd integer.                                                                                                                                                                                                                 |
| `--feature-refinement`           | `grid` (discrete extrema), `quadratic` (local quadratic), `spherical_quadratic` (spherical quadratic), `bspline` (TRACK/SMOOPY-compatible rectangular B-spline), or `spherical_bspline` (spherical B-spline). Simple defaults to `grid`; Hodges defaults to `bspline`; HEALPix defaults to `quadratic`. |
| `--dmax-zones`                   | Path to regional DMAX definitions file (rows of `lon_min lon_max lat_min lat_max dmax`).                                                                                                                                                                                                                |
| `--adaptive-smoothness`          | Path to adaptive smoothness parameters file (2x4 or 4x2 matrix).                                                                                                                                                                                                                                        |
| `--no-segmentation`              | Disable temporal MGE segmentation and run monolithic linking. It cannot be combined with an explicit `--segment-frames`.                                                                                                                                                                                |
| `--no-progress`                  | Disable the interactive Hodges Dask progress display. The display is enabled by default when standard error is a terminal.                                                                                                                                                                              |

Supplying both `--lmin` and `--lmax` applies the requested optional spectral
filter. Supplying only one is an error. When both are omitted, no optional
spectral filter is applied. `--taper-points` is independent and applies a
spatial taper when greater than zero. Projection and HEALPix conversion may
still use a finite transform bandwidth derived from the source and target
grids; that transform is regridding work, not optional filtering.

`--spectral-taper` is separate from the spatial boundary taper. It is ignored
by Simple. The Hodges default is `1.0`; the HEALPix default is `0.1`.

The production default relative-vorticity threshold is `1e-5 s^-1`. The `1e-4 s^-1` value used by legacy regression test datasets is test-specific.

#### Backends and input processing

| Option                   | Description                                                                                                                                                                       |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-b`, `--backend`        | `dask`, `serial`, or `mpi`. Default: `dask` (or `mpi` if an active MPI environment is detected).                                                                                  |
| `-w`, `--workers`        | Generic Dask worker count for Simple/HEALPix. Defaults to available process CPU concurrency; not accepted by Hodges.                                                              |
| `--frame-workers`        | Hodges Dask frame-processing tasks, including lazy source reads, preprocessing, detection, and refinement.                                                                        |
| `--sht-threads`          | DUCC0 native threads per active Hodges spherical-harmonic transform.                                                                                                              |
| `--mge-workers`          | Hodges Dask MGE segment-linking tasks that may run concurrently.                                                                                                                  |
| `-c`, `--segment-frames` | MGE temporal segment length for Hodges/HEALPix; default `62`. Not used by Simple and independent of worker counts. An explicit value cannot be combined with `--no-segmentation`. |
| `-e`, `--engine`         | Xarray engine: `h5netcdf` by default, or explicitly `netcdf4` for legacy NetCDF3 and `cfgrib` for GRIB.                                                                           |

Simple, Hodges, and HEALPix support serial, threaded Dask, and MPI execution.
Simple gathers detections before linking; Hodges runs frame tasks and then
independent MGE segment tasks before deterministic splicing. HEALPix retains the
generic `--workers` control.

### Hodges and HEALPix linking parameters

| Option                     | Description                                                                                                                                       |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--min-object-grid-points` | Minimum number of grid points in a thresholded object.                                                                                            |
| `--w1`, `--w2`             | Direction and displacement-magnitude weights in the MGE cost function.                                                                            |
| `--dmax`                   | Maximum displacement in degrees before regional adjustment.                                                                                       |
| `--phimax`                 | Smoothness or phantom-point penalty.                                                                                                              |
| `--min-track-points`       | Minimum number of linked time steps retained as a track.                                                                                          |
| `--mge-max-iterations`     | Maximum number of MGE iteration rounds.                                                                                                           |
| `--time-step`              | Expected input cadence for Hodges missing-frame handling.                                                                                         |
| `--feature-refinement`     | Feature-point location method: `grid`, `quadratic`, `spherical_quadratic`, TRACK/SMOOPY-compatible rectangular `bspline`, or `spherical_bspline`. |
| `--dmax-zones`             | Path to regional DMAX definitions file.                                                                                                           |
| `--adaptive-smoothness`    | Path to adaptive smoothness parameters file.                                                                                                      |

`--time-step` expects `<positive integer><s|m|h|D>`; for example, `6h`. It
provides the expected input cadence used by Hodges missing-frame handling.

## `stormtracker sample`

Samples external variables along trajectory points and writes the enriched tracks.

```bash
stormtracker sample -i tracks.trackjson -d era5_mslp.nc --variable msl -o sampled_tracks.trackjson
```

| Option           | Description                                                                                             |
| :--------------- | :------------------------------------------------------------------------------------------------------ |
| `-i`, `--input`  | Path to input trajectory file (`TrackJSON` or `IMILAST`).                                               |
| `-d`, `--data`   | Path to meteorological gridded data file to sample from.                                                |
| `-o`, `--output` | Path to save the sampled trajectory file.                                                               |
| `--variable`     | Variable name in the source file to sample.                                                             |
| `-m`, `--method` | Sampling method: `nearest`, `bilinear`, `mean`, `max`, or `min`; default `nearest`.                     |
| `-r`, `--radius` | Radius in kilometres for the `mean`, `max`, and `min` spatial methods; default `0.0`.                   |
| `--name`         | Output variable name stored in the track metadata. Defaults to the sampled variable name.               |
| `-e`, `--engine` | Xarray engine: `h5netcdf` by default, or explicitly `netcdf4` for legacy NetCDF3 and `cfgrib` for GRIB. |

## `stormtracker compare`

Compares reference and candidate storm tracks using `nearest`, `mutual_nearest`, or `global_assignment` matching.

```bash
stormtracker compare -r reference.trackjson -c candidate.trackjson --matching global_assignment -s 2.0 -l 0.6 -j
```

| Option                        | Description                                                                              |
| :---------------------------- | :--------------------------------------------------------------------------------------- |
| `-r`, `--reference`           | Reference track file path (`TrackJSON`, `IMILAST`, or TRACK/`tdump`).                    |
| `-c`, `--candidate`           | Candidate track file path (`TrackJSON`, `IMILAST`, or TRACK/`tdump`).                    |
| `--matching`                  | Matching method: `nearest`, `mutual_nearest`, or `global_assignment`. Default `nearest`. |
| `-s`, `--max-mean-separation` | Maximum mean great-circle separation in degrees. Default `2.0`.                          |
| `-l`, `--min-overlap`         | Minimum temporal overlap fraction. Default `0.6`.                                        |
| `--variable`                  | Common trajectory variable used for intensity statistics.                                |
| `-m`, `--detection-mode`      | Peak intensity extremum mode (`auto`, `min`, `max`). Default `auto`.                     |
| `-o`, `--output-report`       | Write full comparison report as JSON to the specified path.                              |
| `-M`, `--matched-output`      | Write candidate tracks selected by at least one reference track to JSON.                 |
| `-j`, `--json`                | Output report in JSON format to stdout.                                                  |

## `stormtracker convert`

Converts trajectory files between supported formats.

```bash
stormtracker convert -i tracks.trackjson -o tracks.trackjson -F json
```

| Option                   | Description                                                                    |
| :----------------------- | :----------------------------------------------------------------------------- |
| `-i`, `--input`          | Input trajectory file path.                                                    |
| `-o`, `--output`         | Output trajectory file path.                                                   |
| `-f`, `--in-format`      | Input format (`auto`, `json`, `track`, `imilast`). Default `auto`.             |
| `-F`, `--out-format`     | Output format (`auto`, `json`, `track`, `imilast`). Default `auto`.            |
| `--variable`             | Override or set primary variable name.                                         |
| `--unit`                 | Unit for a renamed or ambiguous variable.                                      |
| `-m`, `--detection-mode` | Extremum mode for the primary variable (`auto`, `min`, `max`). Default `auto`. |
