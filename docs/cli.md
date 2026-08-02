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
stormtracker track -i input.nc -v vo -o tracks.json -m max -a hodges -f json
```

### Required arguments

| Option | Description |
| :--- | :--- |
| `-i`, `--input` | Input NetCDF, GRIB, or other Xarray-readable dataset path. |
| `-v`, `--var` | Variable to track, for example `msl` or `vo`. |
| `-o`, `--output` | Output track file. |

### Tracking and preprocessing

| Option | Description |
| :--- | :--- |
| `-a`, `--algorithm` | `simple` or `hodges`; default `simple`. |
| `-f`, `--format` | `imilast`, `hodges` (TRACK tdump), or `json`; default `imilast`. |
| `-m`, `--mode` | `min` or `max`; default `min`. |
| `--map-proj` | `global`, `nh_stereo`, `sh_stereo`, or `healpix`. Selecting `healpix` uses `HealpixTracker` regardless of `--algorithm`. |
| `--resolution` | Polar stereographic grid spacing in kilometres; default `100`. |
| `--extent` | Polar stereographic bounds as `xmin,xmax,ymin,ymax` in kilometres. |
| `-t`, `--threshold` | Detection threshold. When omitted, the tracker selects the variable-specific default. |
| `-n`, `--num` | Process the first specified number of time steps. |
| `--filter`, `--no-filter` | Explicitly enable or disable spectral filtering. |
| `--filter-range` | Inclusive wave-number range as `MIN-MAX` or `MAX`. This option enables filtering and is mutually exclusive with `--filter`/`--no-filter`. |
| `--subgrid-refine`, `--no-subgrid-refine` | Explicitly enable or disable sub-grid refinement. |

Filtering and refinement are tri-state options at the command-line orchestration layer. If neither form is supplied, Simple tracking uses no filtering and no sub-grid refinement, while Hodges and HEALPix tracking use T5-42 filtering and enabled refinement. Direct tracker calls retain their class-specific defaults.

The production default relative-vorticity threshold is `1e-5 s^-1`. The `1e-4 s^-1` value used by legacy regression fixtures is test-specific.

### Backends and input processing

| Option | Description |
| :--- | :--- |
| `-b`, `--backend` | `serial`, `dask`, or `mpi`. If omitted, an MPI environment is detected first; otherwise specifying workers selects Dask; otherwise serial is used. |
| `-w`, `--workers` | Number of workers. For Dask, this also selects the Dask backend when `--backend` is omitted. |
| `-c`, `--chunk-size` | Number of detection time steps per chunk. |
| `--overlap` | Compatibility option retained from the former chunk-merging interface. Gather-then-Link does not require overlapping chunks. |
| `-e`, `--engine` | Xarray engine: `h5netcdf`, `netcdf4`, or `cfgrib`. |

Simple supports serial, threaded Dask detection, and MPI detection. Dask and MPI workers return raw detections, which are sorted by time and linked once. Hodges supports serial execution, including serial chunked detection followed by one MGE linking pass. HEALPix supports serial execution. Selecting Dask or MPI for Hodges or HEALPix raises `NotImplementedError`.

### Hodges and HEALPix linking parameters

| Option | Description |
| :--- | :--- |
| `--min-points` | Minimum number of grid points in a thresholded object. |
| `--taper` | Number of points in the spatial boundary taper. |
| `--w1`, `--w2` | Direction and displacement-magnitude weights in the MGE cost function. |
| `--dmax` | Maximum displacement in degrees before regional adjustment. |
| `--phimax` | Smoothness or phantom-point penalty. |
| `--iterations` | Maximum number of MGE forward/backward passes. |
| `--min-lifetime` | Minimum number of time steps retained by `HodgesTracker`. `HealpixTracker` currently accepts the value but does not apply lifetime pruning. |
| `--max-missing` | Maximum number of consecutive missing frames. |
| `--zone-file` | TRACK-style regional `dmax` file. |
| `--zones` | JSON rows of `[lon_min, lon_max, lat_min, lat_max, dmax]`. |
| `--adapt-file` | TRACK-style adaptive-smoothness file. |
| `--adapt-params` | JSON `2 x 4` array of displacement thresholds and smoothness values. |

## `stormtracker sample`

Samples a variable from an Xarray-readable dataset at existing track coordinates. The command reads and writes JSON track files.

```bash
stormtracker sample \
  -i tracks.json \
  -d precipitation.nc \
  -v pr \
  -o tracks_with_pr.json \
  -m mean \
  -r 500
```

| Option | Description |
| :--- | :--- |
| `-i`, `--input` | Input JSON track file. |
| `-d`, `--data` | Dataset containing the sampled variable. |
| `-v`, `--var` | Variable name in the dataset. |
| `-o`, `--output` | Output JSON track file. |
| `-m`, `--method` | `nearest`, `bilinear`, `mean`, `max`, or `min`; default `nearest`. |
| `-r`, `--radius` | Radius in kilometres for `mean`, `max`, and `min`. These methods require a positive radius in the CLI. |
| `--name` | Output variable name stored in the track data. |
| `-e`, `--engine` | Optional Xarray input engine. |

For radius methods, candidate grid cells are selected by a latitude-longitude bounding box and retained when their great-circle distance is within the requested radius.

## `stormtracker compare`

Compares candidate tracks with reference tracks using temporal overlap and
whole-overlap mean great-circle separation. Each reference track selects its
closest eligible candidate independently, so a candidate may be selected for
multiple references. Input files are selected by extension: `.json` for JSON
and `.txt` or `.dat` for IMILAST.

```bash
stormtracker compare \
  --reference era5.json \
  --candidate model.json \
  --max-mean-separation 2 \
  --min-overlap 0.6 \
  --intensity-var vo \
  --report comparison.json \
  --json
```

| Option | Description |
| :--- | :--- |
| `--reference` | Reference track file. |
| `--candidate` | Candidate track file. |
| `--max-mean-separation` | Maximum mean geodetic separation in degrees; default `2`. |
| `--min-overlap` | Minimum overlap ratio, defined as `2 * overlap / (n_ref + n_candidate)`; default `0.6`. |
| `--intensity-var` | Optional common variable for intensity-difference statistics. |
| `--report` | Write the complete comparison report as JSON. |
| `--matched-candidate-output` | Write candidates selected by at least one reference as JSON. |
| `--json` | Print the full report JSON to standard output. |

The report includes assigned and unassigned IDs, overlap, mean/percentile
separation, lifecycle and path metrics, and optional intensity bias, MAE, RMSE,
and correlation. The Hodges documentation records the source comparison method
and its common-cadence requirement.

## `stormtracker convert`

Converts track files and generates the HTML track explorer.

```bash
# IMILAST to JSON
stormtracker convert -i tracks.txt -o tracks.json -f imilast -F json

# Standalone HTML explorer
stormtracker convert -i tracks.json -o explorer.html -f json -F html

# HTML plus a separate JavaScript data file
stormtracker convert -i tracks.json -o explorer.html -f json -F html --split
```

| Option | Description |
| :--- | :--- |
| `-i`, `--input` | Input path. |
| `-o`, `--output` | Output path. |
| `-f`, `--in-format` | `imilast` or `json`. |
| `-F`, `--out-format` | `imilast`, `hodges`, `json`, or `html`. |
| `--type` | Override inferred track type with `msl` or `vo`. |
| `--split` | For HTML output, place track data in a separate `.tracks.js` file. |
