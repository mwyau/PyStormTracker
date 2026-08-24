# TRACK benchmark input semantics

This file is the line-by-line decoding of the checked-in TRACK 1.5.4 input
streams and auxiliary files in this directory. The meanings below are based
on the TRACK 1.5.4 source prompts, not on positional guesses from a wrapper.
The numbered lines are significant: blank lines are not inserted into the
streams.

## `configs/regular_latlon/specfilt_regular_2p5_T42`

This stream is for a 144 by 73 regular latitude/longitude source grid. It
selects NetCDF field `msl`, uses least-squares spherical harmonics (method 0),
creates a T42 Gaussian output grid, and writes two spectral bands.

| Line | Value     | Meaning                                                                     |
| ---: | --------- | --------------------------------------------------------------------------- |
|    1 | `n`       | Do not load a country map.                                                  |
|    2 | `0`       | Do not use an existing initialization.                                      |
|    3 | `4`       | Read the source as NetCDF.                                                  |
|    4 | `n`       | Do not print the NetCDF summary.                                            |
|    5 | `1`       | Identify fields using NetCDF names and dimension values.                    |
|    6 | `y`       | Treat the file as COARDS-organized.                                         |
|    7 | `msl`     | Select the mean sea-level pressure variable.                                |
|    8 | `n`       | Do not translate the source grid.                                           |
|    9 | `y`       | Retain the equator in the grid.                                             |
|   10 | `y`       | Retain the Southern Hemisphere pole.                                        |
|   11 | `y`       | Retain the Northern Hemisphere pole.                                        |
|   12 | `n`       | Do not make the regular source grid periodic before filtering.              |
|   13 | `g`       | Use the geodesic distance norm.                                             |
|   14 | `n`       | Do not use a projection other than the default Plate Carrée interpretation. |
|   15 | `1`       | First source-grid X index.                                                  |
|   16 | `144`     | Source-grid X extent.                                                       |
|   17 | `1`       | First source-grid Y index.                                                  |
|   18 | `73`      | Source-grid Y extent.                                                       |
|   19 | `y`       | Enter TRACK's analysis routines.                                            |
|   20 | `4`       | Select spatial spectral filtering.                                          |
|   21 | `1`       | Start at source frame 1.                                                    |
|   22 | `1`       | Read every source frame.                                                    |
|   23 | `1000000` | End-frame sentinel; TRACK stops at source EOF.                              |
|   24 | `0`       | Select least-squares spherical-harmonic filtering.                          |
|   25 | `0`       | Perform a full spectral decomposition, not band subtraction.                |
|   26 | `42`      | Source spectral truncation.                                                 |
|   27 | `0`       | Do not use memory-mapped least-squares storage.                             |
|   28 | `y`       | Reconstruct filtered fields on a new Gaussian grid.                         |
|   29 | `1`       | Create the Gaussian grid rather than read one from a file.                  |
|   30 | `42`      | New-grid triangular truncation.                                             |
|   31 | `n`       | Omit the longitude wraparound from the output file.                         |
|   32 | `2`       | Create two total-wavenumber bands.                                          |
|   33 | `0`       | First band lower boundary, T0.                                              |
|   34 | `5`       | First band upper boundary, T5.                                              |
|   35 | `42`      | Second band upper boundary, T42.                                            |
|   36 | `1`       | Do not mask band 1.                                                         |
|   37 | `1`       | Do not mask band 2.                                                         |
|   38 | `y`       | Apply the Hoskins coefficient filter.                                       |
|   39 | `0.1`     | Hoskins cutoff constant.                                                    |
|   40 | `y`       | Apply the Hoskins filter to band 1.                                         |
|   41 | `y`       | Apply the Hoskins filter to band 2.                                         |
|   42 | `n`       | Do not clamp large field values before filtering.                           |
|   43 | `n`       | Trailing answer; not read by the completed global least-squares path.       |
|   44 | `n`       | Trailing answer; not read by the completed global least-squares path.       |

The 0.25-degree stream in `specfilt_regular_0p25_T42` is identical except
that line 16 is `1440` and line 18 is `721`.

## `configs/f320/specfilt_f320_to_t42`

This is the fast spectral-transform path for the native 1280 by 640 Gaussian
ERA5 grid. The source file contains only `msl`, so TRACK selects it
automatically and no field-name answer is needed.

| Line | Value     | Meaning                                                  |
| ---: | --------- | -------------------------------------------------------- |
|    1 | `n`       | Do not load a country map.                               |
|    2 | `0`       | Do not use an existing initialization.                   |
|    3 | `4`       | Read the source as NetCDF.                               |
|    4 | `n`       | Do not print the NetCDF summary.                         |
|    5 | `1`       | Identify fields using NetCDF names and dimension values. |
|    6 | `y`       | Treat the file as COARDS-organized.                      |
|    7 | `n`       | Do not translate the Gaussian source grid.               |
|    8 | `n`       | Do not make the source grid periodic before filtering.   |
|    9 | `g`       | Use the geodesic distance norm.                          |
|   10 | `n`       | Do not use a different map projection.                   |
|   11 | `1`       | First source-grid X index.                               |
|   12 | `1280`    | Source-grid X extent.                                    |
|   13 | `1`       | First source-grid Y index.                               |
|   14 | `640`     | Source-grid Y extent.                                    |
|   15 | `y`       | Enter TRACK's analysis routines.                         |
|   16 | `4`       | Select spatial spectral filtering.                       |
|   17 | `1`       | Start at source frame 1.                                 |
|   18 | `1`       | Read every source frame.                                 |
|   19 | `1000000` | End-frame sentinel; TRACK stops at source EOF.           |
|   20 | `1`       | Select the fast spectral transform.                      |
|   21 | `n`       | No derived field is required.                            |
|   22 | `42`      | Spectral truncation.                                     |
|   23 | `y`       | Reconstruct on a new grid.                               |
|   24 | `1`       | Create a Gaussian output grid.                           |
|   25 | `42`      | New Gaussian-grid truncation.                            |
|   26 | `2`       | Create two total-wavenumber bands.                       |
|   27 | `y`       | Apply the Hoskins coefficient filter.                    |
|   28 | `0.1`     | Hoskins cutoff constant.                                 |
|   29 | `0`       | First band lower boundary, T0.                           |
|   30 | `5`       | First band upper boundary, T5.                           |
|   31 | `42`      | Second band upper boundary, T42.                         |
|   32 | `n`       | Do not restrict field values before filtering.           |

The exact line count and prompt branch are source-dependent. In particular,
the fast path does not ask the regular-grid field-selection questions.

## `configs/f320/specfilt_f320_to_f320`

Lines 1--22 are the same fast-path setup as `specfilt_f320_to_t42`. The
F320-output stream then uses lines 23--30:

| Line | Value | Meaning                                                |
| ---: | ----- | ------------------------------------------------------ |
|   23 | `n`   | Synthesize the T6-42 result on the existing F320 grid. |
|   24 | `2`   | Create two total-wavenumber bands.                     |
|   25 | `y`   | Apply the Hoskins coefficient filter.                  |
|   26 | `0.1` | Hoskins cutoff constant.                               |
|   27 | `0`   | First band lower boundary, T0.                         |
|   28 | `5`   | First band upper boundary, T5.                         |
|   29 | `42`  | Second band upper boundary, T42.                       |
|   30 | `n`   | Do not restrict field values before filtering.         |

## `configs/f320/initial.f320_to_t42` and `initial.f320_to_f320`

The initial-grid files answer the `initial` grid-definition prompts used by
the tracking configuration. `initial.f320_to_t42` is:

| Line | Value | Meaning                                        |
| ---: | ----- | ---------------------------------------------- |
|    1 | `0`   | Use the standard latitude/longitude grid form. |
|    2 | `y`   | Use the supplied grid as a periodic grid.      |
|    3 | `n`   | Do not translate grid coordinates.             |
|    4 | `y`   | Treat the X direction as periodic.             |
|    5 | `g`   | Use the global domain.                         |
|    6 | `n`   | Do not apply a projection.                     |
|    7 | `1`   | First X index.                                 |
|    8 | `129` | X extent including TRACK's cyclic endpoint.    |
|    9 | `1`   | First Y index.                                 |
|   10 | `64`  | Y extent.                                      |

`initial.f320_to_f320` has the same meanings, with line 8 equal to `1281` and
line 10 equal to `640`. The extra X endpoint is an internal cyclic-grid
convention; physical T42 output has 128 distinct longitudes.

## `configs/*/RUNDATIN.era5_MSLP_latlng` and `_A`

These files are consumed by `RUN_AT.in`. The path line contains a `%INITIAL%`
placeholder; `RUN_AT.in` substitutes the selected initial-field filename. The
numbered answers configure thresholding, feature detection, MGE linking, and
post-MGE filtering. The two files differ in the threshold-polarity branch.

### Common/base branch (`RUNDATIN.era5_MSLP_latlng`)

| Line | Value                 | Meaning                                                           |
| ---: | --------------------- | ----------------------------------------------------------------- |
|    1 | `n`                   | Do not load a country map.                                        |
|    2 | `1`                   | Use initial field 1.                                              |
|    3 | `PATH/data/%INITIAL%` | Initial-field path template; the wrapper substitutes `%INITIAL%`. |
|    4 | `n`                   | Do not use the analysis menu.                                     |
|    5 | `n`                   | Do not use existing object data.                                  |
|    6 | `n`                   | Do not calculate a tendency field.                                |
|    7 | `y`                   | Apply field scaling.                                              |
|    8 | `0.01`                | Field scale.                                                      |
|    9 | `n`                   | Do not apply an offset.                                           |
|   10 | `1.`                  | Threshold magnitude after scaling.                                |
|   11 | `n`                   | Do not add another field.                                         |
|   12 | `1`                   | Detect maxima in this branch.                                     |
|   13 | `n`                   | Do not invert the sign by hemisphere.                             |
|   14 | `2#`                  | Segment-start substitution marker.                                |
|   15 | `1`                   | Frame interval.                                                   |
|   16 | `30!`                 | Segment-end substitution marker.                                  |
|   17 | `e`                   | Edge connectivity.                                                |
|   18 | `y`                   | Search object boundaries.                                         |
|   19 | `y`                   | Make the tracking grid periodic in X.                             |
|   20 | `2`                   | Remove objects with two or fewer points.                          |
|   21 | `7`                   | Surface-fit/local-optimization feature-point method.              |
|   22 | `n`                   | Do not use anisotropy.                                            |
|   23 | `0`                   | SMOOPY interpolation/smoothing choice.                            |
|   24 | `n`                   | Do not apply the unphysical-object filter.                        |
|   25 | `n`                   | Do not retain/filter only the largest feature.                    |
|   26 | `n`                   | Do not apply the too-close feature filter.                        |
|   27 | `n`                   | Do not use a time-average field.                                  |
|   28 | `0`                   | Boundary-maxima exclusion.                                        |
|   29 | `0.`                  | Additional smoothing parameter.                                   |
|   30 | `y`                   | Use constrained feature optimization.                             |
|   31 | `d`                   | Use the default constraint set.                                   |
|   32 | `n`                   | Do not write object data.                                         |
|   33 | `n`                   | Do not make a first-frame plot.                                   |
|   34 | `0`                   | Additional-frame plot factor.                                     |
|   35 | `n`                   | Do not rotate feature locations.                                  |
|   36 | `n`                   | Do not use existing tracks.                                       |
|   37 | `0.2`                 | First MGE cost weight; TRACK halves it internally.                |
|   38 | `0.8`                 | Second MGE cost weight.                                           |
|   39 | `n`                   | Do not search for missing frames during MGE.                      |
|   40 | `y`                   | Use regional constraints.                                         |
|   41 | `y`                   | Use adaptive constraints.                                         |
|   42 | `6.5`                 | Regional upper-bound displacement in degrees.                     |
|   43 | `1.0`                 | Base adaptive smoothness parameter.                               |
|   44 | `n`                   | Do not make an initial-track plot.                                |
|   45 | `n`                   | Do not use a different initialization.                            |
|   46 | `n`                   | Do not search for missing tracks after MGE.                       |
|   47 | `y`                   | Apply post-MGE zonal/adaptive filtering.                          |
|   48 | `n`                   | Do not search for missing tracks in the post pass.                |
|   49 | `y`                   | Apply post-pass regional filtering.                               |
|   50 | `y`                   | Apply post-pass adaptive filtering.                               |
|   51 | `0`                   | Minimum plot-point count.                                         |
|   52 | `0`                   | Track-plot selection.                                             |
|   53 | `n`                   | Do not repeat the tracking setup.                                 |

The apparent `30!` and `2#` tokens are intentional `RUN_AT` substitution
markers, not literal frame numbers in the final generated segment scripts.

### Minimum branch (`RUNDATIN.era5_MSLP_latlng_A`)

|  Line | Value                           | Meaning                                              |
| ----: | ------------------------------- | ---------------------------------------------------- |
| 1--11 | Same values as the base branch. | Same source, scaling, offset, and threshold setup.   |
|    12 | `0`                             | Detect minima in this branch.                        |
|    13 | `n`                             | Retain original field values.                        |
|    14 | `n`                             | Do not invert the sign by hemisphere.                |
|    15 | `2#`                            | Segment-start substitution marker.                   |
|    16 | `1`                             | Frame interval.                                      |
|    17 | `30!`                           | Segment-end substitution marker.                     |
|    18 | `e`                             | Edge connectivity.                                   |
|    19 | `y`                             | Search object boundaries.                            |
|    20 | `y`                             | Make the tracking grid periodic in X.                |
|    21 | `2`                             | Remove objects with two or fewer points.             |
|    22 | `7`                             | Surface-fit/local-optimization feature-point method. |
|    23 | `n`                             | Do not use anisotropy.                               |
|    24 | `0`                             | SMOOPY interpolation/smoothing choice.               |
|    25 | `n`                             | Do not apply the unphysical-object filter.           |
|    26 | `n`                             | Do not retain/filter only the largest feature.       |
|    27 | `n`                             | Do not apply the too-close feature filter.           |
|    28 | `n`                             | Do not use a time-average field.                     |
|    29 | `0`                             | Boundary-maxima exclusion.                           |
|    30 | `0.`                            | Additional smoothing parameter.                      |
|    31 | `y`                             | Use constrained feature optimization.                |
|    32 | `d`                             | Use the default constraint set.                      |
|    33 | `n`                             | Do not write object data.                            |
|    34 | `n`                             | Do not make a first-frame plot.                      |
|    35 | `0`                             | Additional-frame plot factor.                        |
|    36 | `n`                             | Do not rotate feature locations.                     |
|    37 | `n`                             | Do not use existing tracks.                          |
|    38 | `0.2`                           | First MGE cost weight; TRACK halves it internally.   |
|    39 | `0.8`                           | Second MGE cost weight.                              |
|    40 | `n`                             | Do not search for missing frames during MGE.         |
|    41 | `y`                             | Use regional constraints.                            |
|    42 | `y`                             | Use adaptive constraints.                            |
|    43 | `6.5`                           | Regional upper-bound displacement in degrees.        |
|    44 | `1.0`                           | Base adaptive smoothness parameter.                  |
|    45 | `n`                             | Do not make an initial-track plot.                   |
|    46 | `n`                             | Do not use a different initialization.               |
|    47 | `n`                             | Do not search for missing tracks after MGE.          |
|    48 | `y`                             | Apply post-MGE zonal/adaptive filtering.             |
|    49 | `n`                             | Do not search for missing tracks in the post pass.   |
|    50 | `y`                             | Apply post-pass regional filtering.                  |
|    51 | `y`                             | Apply post-pass adaptive filtering.                  |
|    52 | `0`                             | Minimum plot-point count.                            |
|    53 | `0`                             | Track-plot selection.                                |
|    54 | `n`                             | Do not repeat the tracking setup.                    |

For the minimum branch, type 0 changes the internal sign to negative. With
the scale `.01` and threshold magnitude `1`, this represents a cyclone
threshold of -100 Pa in the source value units.

## `configs/*/zone.dat0`

| Line | Value                       | Meaning                                                                       |
| ---: | --------------------------- | ----------------------------------------------------------------------------- |
|    1 | `3`                         | Number of latitude zones.                                                     |
|    2 | `0.0 360.0 -90.0 -20.0 6.5` | Global longitude span, southern latitude zone, 6.5-degree displacement bound. |
|    3 | `0.0 360.0 -20.0 20.0 3.0`  | Global longitude span, tropical latitude zone, 3.0-degree displacement bound. |
|    4 | `0.0 360.0 20.0 90.0 6.5`   | Global longitude span, northern latitude zone, 6.5-degree displacement bound. |

For geodesic constraints TRACK converts the longitude/latitude bounds and the
distance bound from degrees to radians internally.

## `configs/*/adapt.dat0`

| Line | Value             | Meaning                                                                  |
| ---: | ----------------- | ------------------------------------------------------------------------ |
|    1 | `1.0 2.0 5.0 8.0` | Piecewise adaptive-constraint distance breakpoints, in degrees.          |
|    2 | `1.0 0.3 0.1 0.0` | Piecewise adaptive upper-bound values associated with those breakpoints. |

TRACK converts the distance breakpoints to radians and constructs the
piecewise slopes/intercepts used by the adaptive track-smoothness constraint.

## `scripts/preflight_input.py`

This is a validation helper rather than a TRACK answer stream. Its command
line arguments identify a NetCDF file and expected grid dimensions; it checks
the dimensions, six-hourly cadence, time metadata, `msl` presence, and pressure
units before a run. It does not change the input file.

## Segment substitution performed by `RUN_AT.in`

The tracking wrapper starts at frame 1 and uses overlapping segments. For
`-n=1,62,2`, `-n=1,62,6`, or `-n=1,62,24`, the first segment is frames 1--62;
each ordinary next segment starts at the previous end minus `BACK` (2) and
extends by `I=61` frames; the final segment adds the `FOREWARD` tail (15) to
the end sentinel. Thus these are overlapping-and-spliced runs, not
independent non-overlapping chunks. The generated segment files contain the
actual substituted numbers in place of `2#`, `30!`, and the next-start marker.
