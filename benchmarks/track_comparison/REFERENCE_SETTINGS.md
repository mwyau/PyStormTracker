# TRACK 1.5.4 benchmark reference settings

This file records the scientific and orchestration settings that are implicit in
the checked-in TRACK answer transcripts. It supplements `README.md`; it does not
replace the exact files under `configs/f320/`.

The purpose is to make the benchmark auditable without requiring a reader to
reverse-engineer a sequence of bare interactive answers.

The eight-case benchmark remains:

- four F320 2024 cases: F320 -> T42 vs F320 -> F320 synthesis, January vs full year;
- four regular-lat/lon extensions: 2.5° vs 0.25°, December 2025 vs DJF 2025-2026;
- all cases are MSLP;
- the regular-grid extensions target the same T42 Gaussian tracking grid;
- polar-identification scenarios are intentionally outside this benchmark.

## 1. Spectral preparation used by the F320 reference cases

Both checked-in spectral answer streams analyze the F320 input at truncation
42 and produce two total-wavenumber bands:

```text
band000: l = 0..5
band001: l = 6..42
```

The tracking workflow keeps `band001` and removes `band000`.

Therefore the actual tracked field is **T6-42**, not generic "T42" or
unfiltered MSLP.

The PyStormTracker benchmark does not read TRACK's binary filtered-field
product. It reads the raw source NetCDF and constructs the equivalent T6-42
field in memory with PyStormTracker's public `SHTFilter`:

```text
lmin = 6
lmax = 42
taper_val = 0.1
geometry = "auto"
T42 cases: out_geometry = "GL", out_ntheta = 64, out_nphi = 128
F320 cases: retain the detected F320 Gaussian geometry
```

This is a PST-native preprocessing measurement; it is configured to represent
the TRACK T6-42 workflow but is not a claim that the two numerical transform
implementations produce bit-identical fields.

Both spectral streams enable the same coefficient taper:

```text
Hoskins taper at lmax=42: 0.1
```

The two supplied spatial configurations differ only in where the filtered field
is reconstructed:

| Answer stream           | Filtered band | Reconstructed grid          | Physical grid size |
| ----------------------- | ------------- | --------------------------- | ------------------ |
| `specfilt_f320_to_t42`  | T6-42         | new T42 Gaussian grid       | 128 x 64           |
| `specfilt_f320_to_f320` | T6-42         | original F320 Gaussian grid | 1280 x 640         |

`full` therefore means **T6-42 reconstructed on the full F320 spatial grid**.
It does not mean unfiltered.

Expected spectral products before the guide renames/removes them are:

```text
outdat/specfil.filtT42_band000    l=0..5
outdat/specfil.filtT42_band001    l=6..42
```

## 2. TRACK answer transcripts are ordered interactive input

`RUNDATIN*.in` and `specfilt*` are not key/value configuration files. They are
ordered transcripts of answers that would otherwise be supplied interactively.
Meaning comes from answer order and the TRACK 1.5.4 source/scripts that consume
them.

Do not reorder, delete, normalize, or "clean up" lines.

The checked-in RUNDATIN transcripts contain script-substitution markers such as:

```text
PATH/data/%INITIAL%
2#
30!
```

For runtime copies:

- replace only `PATH` with the absolute disposable TRACK worktree path;
- preserve `%INITIAL%`; the orchestration substitutes the selected `initial.*`;
- preserve `#` and `!`; the TRACK scripts edit those markers during the run;
- do not replace `%INITIAL%` manually;
- do not convert `2#` to `2` or `30!` to `30`.

The base and `_A` RUNDATIN files form a pair used by the sign branches. Keep the
pair together rather than treating `_A` as a separate benchmark scenario.

## 3. Decoded MSLP tracking settings

The checked-in latitude/longitude RUNDATIN pair represents the following
scientific choices.

| Setting                                      |                  Reference value | Meaning                                             |
| -------------------------------------------- | -------------------------------: | --------------------------------------------------- |
| field scale                                  |                           `0.01` | TRACK scales Pa numerically before thresholding     |
| threshold magnitude after scaling            |                            `1.0` | paired sign branch determines minima/maxima         |
| cyclone threshold on original filtered field |                        `-100 Pa` | minimum branch equivalent after `0.01` scale        |
| longitude periodicity                        |                          enabled | global cyclic x/longitude                           |
| boundary search                              |                          enabled | source boundary-handling path is active             |
| source object-size input                     |                              `2` | objects with `point_num <= 2` are removed           |
| minimum retained object size                 |                    3 grid points | consequence of the source test above                |
| feature-point location                       | surface fit + local optimization | rectangular TRACK/SMOOPY refinement path            |
| spline smoothing parameter                   |                              `0` | interpolation/no additional smoothing               |
| constrained feature optimization             |                          enabled | source constraints remain active                    |
| MGE displacement weight `w1`                 |                            `0.2` | displacement contribution to trajectory cost        |
| MGE smoothness weight `w2`                   |                            `0.8` | smoothness contribution to trajectory cost          |
| missing input frames                         |                         disabled | reference sequence is treated as complete           |
| global/reference `dmax`                      |                           `6.5°` | upper displacement scale before regional adjustment |
| base `phimax`                                |                            `1.0` | base smoothness constraint                          |
| regional displacement zones                  |                          enabled | values from `zone.dat0`                             |
| adaptive smoothness                          |                          enabled | values from `adapt.dat0`                            |
| post-MGE constraint filtering                |                          enabled | regional/adaptive constraints are reapplied         |

The corrected PST reproduction run uses these historical validation settings
explicitly:

```text
feature_refinement = "bspline"  # current name for rectangular SMOOPY
track_smoopy_optimization_scale = 0.01
mge_max_iterations = 10
min_track_points = 1             # retain raw one- and two-point tracks
segment_frames = 62
```

The ordinary public defaults remain unchanged (`scale=1.0`, `MGE=3`, and
`min_track_points=3`). These benchmark values are recorded so raw PST output
is compared with TRACK's raw `tr_trs_neg` output on the same track-retention
basis.

The threshold transcript contains a positive magnitude because the sign is
handled by the paired minimum/maximum branch. Do **not** edit the transcript to
replace `1.0` by `-1.0`.

## 4. Regional displacement and adaptive smoothness tables

`zone.dat0` defines:

```text
longitude 0..360°, latitude -90..-20°  -> dmax = 6.5°
longitude 0..360°, latitude -20.. 20°  -> dmax = 3.0°
longitude 0..360°, latitude  20.. 90°  -> dmax = 6.5°
```

`adapt.dat0` defines:

```text
distance (deg): 1.0   2.0   5.0   8.0
phimax:         1.0   0.3   0.1   0.0
```

TRACK interpolates the adaptive table as part of its trajectory constraint.
These files are part of the benchmark definition, not optional tuning inputs.

## 5. Cyclic endpoint representation

The physical Gaussian grids contain:

```text
T42:  128 longitudes x 64 latitudes
F320: 1280 longitudes x 640 latitudes
```

The rectangular TRACK initialization transcripts use a duplicated cyclic
longitude endpoint internally:

```text
128 physical longitudes  -> indices through 129
1280 physical longitudes -> indices through 1281
```

Therefore:

```text
initial.f320_to_t42  <-> T6-42 field reconstructed on T42
initial.f320_to_f320 <-> T6-42 field reconstructed on F320
```

Do not reduce the initialization range to 128/1280 merely because those are the
physical longitude counts.

The mapping is strict:

```text
initial.f320_to_t42
    -> ERA5_mslp_6hr_2024_DET_T42filt.dat

initial.f320_to_f320
    -> ERA5_mslp_6hr_2024_DET_T42filt_full.dat
```

## 6. Meaning of `-n=1,62,N`

`62` is part of TRACK's bounded tracking/splicing orchestration. It is not a
request to analyze only 62 frames.

For the 1464-frame full-year reference run:

```text
-n=1,62,24
```

creates 24 overlapping tracking chunks. The first chunk covers frames 1..62;
later chunks begin two frames before the previous chunk ends, and the final
chunk is extended to cover the remaining tail.

The shorter benchmark variants retain the same 62-frame orchestration:

```text
124 frames  -> -n=1,62,2
360 frames  -> -n=1,62,6
1464 frames -> -n=1,62,24
```

This overlap/splice behavior is why a benchmark run must use a true shorter
input file for a shorter period rather than truncating trajectories from a
longer run.

## 7. Output interpretation

A successful MSLP run produces the corresponding raw and post-filtered sign
branches, normally gzip-compressed by the wrapper:

```text
tr_trs_neg[.gz]   raw tracked minima / cyclones
tr_trs_pos[.gz]   raw tracked maxima / anticyclones
ff_trs_neg[.gz]   post-RSPLICE minima / cyclones
ff_trs_pos[.gz]   post-RSPLICE maxima / anticyclones
```

For cyclone comparison, use the negative branch.

Keep `tr_trs_neg` when diagnosing tracking itself. `ff_trs_neg` includes the
post-tracking activity filter and therefore should not be compared with an
unfiltered PyStormTracker trajectory population.

To inspect compressed output without changing it:

```bash
gzip -dc tr_trs_neg.gz | less
gzip -dc ff_trs_neg.gz | less
```

## 8. Exact RSPLICE interpretation

A common shorthand describes the post-filter as roughly "longer than two days
and farther than 1000 km". For source reproduction, use the exact configured
TRACK tests instead:

```text
minimum retained point count: 8
minimum endpoint geodesic separation: 10 degrees
```

Boundary behavior matters:

- TRACK removes a trajectory when its point count is **less than 8**; exactly 8
  points is retained;
- TRACK removes a trajectory when endpoint separation is **less than 10°**;
  exactly 10° is retained.

This is a point-count plus endpoint-separation test. Do not silently reinterpret
it as literal elapsed hours or cumulative path length.

## 9. Regular-grid extension invariant

The 2.5° and 0.25° regular-lat/lon inputs are extensions, not supplied F320
answer-stream cases.

For both regular-grid resolutions the intended experiment is:

```text
regular global lat/lon NetCDF
    -> TRACK global least-squares spectral decomposition (method 0)
    -> same T6-42 band selection and 0.1 taper
    -> T42 Gaussian reconstruction
    -> initial.T42
    -> same RUNDATIN / zone / adaptive tracking settings
```

The final answer streams are checked in under
`configs/regular_latlon/specfilt_regular_*_T42`. Their complete line-by-line
meaning is in [`INPUT_SEMANTICS.md`](INPUT_SEMANTICS.md). The actual regular
files contain int64 `valid_time`, which TRACK 1.5.4 rejects; the benchmark uses
compatibility views with only that coordinate converted to int32 hours since
1900\. The pressure and spatial coordinate values are unchanged.

## 10. Build/runtime troubleshooting invariants

If the build fails on stale absolute/system header dependencies, first confirm
the checked-in build-only patch is applied:

```bash
git diff -- lib/src/Makefile lib/src/Makefile.opt src/Makefile.linux
```

The generated legacy dependency blocks should be guarded by:

```make
ifeq ($(TRACK_LEGACY_DEPS),1)
...
endif
```

For the normal benchmark build, leave `TRACK_LEGACY_DEPS=0`.

If compilation or linking still fails:

1. diagnose the first compiler/linker error;
1. inspect TRACK 1.5.4's upstream `INSTALL`, `config.make`, and Makefiles;
1. verify compiler and NetCDF C/Fortran development paths;
1. do not patch scientific TRACK source merely to bypass an environment error.

Runtime requirements include:

```bash
command -v csh
export PATH="$PWD:$PWD/bin:$PATH"
```

Run `master` from the disposable TRACK root.

If `specfil.filtT42_band001` is missing, check that:

- the source NetCDF is visible under `indat/` or through the expected link;
- the spectral answer stream is in the TRACK root;
- `outdat/` exists and is writable;
- `bin/track.linux` is the intended executable;
- the command is being run from the TRACK root.

## 11. Minimum provenance to retain with benchmark results

For every reportable benchmark campaign, retain at least:

```text
TRACK tag and exact commit SHA
TRACK source diff after applying the build-only patch
build-patch SHA256
compiler versions
NetCDF C/Fortran versions and flags
compiled bin/track.linux SHA256
input NetCDF SHA256
input dimensions, timestamps, cadence, variable and units
checksums of the checked-in F320 configuration
exact spectral answer stream used
exact initial.* file used
exact RUNDATIN pair used
complete master command
whether output is raw tr_trs_* or post-RSPLICE ff_trs_*
warm/cold filesystem-cache policy
all individual timing repetitions
```

A minimal command set includes:

```bash
git rev-parse HEAD
git diff --check
sha256sum bin/track.linux
sha256sum "$ERA5"
```

The benchmark is only comparable across machines/runs if the binary, data,
scientific configuration, and timing boundary are all explicit.
