# TRACK 1.5.4 MSLP benchmark results

This is the final serial validation record for the eight benchmark cases. Each
case has one prior complete measured record plus two additional clean measured
repetitions; all three records and the medians are retained externally. The
checked-in settings are under benchmarks/track_comparison/; raw ERA5 files,
the disposable TRACK tree, and generated products are outside Git in the local
validation area. The PST continuation status is recorded separately in
[`PROGRESS.md`](PROGRESS.md); the matrix below remains the completed TRACK
reference, not a completed PST campaign.

## Provenance

```text
TRACK tag:       TRACK-1.5.4
TRACK commit:    6ded301a5f5183d73e5b49c16019024b9a53eff7
runtime tree:    /home/albert/TRACK-run
validation tree: /home/albert/PyStormTracker-Validation
host:            opencode; Ubuntu 26.04 LTS; Linux 7.0.14-8-pve
CPU:             AMD Ryzen 9 5950X, 16 cores / 32 threads
binary SHA256:   3fd4b0e9bb9e5d6924d8069491ce396fa0294b1f4eb194c035666149eeac149d
build patch:     cd2fb9dbd7bef7da66ac39666bb7327b5c6fc70be66f30663b970e26dd524d4d
repeat runner:    18399d557498723c154861bd862a2f3a9f1ece4e417a3f03ae4255d663303e12
compiler:        gcc/gfortran 15.2.0
make:            GNU Make 4.4.1
NetCDF C:        4.9.3, NetCDF Fortran 4.6.2, NetCDF-4 support: yes
runtime source diff: only lib/src/Makefile, lib/src/Makefile.opt, and
                     src/Makefile.linux; 9 added build-dependency lines
```

The unpatched build failed before producing a binary because generated
dependency lists referenced stale /usr/include/sys/cdefs.h prerequisites. The
checked-in patch only wraps those generated dependency lists in
TRACK_LEGACY_DEPS=1; the normal build used TRACK_LEGACY_DEPS=0. The patched
build completed and git apply --check passes against the pristine TRACK
checkout.

## Independent source findings

- src/track.c:749-762 confirms option 4: least squares 0, fast spectral
  transform 1, and limited-area DCT 2.
- The F320 streams select the fast transform and the regular streams select the
  global least-squares path in src/spectral_filter.c.
- The regular source files contain msl, number, and expver; line 7 of the
  regular streams must therefore answer msl. The F320 file contains only msl
  and auto-selects its sole field.
- TRACK's NetCDF reader rejects the regular files' int64 valid_time variable
  (data type 10). Direct runs on those original files fail before filtering.
  The accepted regular runs use compatibility views that change only the time
  coordinate to int32 hours since 1900; pressure, latitude, longitude, and data
  values are unchanged.
- The regular least-squares path creates a T42 Gaussian output with 128 by 64
  physical points. The F320 -> F320 path synthesizes 1280 by 640 physical
  points.
- RUN_AT.in performs overlapping segment processing and then RSPLICE. The
  -nums values are 1,62,2, 1,62,6, and 1,62,24 for 124, 360, and 1464 frames
  respectively; they are not independent non-overlapping chunks.

The complete line-by-line answer semantics are in INPUT_SEMANTICS.md.

## Final three-repeat benchmark matrix

Each row reports the median of three complete serial measured runs. Spectral
and tracking are individual GNU `/usr/bin/time` wall-clock stages. `Total`
means the per-run stage sum (spectral + tracking); `workflow` is the separately
measured elapsed time around both stages and includes wrapper handoff overhead.
RSS columns are median maximum resident set sizes. Runs were executed one at a
time with a warm-filesystem-cache policy and no rebuild between repetitions.

Raw and post values are `tracks / points` for the negative (cyclone/minimum)
branch. Positive-branch counts are retained in each external `products.tsv`.

| Case                       | Source/grid     | Frames | Spectral method | Output / segmentation   | Raw neg tracks/points | Post neg tracks/points | Spectral s | Tracking s |  Total s | Workflow s | RSS spectral/tracking KB |
| -------------------------- | --------------- | -----: | --------------- | ----------------------- | --------------------: | ---------------------: | ---------: | ---------: | -------: | ---------: | -----------------------: |
| F320 -> T42, January       | F320 Gaussian   |    124 | fast            | T42 128x64 / 1,62,2     |            709 / 5127 |             126 / 2455 |      1.400 |      3.520 |    4.930 |      4.949 |            42824 / 17608 |
| F320 -> T42, full year     | F320 Gaussian   |   1464 | fast            | T42 128x64 / 1,62,24    |          7761 / 60654 |           1471 / 30998 |     18.470 |     41.270 |   59.700 |     59.722 |            40736 / 42024 |
| F320 -> F320, January      | F320 Gaussian   |    124 | fast            | F320 1280x640 / 1,62,2  |            779 / 5529 |             132 / 2605 |      3.410 |    162.200 |  165.630 |    165.647 |          42008 / 2272428 |
| F320 -> F320, full year    | F320 Gaussian   |   1464 | fast            | F320 1280x640 / 1,62,24 |          8595 / 65539 |           1544 / 32760 |     44.790 |   1930.360 | 1973.040 |   1973.061 |          41012 / 2273060 |
| regular 2.5 deg, December  | regular lat/lon |    124 | least squares   | T42 128x64 / 1,62,2     |            745 / 4965 |             128 / 2271 |      9.010 |      3.670 |   12.680 |     12.697 |            31220 / 17568 |
| regular 2.5 deg, DJF       | regular lat/lon |    360 | least squares   | T42 128x64 / 1,62,6     |          2006 / 14609 |             347 / 6994 |     18.510 |     10.780 |   29.260 |     29.288 |            41860 / 42048 |
| regular 0.25 deg, December | regular lat/lon |    124 | least squares   | T42 128x64 / 1,62,2     |            729 / 4944 |             127 / 2328 |    551.790 |      3.470 |  555.240 |    555.266 |           156284 / 17620 |
| regular 0.25 deg, DJF      | regular lat/lon |    360 | least squares   | T42 128x64 / 1,62,6     |          1971 / 14496 |             355 / 7153 |   1195.850 |      9.990 | 1205.870 |   1205.891 |           162200 / 41916 |

The four compressed trajectory products were present and non-empty after every
accepted correctness/measured run. The durable negative and positive sign
summary from the retained run records is:

| Case                       | Raw minimum tracks/points | Raw maximum tracks/points | Post minimum tracks/points | Post maximum tracks/points |
| -------------------------- | ------------------------: | ------------------------: | -------------------------: | -------------------------: |
| F320 -> T42, January       |                709 / 5127 |                652 / 3935 |                 126 / 2455 |                  80 / 1572 |
| F320 -> T42, full year     |              7761 / 60654 |              7320 / 49329 |               1471 / 30998 |               1057 / 20739 |
| F320 -> F320, January      |                779 / 5529 |                748 / 4298 |                 132 / 2605 |                  87 / 1657 |
| F320 -> F320, full year    |              8595 / 65539 |              8288 / 53444 |               1544 / 32760 |               1106 / 21656 |
| regular 2.5 deg, December  |                745 / 4965 |                639 / 4177 |                 128 / 2271 |                 103 / 1777 |
| regular 2.5 deg, DJF       |              2006 / 14609 |              1867 / 12260 |                 347 / 6994 |                 287 / 5272 |
| regular 0.25 deg, December |                729 / 4944 |                600 / 4117 |                 127 / 2328 |                 102 / 1835 |
| regular 0.25 deg, DJF      |              1971 / 14496 |              1813 / 12083 |                 355 / 7153 |                 285 / 5385 |

Each external `products.tsv` also records the compressed product path and byte
size for `tr_trs_neg`, `tr_trs_pos`, `ff_trs_neg`, and `ff_trs_pos` for each
run.

## Individual measured repetitions

The tuple in each stage column is `wall / user / system / max-RSS`, with
seconds for the first three values and KB for RSS. `Total sum` is spectral plus
tracking for that repetition; `workflow` is the separately observed elapsed
wall time. `run1` is the previously accepted complete record copied into the
repeat layout; `run2` and `run3` were rerun after the untimed correctness
campaign. The external records are under
`/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats/`.

| Case                       | Run | Spectral wall/user/sys/RSS          | Tracking wall/user/sys/RSS            | Total sum | Workflow |
| -------------------------- | --- | ----------------------------------- | ------------------------------------- | --------: | -------: |
| F320 -> T42, January       | 1   | 1.380 / 1.23 / 0.15 / 42872         | 3.550 / 3.37 / 0.18 / 17592           |     4.930 |    4.949 |
| F320 -> T42, January       | 2   | 1.400 / 1.19 / 0.20 / 42824         | 3.520 / 3.37 / 0.15 / 17752           |     4.920 |    4.941 |
| F320 -> T42, January       | 3   | 1.420 / 1.21 / 0.20 / 41840         | 3.520 / 3.33 / 0.19 / 17608           |     4.940 |    4.960 |
| F320 -> T42, full year     | 1   | 18.200 / 15.20 / 2.99 / 41096       | 41.490 / 40.25 / 1.23 / 41928         |    59.690 |   59.712 |
| F320 -> T42, full year     | 2   | 18.470 / 14.88 / 3.57 / 40736       | 41.270 / 40.05 / 1.21 / 42024         |    59.740 |   59.764 |
| F320 -> T42, full year     | 3   | 18.470 / 14.85 / 3.59 / 40188       | 41.230 / 40.00 / 1.22 / 42060         |    59.700 |   59.722 |
| F320 -> F320, January      | 1   | 3.390 / 3.13 / 0.25 / 41824         | 161.280 / 117.85 / 43.34 / 2272956    |   164.670 |  164.691 |
| F320 -> F320, January      | 2   | 3.410 / 3.12 / 0.24 / 42260         | 162.690 / 119.06 / 43.56 / 2272420    |   166.100 |  166.122 |
| F320 -> F320, January      | 3   | 3.430 / 3.15 / 0.24 / 42008         | 162.200 / 117.69 / 44.43 / 2272428    |   165.630 |  165.647 |
| F320 -> F320, full year    | 1   | 45.950 / 37.79 / 4.14 / 40880       | 1941.140 / 1394.89 / 545.16 / 2273060 |  1987.090 | 1987.120 |
| F320 -> F320, full year    | 2   | 44.790 / 36.51 / 4.03 / 41140       | 1927.520 / 1388.55 / 537.90 / 2273140 |  1972.310 | 1972.337 |
| F320 -> F320, full year    | 3   | 42.680 / 37.68 / 4.16 / 41012       | 1930.360 / 1391.35 / 537.96 / 2272908 |  1973.040 | 1973.061 |
| regular 2.5 deg, December  | 1   | 9.010 / 8.98 / 0.02 / 31220         | 3.670 / 3.49 / 0.18 / 17640           |    12.680 |   12.697 |
| regular 2.5 deg, December  | 2   | 8.910 / 8.88 / 0.03 / 31132         | 3.640 / 3.46 / 0.18 / 17552           |    12.550 |   12.569 |
| regular 2.5 deg, December  | 3   | 9.060 / 9.03 / 0.02 / 31308         | 3.670 / 3.49 / 0.18 / 17568           |    12.730 |   12.750 |
| regular 2.5 deg, DJF       | 1   | 18.480 / 18.40 / 0.04 / 41860       | 10.780 / 10.41 / 0.37 / 42120         |    29.260 |   29.288 |
| regular 2.5 deg, DJF       | 2   | 18.510 / 18.44 / 0.05 / 42060       | 10.740 / 10.37 / 0.37 / 42048         |    29.250 |   29.269 |
| regular 2.5 deg, DJF       | 3   | 18.530 / 18.47 / 0.04 / 41880       | 10.790 / 10.42 / 0.37 / 42036         |    29.320 |   29.346 |
| regular 0.25 deg, December | 1   | 556.130 / 554.76 / 1.13 / 156456    | 3.470 / 3.30 / 0.18 / 17884           |   559.600 |  559.629 |
| regular 0.25 deg, December | 2   | 547.200 / 545.85 / 1.11 / 156212    | 3.460 / 3.29 / 0.17 / 17492           |   550.660 |  550.688 |
| regular 0.25 deg, December | 3   | 551.790 / 550.41 / 1.14 / 156284    | 3.450 / 3.27 / 0.19 / 17620           |   555.240 |  555.266 |
| regular 0.25 deg, DJF      | 1   | 1194.240 / 1183.47 / 10.28 / 162200 | 9.990 / 9.59 / 0.40 / 41936           |  1204.230 | 1204.250 |
| regular 0.25 deg, DJF      | 2   | 1210.900 / 1198.84 / 11.45 / 162080 | 9.940 / 9.57 / 0.37 / 41912           |  1220.840 | 1220.865 |
| regular 0.25 deg, DJF      | 3   | 1195.850 / 1184.72 / 10.56 / 162272 | 10.020 / 9.60 / 0.41 / 41916          |  1205.870 | 1205.891 |

## Output geometry

All eight band001 products passed the tracking geometry check:

```text
F320 -> T42:       128  64  frames
F320 -> F320:     1280 640  frames
regular -> T42:    128  64  frames
```

The band definitions are T0-5 (band000) and T6-42 (band001), with the Hoskins
coefficient filter at 0.1 applied to both bands. The tracker uses band001 and
the matching F320 -> T42 or F320 -> F320 initial-grid file.

## Exact complete-run command pattern

The checked-in runner
`benchmarks/track_comparison/scripts/run_track_repeat.sh` maps each scientific
case to its input, stream, initialization, and `-nums` values. The measured
repetitions were invoked one at a time as follows:

```bash
RESULT_BASE=/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats \
    benchmarks/track_comparison/scripts/run_track_repeat.sh \
    <case> run2 measure
RESULT_BASE=/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats \
    benchmarks/track_comparison/scripts/run_track_repeat.sh \
    <case> run3 measure
```

The stage command pattern inside the runner is:

```bash
t0=$(date +%s%N)
/usr/bin/time -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULT/spectral.time" \
    bin/track.linux -i "$INPUT" -f "$EXT" < "$SPECTRAL_STREAM" \
    > "$RESULT/spectral.log" 2>&1
ln -s "$TRACK_ROOT/outdat/specfil.${EXT}_band001" \
    "$TRACK_ROOT/indat/${EXT}.dat"
/usr/bin/time -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULT/tracking.time" \
    ./master -excn=track.linux -fext="$EXT" -inpf="${EXT}.dat" \
    -jd=/tmp/codex-run-at.in -kinit="$INITIAL" -nums="$NUMS" \
    -outdir="$RESULT/tracking" -cdir="$EXT" -rfil=RUN_ \
    -s=RUNDATIN.era5_MSLP_latlng \
    > "$RESULT/tracking/master.log" 2>&1
t2=$(date +%s%N)
```

The accepted records are in the external validation tree at
`/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/repeats/`.
No raw runtime output is checked into this repository.

## Input provenance and limitation

The source files independently inspected with `ncdump -k` and `ncdump -h` were:

| Dataset                    | Path                                                                                                                             | Format                      | Dimensions (x/y/time) | Time range and cadence                   | Field / units   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | --------------------- | ---------------------------------------- | --------------- |
| F320 full                  | `/home/albert/PyStormTracker-Reference-Data/era5-2024/ERA5_mslp_6hr_2024_DET.nc`                                                 | NetCDF 64-bit offset        | 1280 / 640 / 1464     | 2024-01-01 00 to 2024-12-31 18, 6-hourly | `msl`, float Pa |
| F320 January               | `/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/inputs/ERA5_mslp_6hr_2024-01_DET.nc`                   | NetCDF 64-bit offset        | 1280 / 640 / 124      | 2024-01-01 00 to 2024-01-31 18, 6-hourly | `msl`, float Pa |
| regular 2.5 full           | `/home/albert/.cache/pystormtracker/era5_msl_2025-2026_djf_2.5x2.5.nc`                                                           | NetCDF-4                    | 144 / 73 / 360        | 2025-12-01 00 to 2026-02-28 18, 6-hourly | `msl`, float Pa |
| regular 2.5 December view  | `/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/inputs/track-compatible/era5_msl_2025-12_2.5x2.5.nc`   | NetCDF-4 compatibility view | 144 / 73 / 124        | 2025-12-01 00 to 2025-12-31 18, 6-hourly | `msl`, float Pa |
| regular 0.25 full          | `/home/albert/.cache/pystormtracker/era5_msl_2025-2026_djf_0.25x0.25.nc`                                                         | NetCDF-4                    | 1440 / 721 / 360      | 2025-12-01 00 to 2026-02-28 18, 6-hourly | `msl`, float Pa |
| regular 0.25 December view | `/home/albert/PyStormTracker-Validation/results/track_comparison-20260818/inputs/track-compatible/era5_msl_2025-12_0.25x0.25.nc` | NetCDF-4 compatibility view | 1440 / 721 / 124      | 2025-12-01 00 to 2025-12-31 18, 6-hourly | `msl`, float Pa |

The F320 coordinates are Gaussian latitude nodes, with longitude in degrees
east and no duplicate cyclic endpoint. The regular sources have decreasing
latitude from 90 to -90 and longitude from 0 through 357.5 degrees (2.5°) or
359.75 degrees (0.25°), also without a duplicate endpoint. F320 uses
`9.96921e36` as its missing-value marker; the downloaded regular `msl` field
uses NaN metadata. The original regular files store `valid_time` as int64
seconds since 1970; TRACK 1.5.4 rejects that type. The accepted compatibility
views change only that time coordinate to int32 hours since 1900, leaving
pressure, latitude, longitude, and dimensions unchanged.

```text
F320 full source SHA256: a2843cd3277da18b1b9e4c1ac5697e5785bbe65d8879aa3b793ee66280d3b6ff
F320 January subset:     645bd205658403d77348971237dde62eeb166ba7a6b93c7a33037d1db816f099
regular 2.5 full:        19477e18e4239b9f8ea5a7b7a56c2f3790fbc661bbff1a949e59ebda1a61fc40
regular 2.5 December view: bd3efb9450f22229f44d25235ec0a3fa1eb8258e3c1feec54bb7afb35b601965
regular 0.25 full:       a1847093356472303585eb9acdbfb8c993795a2e643e80d5f7cc803d0919216d
regular 0.25 December view: 3cd68088376f9bc45e702dd2ff939af8f041d6ae70f1e3efc904fd4c2c260b00
```

The F320 source is classic NetCDF 64-bit offset with 1464 six-hourly frames.
The downloaded regular sources are NetCDF-4 with 360 six-hourly frames. Their
December subsets contain 124 frames. The compatibility-view conversion is a
TRACK 1.5.4 I/O limitation, not an external meteorological regridding step.

## Timing interpretation and limitations

Every case had an untimed correctness run before the new measured repeats.
The three measured repetitions used the same already-built executable, fixed
inputs/configuration, serial execution, and warm filesystem caches. Compilation,
input subsetting, compatibility-view creation, checksum work, and cleanup are
excluded. The spectral timings are not a pure resolution-scaling comparison:
F320 uses TRACK's fast Gaussian transform while ordinary regular grids use the
least-squares spherical-harmonic path. Tracking times after reconstruction to
T42 are the more direct comparison across F320->T42 and both regular cases.

During repeat-campaign setup, two disposable correctness attempts initially
used the full DJF regular file for a December label; they were stopped or
excluded and are not in the matrix. Explicit 124-frame December compatibility
views were installed, and all accepted December correctness and measured runs
were rerun with those inputs. One native F320-grid attempt also stopped when the
filesystem filled with old disposable spectral fields; those fields were
cleared and the complete correctness and measured repetitions were rerun.

## Discrepancies corrected in the repository

The prior regular-grid documentation described the fast path as the relevant
non-Gaussian route, omitted the required msl answer, and left the regular
streams as unverified hypotheses. Source inspection and interactive runs
established the least-squares prompt branch and the field-selection answer;
the streams, hashes, prompt map, and README now reflect those results. The
final settings and the runner are all under this repository's benchmarks/
tree, not in the external Validation repository. The external tree contains
only local source-data views, generated products, timing files, and compact
run metadata used to produce this report.
