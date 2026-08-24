# TRACK 1.5.4 comparison baseline

This directory is the reproducible TRACK reference/performance harness used to
compare TRACK 1.5.4 with PyStormTracker. It intentionally keeps the small,
durable inputs needed to reproduce TRACK runs in the software repository while
leaving meteorological data and generated outputs outside Git.

The final serial validation matrix is in [`RESULTS.md`](RESULTS.md), and the
line-by-line answer semantics are in [`INPUT_SEMANTICS.md`](INPUT_SEMANTICS.md).

TRACK is pinned to:

```text
tag:    TRACK-1.5.4
commit: 6ded301a5f5183d73e5b49c16019024b9a53eff7
```

The checked-in 2024 F320 configuration is the supplied reference configuration.
The 2025-2026 regular-latitude/longitude cases are an additional benchmark
extension and are labelled separately below.

## What is checked in

```text
benchmarks/track_comparison/
├── README.md
├── INPUT_SEMANTICS.md
├── REFERENCE_SETTINGS.md
├── RESULTS.md
├── .gitignore
├── configs/
│   ├── f320/
│   │   ├── adapt.dat0
│   │   ├── zone.dat0
│   │   ├── initial.f320_to_t42
│   │   ├── initial.f320_to_f320
│   │   ├── RUNDATIN.era5_MSLP_latlng.in
│   │   ├── RUNDATIN.era5_MSLP_latlng_A.in
│   │   ├── specfilt_f320_to_t42
│   │   ├── specfilt_f320_to_f320
│   │   └── SHA256SUMS
│   └── regular_latlon/
│       ├── README.md
│       ├── PROMPT_MAP.md
│       ├── specfilt_regular_2p5_T42
│       └── specfilt_regular_0p25_T42
├── patches/
│   └── track-1.5.4-build-modern-deps.patch
└── scripts/
    ├── preflight_input.py
    └── run_track_repeat.sh
```

Do not check in raw ERA5 data, TRACK binary intermediate fields, trajectory
outputs, timing logs, compiled TRACK executables, or disposable TRACK trees.

## Benchmark matrix

### Supplied 2024 F320 reference matrix

All four cases start from the same raw 6-hourly ERA5 2024 MSLP field on the
full F320 Gaussian grid: 1280 distinct longitudes by 640 Gaussian latitudes.

The scientific spectral operation is the same T6-42 preparation. The two
spatial cases differ in the grid on which the filtered field is synthesized.

| Case                   | Raw input | Filtered output/tracking grid                        | Period   | Frames | TRACK segmentation |
| ---------------------- | --------- | ---------------------------------------------------- | -------- | -----: | ------------------ |
| F320 → T42, January    | F320      | T42 Gaussian, 128×64 plus cyclic endpoint internally | Jan 2024 |    124 | `-n=1,62,2`        |
| F320 → T42, full year  | F320      | T42 Gaussian, 128×64 plus cyclic endpoint internally | 2024     |   1464 | `-n=1,62,24`       |
| F320 → F320, January   | F320      | T6-42 field synthesized on the F320 Gaussian grid    | Jan 2024 |    124 | `-n=1,62,2`        |
| F320 → F320, full year | F320      | T6-42 field synthesized on the F320 Gaussian grid    | 2024     |   1464 | `-n=1,62,24`       |

`initial.f320_to_t42` contains 129 longitude positions because TRACK's
rectangular representation includes the periodic endpoint.
`initial.f320_to_f320` analogously contains 1281 positions for the 1280
distinct F320 longitudes.

### Regular-lat/lon extension

The ERA5 data catalog also defines MSLP on ordinary global regular grids for
December 2025 through February 2026.

| Case            | Grid  | Period            | Frames | Expected dimensions | Expected segments |
| --------------- | ----- | ----------------- | -----: | ------------------- | ----------------: |
| coarse month    | 2.5°  | Dec 2025          |    124 | 144×73              |                 2 |
| coarse season   | 2.5°  | Dec 2025–Feb 2026 |    360 | 144×73              |                 6 |
| high-res month  | 0.25° | Dec 2025          |    124 | 1440×721            |                 2 |
| high-res season | 0.25° | Dec 2025–Feb 2026 |    360 | 1440×721            |                 6 |

The intended comparison target for these extension cases is T42 Gaussian
tracking output so that the TRACK detector/linker configuration is held fixed
while the source-grid resolution changes.

These four regular-grid cases are **not part of the supplied 2024 reference
packet**. TRACK's global least-squares path in `spectral_filter.c` is used for
the ordinary regular latitude/longitude inputs; the fast transform is reserved
for the Gaussian F320 input. The exact checked-in streams and every answer's
meaning are documented in [`INPUT_SEMANTICS.md`](INPUT_SEMANTICS.md).

The downloaded regular files use int64 `valid_time`, which TRACK 1.5.4 rejects
in its generic NetCDF reader. Runs therefore use a compatibility view that
changes only the time coordinate to int32 hours since 1900. This source
limitation and the view provenance are part of the benchmark record.

## 1. Dependencies

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    gcc \
    gfortran \
    make \
    xutils-dev \
    patch \
    git \
    csh \
    time \
    libhdf5-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    netcdf-bin \
    nco
```

`xutils-dev` supplies `makedepend`. `nco` is used to make true shorter input
files rather than truncating trajectories after a longer TRACK run.

Verify NetCDF-4 support:

```bash
nc-config --version
nf-config --version
nc-config --prefix
nc-config --has-nc4
test "$(nc-config --has-nc4)" = yes
```

## 2. Clone and pin TRACK

```bash
mkdir -p ~/track-comparison
cd ~/track-comparison

git clone https://gitlab.act.reading.ac.uk/track/track.git TRACK
cd TRACK
git fetch --tags
git checkout --detach TRACK-1.5.4

test "$(git rev-parse HEAD)" = \
    6ded301a5f5183d73e5b49c16019024b9a53eff7
```

Keep this checkout pristine. Use a disposable worktree for actual builds/runs:

```bash
git worktree add --detach ../TRACK-run \
    6ded301a5f5183d73e5b49c16019024b9a53eff7
cd ../TRACK-run
```

## 3. Apply the build-only patch

Set the benchmark directory from a PyStormTracker checkout:

```bash
PST=/path/to/PyStormTracker
BASELINE="$PST/benchmarks/track_comparison"
```

Then:

```bash
git apply --check "$BASELINE/patches/track-1.5.4-build-modern-deps.patch"
git apply "$BASELINE/patches/track-1.5.4-build-modern-deps.patch"
git diff --check
```

The patch gates stale generated dependency lists behind
`TRACK_LEGACY_DEPS=1`; it does not change tracking science.

## 4. Build TRACK with NetCDF-4

```bash
export PATH="$PWD:$PWD/bin:$PATH"
export CC=gcc
export FC=gfortran
export ARFLAGS=
export NETCDF="$(nc-config --prefix)"
export TRACK_LEGACY_DEPS=0

test "$(nc-config --has-nc4)" = yes

./master -build -i=linux -f=linux

test -x bin/track.linux
file bin/track.linux
ldd bin/track.linux | grep -Ei 'netcdf|hdf5' || true
```

Do not substitute the old `NONETCDF` validation build. The raw ERA5 workflow
requires a NetCDF-enabled executable.

Record the build:

```bash
mkdir -p benchmark-results
{
    date -Is
    uname -a
    git rev-parse HEAD
    gcc --version | head -1
    gfortran --version | head -1
    make --version | head -1
    printf 'NETCDF=%s\n' "$NETCDF"
    nc-config --version
    nc-config --has-nc4
    nc-config --cflags
    nc-config --libs
    nf-config --version
    nf-config --fflags
    nf-config --flibs
    sha256sum bin/track.linux
} | tee benchmark-results/TRACK_BUILD_PROVENANCE.txt
```

## 5. Install the F320 reference configuration

Verify the checked-in files first:

```bash
(
    cd "$BASELINE/configs/f320"
    sha256sum -c SHA256SUMS
)
```

Copy them into the disposable TRACK tree:

```bash
cp "$BASELINE/configs/f320/adapt.dat0" data/
cp "$BASELINE/configs/f320/zone.dat0" data/
cp "$BASELINE/configs/f320/initial.f320_to_t42" data/
cp "$BASELINE/configs/f320/initial.f320_to_f320" data/

cp "$BASELINE/configs/f320/RUNDATIN.era5_MSLP_latlng.in" indat/
cp "$BASELINE/configs/f320/RUNDATIN.era5_MSLP_latlng_A.in" indat/

cp "$BASELINE/configs/f320/specfilt_f320_to_t42" .
cp "$BASELINE/configs/f320/specfilt_f320_to_f320" .
```

The two `RUNDATIN` files contain:

```text
PATH/data/%INITIAL%
```

Edit only the copied runtime versions. Preserve `#` and `!` exactly:

```bash
TRACK_ROOT=$(pwd -P)
sed -i "s|PATH/data/%INITIAL%|${TRACK_ROOT}/data/%INITIAL%|g" \
    indat/RUNDATIN.era5_MSLP_latlng.in \
    indat/RUNDATIN.era5_MSLP_latlng_A.in

! grep -n 'PATH/data/%INITIAL%' indat/RUNDATIN.era5_MSLP_latlng*.in
```

## 6. Verify the raw F320 file

Assume the data file is outside TRACK, for example in a sibling data repo:

```bash
ERA5=/path/to/ERA5_mslp_6hr_2024_DET.nc
test -f "$ERA5"

ncdump -k "$ERA5"
ncdump -h "$ERA5" | sed -n '1,120p'
```

If running from a PyStormTracker checkout with `xarray` installed:

```bash
python "$BASELINE/scripts/preflight_input.py" f320-2024 "$ERA5"
```

For January, make an actual 124-frame file:

```bash
ncks -O -d time,0,123 "$ERA5" \
    indat/ERA5_mslp_6hr_2024-01_DET.nc

python "$BASELINE/scripts/preflight_input.py" \
    f320-jan2024 indat/ERA5_mslp_6hr_2024-01_DET.nc
```

Do not time the `ncks` subset operation unless data preparation itself is the
quantity under study.

## 7. Timing format

Use GNU `/usr/bin/time`, not a shell built-in:

```bash
/usr/bin/time \
    -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o stage.time \
    command ... \
    > stage.log 2>&1
```

Keep the `.time` and `.log` files. The first is the benchmark measurement; the
second is TRACK console output.

Record spectral preparation and tracking separately. An end-to-end wall time is
their sum when no other stage is included.

## 8. F320 → T42: full 2024

Link the raw file:

```bash
ln -sfn "$ERA5" indat/ERA5_mslp_6hr_2024_DET.nc
mkdir -p benchmark-results/f320_to_t42_full_year benchmark-output/f320_to_t42_full_year

RESULTS=$(realpath benchmark-results/f320_to_t42_full_year)
OUT=$(realpath benchmark-output/f320_to_t42_full_year)
```

Filter and time:

```bash
/usr/bin/time \
    -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULTS/spectral-filter.time" \
    bin/track.linux \
    -i ERA5_mslp_6hr_2024_DET.nc \
    -f filtT42 \
    < specfilt_f320_to_t42 \
    > "$RESULTS/spectral-filter.log" 2>&1

mv outdat/specfil.filtT42_band001 \
    indat/ERA5_mslp_6hr_2024_DET_T42filt.dat
rm -f outdat/specfil.filtT42_band000
```

Track and time:

```bash
/usr/bin/time \
    -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULTS/tracking.time" \
    ./master \
    -c=ERA5_mslp_6hr_2024_DET_latlng \
    -e=track.linux \
    -d=now \
    -i=ERA5_mslp_6hr_2024_DET_T42filt.dat \
    -f=f2024 \
    -j=RUN_AT.in \
    -k=initial.f320_to_t42 \
    -n=1,62,24 \
    -o="$OUT" \
    -r=RUN_AT_ \
    -s=RUNDATIN.era5_MSLP_latlng \
    > "$RESULTS/tracking.log" 2>&1
```

## 9. F320 → F320: full 2024

```bash
mkdir -p benchmark-results/f320_to_f320_full_year benchmark-output/f320_to_f320_full_year
RESULTS=$(realpath benchmark-results/f320_to_f320_full_year)
OUT=$(realpath benchmark-output/f320_to_f320_full_year)

/usr/bin/time \
    -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULTS/spectral-filter.time" \
    bin/track.linux \
    -i ERA5_mslp_6hr_2024_DET.nc \
    -f filtT42 \
    < specfilt_f320_to_f320 \
    > "$RESULTS/spectral-filter.log" 2>&1

mv outdat/specfil.filtT42_band001 \
    indat/ERA5_mslp_6hr_2024_DET_T42filt_full.dat
rm -f outdat/specfil.filtT42_band000

/usr/bin/time \
    -f 'wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$RESULTS/tracking.time" \
    ./master \
    -c=ERA5_mslp_6hr_2024_DET_latlng_full \
    -e=track.linux \
    -d=now \
    -i=ERA5_mslp_6hr_2024_DET_T42filt_full.dat \
    -f=f2024 \
    -j=RUN_AT.in \
    -k=initial.f320_to_f320 \
    -n=1,62,24 \
    -o="$OUT" \
    -r=RUN_AT_ \
    -s=RUNDATIN.era5_MSLP_latlng \
    > "$RESULTS/tracking.log" 2>&1
```

## 10. January 2024 variants

Use the 124-frame January file and the same two scientific configurations.

### F320 → T42, January

Filtering:

```bash
bin/track.linux \
    -i ERA5_mslp_6hr_2024-01_DET.nc \
    -f filtT42 < specfilt_f320_to_t42

mv outdat/specfil.filtT42_band001 \
    indat/ERA5_mslp_6hr_2024-01_DET_T42filt.dat
rm -f outdat/specfil.filtT42_band000
```

Tracking arguments change to:

```text
-c=ERA5_mslp_6hr_2024-01_DET_latlng
-i=ERA5_mslp_6hr_2024-01_DET_T42filt.dat
-f=f2024jan
-k=initial.f320_to_t42
-n=1,62,2
```

Wrap both stages with the same `/usr/bin/time` format used above.

### F320 → F320, January

Filtering:

```bash
bin/track.linux \
    -i ERA5_mslp_6hr_2024-01_DET.nc \
    -f filtT42 < specfilt_f320_to_f320

mv outdat/specfil.filtT42_band001 \
    indat/ERA5_mslp_6hr_2024-01_DET_T42filt_full.dat
rm -f outdat/specfil.filtT42_band000
```

Tracking arguments change to:

```text
-c=ERA5_mslp_6hr_2024-01_DET_latlng_full
-i=ERA5_mslp_6hr_2024-01_DET_T42filt_full.dat
-f=f2024jan
-k=initial.f320_to_f320
-n=1,62,2
```

Again, retain separate spectral-filter and tracking timing files.

## 11. Expected TRACK outputs

A successful MSLP workflow should produce products corresponding to:

```text
tr_trs_neg
tr_trs_pos
ff_trs_neg
ff_trs_pos
```

The wrapper may compress them, e.g. `.gz`.

For cyclone/minimum comparison:

```text
tr_trs_neg   raw tracked minima
ff_trs_neg   post-filtered minima
```

Do not compare `ff_trs_neg` with a PyStormTracker pre-activity-filter output.

## 12. Regular-grid data files

The PyStormTracker-Data catalog defines:

```text
era5_msl_2025-2026_djf_2.5x2.5.nc
era5_msl_2025-2026_djf_0.25x0.25.nc
```

both at 00/06/12/18 UTC from 2025-12-01 through 2026-02-28.

Verify them before TRACK sees them:

```bash
python "$BASELINE/scripts/preflight_input.py" \
    2p5-djf2025-2026 \
    ../PyStormTracker-Data/release-data/era5_msl_2025-2026_djf_2.5x2.5.nc

python "$BASELINE/scripts/preflight_input.py" \
    0p25-djf2025-2026 \
    ../PyStormTracker-Data/release-data/era5_msl_2025-2026_djf_0.25x0.25.nc
```

The actual release asset location may differ; use the catalog filename rather
than assuming the example directory.

Make December-only files from the DJF files with `ncks`:

```bash
ncks -O -d time,0,123 \
    era5_msl_2025-2026_djf_2.5x2.5.nc \
    era5_msl_2025-12_2.5x2.5.nc

ncks -O -d time,0,123 \
    era5_msl_2025-2026_djf_0.25x0.25.nc \
    era5_msl_2025-12_0.25x0.25.nc
```

Then:

```bash
python "$BASELINE/scripts/preflight_input.py" \
    2p5-dec2025 era5_msl_2025-12_2.5x2.5.nc

python "$BASELINE/scripts/preflight_input.py" \
    0p25-dec2025 era5_msl_2025-12_0.25x0.25.nc
```

## 13. Regular-grid verification

Do not treat the supplied F320 `specfilt` answer streams as generic regular-grid
files. The F320 input is already Gaussian; 2.5° and 0.25° ERA5 files are
ordinary equally spaced latitude/longitude grids.

For each regular resolution, the validated workflow is:

1. pin TRACK to the SHA at the top of this document;
1. copy/symlink the NetCDF file into `indat/`;
1. make/use the track-compatible view of the selected NetCDF file;
1. run `bin/track.linux -i <filename> -f <case>` with the checked-in stream;
1. select the least-squares global path (already encoded as answer `0`);
1. verify the resulting `band001` file geometry is the T42 Gaussian
   representation expected by `initial.T42`;
1. replay the same checked-in stream and compare both output bands byte-for-byte;
1. run the normal overlapping TRACK orchestration using `initial.T42`.

Target scientific settings for both regular resolutions are:

```text
input:
    global regular latitude/longitude
    2.5°  -> 144 x 73
    0.25° -> 1440 x 721

spectral target:
    T42 Gaussian tracking grid
    same band/truncation/taper semantics as the F320 -> T42 reference case

tracking:
    initial.T42
    same RUNDATIN/adapt/zone configuration
    Dec 2025: 124 frames -> -n=1,62,2
    DJF 2025-2026: 360 frames -> -n=1,62,6
```

The interactive streams have now been verified against the pinned TRACK
1.5.4 source and replayed on the validated compatibility views. The streams
are the final settings in this repository; raw files and generated outputs
remain outside Git.

## 14. Repeated timing policy

The final matrix in `RESULTS.md` contains three measured repetitions for every
case, plus a preceding untimed correctness run. The checked-in
`scripts/run_track_repeat.sh` runner uses case-specific output directories and
records separate spectral and tracking GNU-time files, total workflow elapsed
time, logs, and product counts.

For every reportable campaign:

1. complete one untimed correctness run;
1. run at least three measured repeats;
1. use the same compiled TRACK executable;
1. do not rebuild between repeats;
1. keep input, configuration, and output-grid choice fixed;
1. record every run, not only the fastest;
1. report the median wall time;
1. describe warm/cold filesystem-cache policy and apply it equally to TRACK and
   PyStormTracker.

Record at least:

```text
spectral_filter_wall_seconds
spectral_filter_user_seconds
spectral_filter_system_seconds
spectral_filter_max_rss_kb

tracking_wall_seconds
tracking_user_seconds
tracking_system_seconds
tracking_max_rss_kb

total_stage_sum_seconds = spectral_filter_wall_seconds + tracking_wall_seconds
workflow_elapsed_seconds = measured around both stages, including handoff
```

Compilation is setup cost and is not part of scientific runtime.

## 15. Cleanup between runs

Delete only generated products for the case being rerun:

```bash
rm -f outdat/specfil.*_band000 outdat/specfil.*_band001
rm -f indat/*_T42filt.dat indat/*_T42filt_full.dat
rm -rf benchmark-output/<case>/*
```

Do not delete the source NetCDF, installed reference configuration, or build
provenance record.

## 16. Acceptance checklist

Before using a timing in a PyStormTracker comparison, verify:

```text
[ ] TRACK SHA = 6ded301a5f5183d73e5b49c16019024b9a53eff7
[ ] build-only modern-dependency patch is the only TRACK source modification
[ ] NetCDF-4 support is enabled
[ ] exact input file, dimensions, timestamps, cadence, units recorded
[ ] F320 config hashes pass SHA256SUMS
[ ] output-grid choice is stated: T42 or F320
[ ] shorter cases use actual shorter input files
[ ] spectral preprocessing and tracking timed separately
[ ] .time and .log files preserved for every measured run
[ ] raw trajectory output distinguished from post-filtered output
[ ] generated outputs are not committed to this repository
    [ ] regular-grid int64-time source limitation and compatibility view are recorded
    [ ] regular-grid streams pass interactive validation and exact replay
```
