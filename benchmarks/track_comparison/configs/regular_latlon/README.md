# Regular latitude/longitude TRACK extension

This directory contains the source-derived TRACK configuration for the four
regular latitude/longitude MSLP benchmark cases.

The regular-grid cases are extensions to the supplied 2024 F320 reference
workflow. They use the same T6-42 scientific filter, 0.1 Hoskins coefficient
taper, T42 Gaussian tracking grid, MSLP threshold/tracking settings, regional
`dmax`, adaptive smoothness, and RSPLICE configuration.

Expected inputs:

| Resolution | Period                      | Dimensions | Frames | Tracking segmentation |
| ---------- | --------------------------- | ---------: | -----: | --------------------- |
| 2.5°       | December 2025               |   144 x 73 |    124 | `-n=1,62,2`           |
| 2.5°       | December 2025-February 2026 |   144 x 73 |    360 | `-n=1,62,6`           |
| 0.25°      | December 2025               | 1440 x 721 |    124 | `-n=1,62,2`           |
| 0.25°      | December 2025-February 2026 | 1440 x 721 |    360 | `-n=1,62,6`           |

## Important source finding

Do **not** use the F320 `specfilt_f320_to_t42` answer stream directly on an
ordinary regular latitude/longitude field.

TRACK's spatial-spectral menu has two relevant global methods:

```text
option 4: spatial spectral filtering
    0 -> least-squares spherical-harmonic decomposition
    1 -> fast spectral transform
```

The F320 reference configuration selects method `1`. The fast implementation
checks that the input is a compatible Gaussian grid and exits otherwise.

For an ordinary global regular latitude/longitude grid, use method `0`.
`src/spectral_filter.c` constructs the spherical basis on the input grid,
solves the spherical-harmonic coefficients by least squares, and can reconstruct
the filtered result on a newly generated Gaussian grid.

Therefore the regular-grid benchmark is:

```text
regular global lat/lon NetCDF
    -> TRACK least-squares spherical-harmonic decomposition
    -> T42 analysis
    -> bands 0..5 and 6..42
    -> Hoskins taper, cutoff value 0.1
    -> T42 quadratic Gaussian reconstruction, no cyclic point in the file
    -> retain band001 (T6-42)
    -> initial.T42
    -> same latitude/longitude RUNDATIN pair, zone.dat0, adapt.dat0
    -> normal MGE + RSPLICE workflow
```

No external regular-to-Gaussian regridding is part of these cases.

## Files

```text
adapt.dat0
zone.dat0
initial.T42
RUNDATIN.era5_MSLP_latlng.in
RUNDATIN.era5_MSLP_latlng_A.in
specfilt_regular_2p5_T42
specfilt_regular_0p25_T42
SHA256SUMS
```

The tracking files are duplicated here so this directory is a self-contained
regular-grid benchmark packet. The spectral streams are resolution-specific:
they include the source dimensions and the explicit `msl` field selection.
Tracking starts after spectral reconstruction on the same T42 Gaussian grid.

The two `specfilt_regular_*` files differ only in the input search-area
dimensions:

```text
2.5°  : X 1..144,  Y 1..73
0.25° : X 1..1440, Y 1..721
```

See [`PROMPT_MAP.md`](PROMPT_MAP.md) and the repository-level
[`INPUT_SEMANTICS.md`](../../INPUT_SEMANTICS.md) for the source-derived,
line-by-line interpretation.

## Verify input before running

The saved NetCDF initialization answers are:

```text
print NetCDF summary: no
identify fields by variable names/dimension values: yes
COARDS organization: yes
select the `msl` field explicitly
retain both poles and the equator
do not make the source grid periodic before spectral filtering
geodesic distance
Plate Carree / no alternate projection
```

The regular files must therefore satisfy the same NetCDF assumptions. Verify
the actual files before use:

```bash
ncdump -k "$REGULAR_FILE"
ncdump -h "$REGULAR_FILE" | sed -n '1,160p'
```

Also use the checked-in preflight helper from the benchmark root.

The downloaded regular files contain int64 `valid_time`. TRACK 1.5.4 rejects
NetCDF int64 variables in its generic reader, so the benchmark uses a
track-compatible view with only the time coordinate represented as int32 hours
since 1900. The `msl`, latitude, longitude, and data values are unchanged.
The original-file rejection and compatibility-view provenance must be recorded
with benchmark results; the view is not a scientific regridding step.

## Install the regular-grid packet

From a disposable patched TRACK worktree:

```bash
BASELINE=/path/to/PyStormTracker/benchmarks/track_comparison
CFG="$BASELINE/configs/regular_latlon"

(
    cd "$CFG"
    sha256sum -c SHA256SUMS
)

cp "$CFG/adapt.dat0" data/
cp "$CFG/zone.dat0" data/
cp "$CFG/initial.T42" data/

cp "$CFG/RUNDATIN.era5_MSLP_latlng.in" indat/
cp "$CFG/RUNDATIN.era5_MSLP_latlng_A.in" indat/

cp "$CFG/specfilt_regular_2p5_T42" .
cp "$CFG/specfilt_regular_0p25_T42" .

TRACK_ROOT=$(pwd -P)
sed -i "s|PATH/data/%INITIAL%|${TRACK_ROOT}/data/%INITIAL%|g" \
    indat/RUNDATIN.era5_MSLP_latlng.in \
    indat/RUNDATIN.era5_MSLP_latlng_A.in
```

Preserve `%INITIAL%`, `#`, and `!` except for the documented `PATH`
substitution.

## 2.5-degree spectral preparation

Assume the selected input has 144 longitudes and 73 latitudes:

```bash
INPUT_2P5=/absolute/path/to/era5_msl_2025-2026_djf_2.5x2.5.nc
ln -sfn "$INPUT_2P5" indat/era5_msl_2025-2026_djf_2.5x2.5.nc
```

Run from the TRACK root:

```bash
bin/track.linux \
    -i era5_msl_2025-2026_djf_2.5x2.5.nc \
    -f regular2p5T42 \
    < specfilt_regular_2p5_T42
```

Expected spectral products:

```text
outdat/specfil.regular2p5T42_band000
outdat/specfil.regular2p5T42_band001
```

Retain T6-42:

```bash
mv outdat/specfil.regular2p5T42_band001 \
    indat/era5_msl_2025-2026_djf_2.5x2.5_T42filt.dat
rm -f outdat/specfil.regular2p5T42_band000
```

The resulting standard-binary field should be 128 x 64 physical T42 Gaussian
points, compatible with `initial.T42` (which uses the 129-position cyclic
endpoint representation internally).

## 0.25-degree spectral preparation

```bash
INPUT_0P25=/absolute/path/to/era5_msl_2025-2026_djf_0.25x0.25.nc
ln -sfn "$INPUT_0P25" indat/era5_msl_2025-2026_djf_0.25x0.25.nc

bin/track.linux \
    -i era5_msl_2025-2026_djf_0.25x0.25.nc \
    -f regular0p25T42 \
    < specfilt_regular_0p25_T42

mv outdat/specfil.regular0p25T42_band001 \
    indat/era5_msl_2025-2026_djf_0.25x0.25_T42filt.dat
rm -f outdat/specfil.regular0p25T42_band000
```

Again, verify the output is the T42 Gaussian representation before tracking.

## Tracking

After spectral preparation, both resolutions use the same tracking files:

```text
-k=initial.T42
-s=RUNDATIN.era5_MSLP_latlng
```

Use:

```text
December 2025, 124 frames       -> -n=1,62,2
DJF 2025-2026, 360 frames       -> -n=1,62,6
```

Make a true 124-frame December input before spectral preparation when measuring
the December case. Do not spectrally process all 360 frames and then truncate
tracks.

## Timing interpretation

These regular-grid cases use TRACK's least-squares transform while the supplied
F320 cases use TRACK's fast Gaussian transform. Record the spectral stage and
tracking stage separately. The least-squares matrix construction is part of
the TRACK spectral stage and may dominate the 0.25° timings.

That distinction is part of the benchmark result: a direct wall-time comparison
of the spectral stages measures both source resolution and transform algorithm.
Tracking is more directly comparable because all regular-grid cases enter the
tracker on the same T42 Gaussian representation.

## Verification status

The answer streams in this directory are **source-derived and runtime-checked**,
not guessed by editing F320 dimensions alone.

The source derivation establishes the correct branch and prompt order:

- global spatial spectral menu option `4`;
- least-squares transform `0`;
- full decomposition `0`;
- analysis truncation `42`;
- new generated Gaussian output grid;
- output truncation `42`;
- omit the output longitude wraparound;
- bands `0..5` and `6..42`;
- retain both output bands;
- Hoskins taper `0.1` on both bands;
- no field-value restriction.

The 2.5° and 0.25° streams were run through EOF with the pinned patched
TRACK-1.5.4 executable. Each produced two 128 x 64 T42 Gaussian standard
binary bands for the validated December subset; the direct replay is recorded
with the benchmark results. The extra trailing `n` answers in each stream are
not consumed by the completed spectral path and are retained for transcript
provenance.
