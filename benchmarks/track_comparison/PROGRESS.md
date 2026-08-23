# PST benchmark progress

Status as of 2026-08-19: the corrected PyStormTracker campaign was stopped at
the user's request after the native F320-grid full-year Dask run exceeded the
expected runtime. The interrupted run produced no track output and is excluded
from all summaries.

The earlier campaign used mismatched settings and is quarantined outside Git at
`/home/albert/PyStormTracker-Validation/results/pst_track_comparison-20260818-unreconciled`.
Its counts are not parity evidence.

## Corrected workflow

PST reads the raw NetCDF source and constructs the T6-42 field in memory with
the public `SHTFilter`; it does not read TRACK's binary filtered-field product.
The exact preprocessing settings are `lmin=6`, `lmax=42`, `taper_val=0.1`, and
`geometry="auto"`. T42 cases synthesize to `GL`, 64 latitudes by 128
longitudes; native F320-grid cases keep the 640 by 1280 Gaussian grid.

The tracking settings are:

```text
variable = "msl"
detection_mode = "min"
object_threshold = -100 Pa
feature_refinement = "bspline"  # rectangular SMOOPY path in the current API
track_smoopy_optimization_scale = 0.01
mge_max_iterations = 3
min_object_grid_points = 3
min_track_points = 1
w1 = 0.2, w2 = 0.8
dmax = 6.5 degrees with the documented three latitude zones
adaptive smoothness = distances [1, 2, 5, 8], phimax [1.0, 0.3, 0.1, 0.0]
segment_frames = 62
time_step = 6 hours
missing frames = disabled
taper_points = 0
```

The ordinary public defaults remain unchanged (`scale=1.0`, `MGE=3`, and
`min_track_points=3`). The historical values above are explicit benchmark
parameters, not hidden attribute mutations.

The earlier corrected campaign forced `mge_max_iterations=10`. A controlled
shared-detection experiment on the January T42 case measured 58.52 s of serial
linking at 3 iterations and 205.28 s at 10. Both settings produced 718 tracks
and 5,145 points, but their packed track membership and coordinate arrays were
not exactly equal. TRACK 1.5.4 sets `tot_term=3`, so the benchmark now uses 3;
the 10-iteration run is retained as an optimization comparison, not as the
source-compatible setting.

## Completed corrected measurements

Counts are raw negative/minimum tracks and points. TRACK reference counts are
the corresponding raw `tr_trs_neg` products.

| Case                   | Backend/repeats  | PST tracks/points | TRACK tracks/points | Status                |
| ---------------------- | ---------------- | ----------------: | ------------------: | --------------------- |
| F320 → T42, January    | Dask, 3          |       718 / 5,145 |         709 / 5,127 | complete              |
| F320 → T42, full year  | Dask, 2          |    7,859 / 60,878 |      7,761 / 60,654 | complete              |
| F320 → F320, January   | Dask, 2          |       789 / 5,549 |         779 / 5,529 | complete              |
| F320 → T42, January    | serial, gate run |       718 / 5,145 |         709 / 5,127 | backend gate complete |
| F320 → F320, full year | Dask, run 2      |                 — |      8,595 / 65,539 | interrupted; excluded |

The full-year T42 result reproduces the archived old PST track count of 7,859;
its five-point difference from the archived 60,883-point result is recorded as
an observable consequence of using PST's native in-memory filter instead of the
TRACK-generated field.

The Dask benchmark uses `frame_workers=None`, `mge_workers=None`, and
`sht_threads=None`. On this host the frame and MGE defaults each resolve to 16
workers, while the Dask SHT default resolves to one DUCC0 thread per active
transform. Native numerical thread limits were set to one. The serial gate
produced byte-identical TrackJSON to the Dask January result.

## Remaining campaign work

The full 8-case, 3-repeat Dask and serial timing matrices were not completed,
so no PST-vs-TRACK CSV/JSON summary is claimed or checked in. The reproducible
runner and GNU timing wrapper are checked in for continuation:

```text
benchmarks/track_comparison/scripts/run_pst_repeat.py
benchmarks/track_comparison/scripts/run_pst_benchmark.sh
benchmarks/track_comparison/scripts/summarize_pst_results.py
```

Separately, the Dask SHT output-grid path had a real `xarray.apply_ufunc`
output-size bug. It was fixed by declaring the lazy output sizes and covered by
`tests/unit/preprocessing/test_spectral.py`; the focused spectral tests pass.
