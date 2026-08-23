# F320 frame-processing optimization

Status: measured on 2026-08-19 on the `benchmarks/pst-native-reconciliation-progress`
checkout. The source input is the existing 124-frame January 2024 ERA5 MSLP
F320 file. Measurements used CPython 3.14.4, NumPy 2.5.2, SciPy 1.18.0,
ducc0 0.41.0, and Numba 0.67.0 on an AMD Ryzen 9 5950X with 16 available
logical CPUs. Native BLAS/OpenMP/DUCC limits were set to one for the PST
workflow runs.

The earlier packed-detector optimization and regression tests are recorded in
commit `d52c363`; the fixed-grid continuation is documented below.

The reusable helpers are [`profile_f320_frame.py`](scripts/profile_f320_frame.py)
and [`profile_f320_spectral.py`](scripts/profile_f320_spectral.py). They warm
imports and numerical caches before recording medians. No timing path was
added to ordinary tracker execution.

## Fixed-grid FITPACK construction continuation

Measured on 2026-08-20 on this checkout with the same January F320 input and
native numerical thread limits of one. This continuation supersedes the
earlier conclusion below that per-frame FITPACK construction remained the
dominant spline cost.

The rectangular `s=0` knot vectors are invariant across four representative
January frames: F320 has knot lengths 1285 and 644, and T42 has lengths 133 and
68\. The cached preparation now stores those knots and FITPACK's fixed banded
Givens/QR factors. Each frame only reorders/extends its values, applies the
cached rotations, and performs the two triangular solves. The direct
FITPACK path remains available as `build_bspline_surface_reference` for
validation and for nonzero smoothing; no detector, MGE, filtering, threshold,
or scheduling semantics changed.

| Warmed single-frame stage (median of 3) |  T42 (s) | F320 (s) |
| --------------------------------------- | -------: | -------: |
| direct FITPACK construction             | 0.000349 | 0.035035 |
| cached coefficient solve                | 0.000065 | 0.010907 |
| cached complete spline build            | 0.000113 | 0.014976 |
| complete detector/refinement call       | 0.003272 | 0.041917 |

The January Dask end-to-end gate used four frame workers, one SHT thread, and
two MGE workers. Cached and direct-reference runs were byte-identical:

| Case         | Tracks / points | TrackJSON SHA-256                                                  |
| ------------ | --------------: | ------------------------------------------------------------------ |
| F320 -> T42  |     718 / 5,145 | `190fad7d5220e832fa6df7cb759240ea1377fabd29832dcda6615e5b663ae34d` |
| F320 -> F320 |     789 / 5,549 | `7beafdf17803d87c1e158ddfc797790f0e6b7bb747ada94bb11ae87b45529e26` |

The preparation memory audit found 28,100 bytes for T42 and 279,236 bytes
for F320, with every retained preparation array read-only. The largest
frame-local F320 coefficient array is 6,558,720 bytes; no per-center or
per-point objects are retained by the spline preparation.

Decision: **ACCEPT** the fixed-grid FITPACK-compatible QR cache for rectangular
TRACK interpolation. **REJECT** replacing the method with a local or dense
approximation, changing the Dask/MPI execution design, or removing the direct
FITPACK reference path.

## Initial single-frame decomposition

The initial profile used one representative January frame and the Python
rectangular detector before the packed implementation. SHT timings use the
same F320 source frame and T6-42 mask as the workflow; the synthesis output is
then passed to the detector/refinement path.

| Component                         | T42/frame (s) | F320/frame (s) | F320/T42 |
| --------------------------------- | ------------: | -------------: | -------: |
| source materialization            |        0.0051 |         0.0051 |     1.0x |
| SHT analysis                      |       0.00231 |        0.00246 |     1.1x |
| spectral mask                     |      0.000064 |       0.000090 |     1.4x |
| synthesis                         |      0.000101 |        0.00273 |      27x |
| rectangular candidate detection   |        0.0348 |          3.635 |     105x |
| full rectangular spline build     |       0.00126 |         0.1245 |      99x |
| FITPACK construction              |      0.000338 |         0.0352 |     104x |
| old coefficient conversion        |      0.000867 |         0.0841 |      97x |
| GDFP refinement                   |       0.00266 |        0.00254 |     1.0x |
| complete detector/refinement call |        0.0390 |          3.707 |      95x |

The independent component medians are not expected to sum exactly to the
complete-call median. They were timed separately to avoid changing the
production call graph.

For the initial F320 spline build, the explicit FITPACK decomposition was:
F1 longitude normalization/order 0.000028 s, F2 latitude ordering 0.000023 s,
F3 frame reorder 0.00217 s, F4 periodic extension 0.000413 s, F5 FITPACK
construction 0.0352 s, F6 tck extraction 0.0000024 s, and F7 Python
coefficient conversion 0.0841 s. The final path retains the same F1--F6
semantics, reuses F1/F2 metadata, and replaces only F7 with the exact reshape
conversion shown below.

The Python detector profile for the F320 frame found 303,471 thresholded grid
points, 13 initial objects, 5 forward/backward convergence passes, four cyclic
boundary merges, nine retained objects, and 51 candidate extrema. Its
substage timings were approximately 0.068 s for threshold/initial labels,
3.199 s for label propagation, 0.234 s for object materialization (including
the repeated `np.where(labels == object_id)` scans and Python point objects),
0.022 s for boundary merging, and 0.226 s for extrema extraction/grouping.
Label propagation was the dominant Python implementation cost.

## Optimizations and exactness

The packed Numba detector keeps the reference algorithm's inclusive threshold,
four-neighbor propagation, duplicated endpoint, cyclic boundary merge, source
ordering, object-size filter, strict local-extremum comparison, adjacent
grouping, tie behavior, candidate values, and compact object IDs. It replaces
transient Python point objects and per-object full-grid rescans with packed
arrays and linked lists.

The rectangular spline path now accepts immutable coordinate-order state owned
by the detector/tracker execution. Longitude normalization/order, latitude
order, and immutable periodic endpoint metadata are prepared once per tracking
run; frame values are still reordered and copied for each frame. No mutable
module-global frame state is used.

FITPACK coefficient extraction changed from the nested Python copy loop to
`coeffs_raw.reshape(nx_knots, ny_knots).copy()`. The representative T42 and
F320 signed/unsigned checks and the SciPy `tck` reconstruction require exact
`np.array_equal` coefficient arrays. The analogous spherical extraction loop
was changed because it has the same FITPACK ordering contract.

The final warmed single-frame profile was:

| Component                            | T42/frame (s) | F320/frame (s) | F320/T42 |
| ------------------------------------ | ------------: | -------------: | -------: |
| source materialization               |        0.0052 |         0.0052 |     1.0x |
| SHT analysis                         |       0.00223 |        0.00240 |     1.1x |
| spectral mask                        |      0.000055 |       0.000087 |     1.6x |
| synthesis                            |      0.000103 |        0.00263 |      26x |
| packed rectangular detection         |      0.000233 |         0.0193 |      83x |
| cached-grid FITPACK construction     |      0.000344 |         0.0342 |      99x |
| new coefficient conversion           |     0.0000034 |       0.000384 |     113x |
| cached full rectangular spline build |      0.000393 |         0.0369 |      94x |
| GDFP refinement                      |       0.00265 |        0.00252 |     1.0x |
| complete detector/refinement call    |       0.00357 |         0.0580 |      16x |

The coefficient change reduces the F320 full spline-build measurement from
about 0.1245 s to about 0.037 s. The Numba detector reduces the isolated
candidate stage from about 3.635 s to about 0.019 s. Cached grid metadata is
a smaller but exact improvement; it removes repeated coordinate ordering and
reduces the final complete-call median from about 0.0608 s to about 0.0580 s
in the helper.

GDFP is already below five percent of the final F320 detector/refinement call,
so its mathematics and scientific parameters were not changed. Duplicate
filtering and diagnostic construction are below the timing resolution of the
F320 residual after the measured detector, spline, and GDFP stages; the final
single-frame residual was under 0.3 ms on T42 and not separable from zero on
F320 with this decomposition.

## Pure spectral benchmark

The all-124-frame spectral helper materializes the source once, then computes
checksums of every transformed frame without detection or refinement. The
effective DUCC pool was recorded in each run. The default serial DUCC request
resolved to a one-thread pool on this host, so it was retained as a separate
default row rather than interpreted as a hardware-scaling result.

| Target | Arrangement                  | Wall (s) | User CPU (s) | System CPU (s) | Max RSS (KB) | DUCC pool |
| ------ | ---------------------------- | -------: | -----------: | -------------: | -----------: | --------: |
| T42    | 1 frame worker, default DUCC |    0.303 |        0.305 |          0.000 |    1,079,764 |         1 |
| T42    | 1 frame worker, explicit 1   |    0.301 |        0.304 |          0.000 |    1,080,308 |         1 |
| T42    | 4 frame workers × 4 DUCC     |    0.071 |        0.347 |          0.025 |    1,080,388 |         4 |
| T42    | current Dask 4 × 1           |    0.285 |        0.555 |          0.011 |    1,079,860 |         1 |
| F320   | 1 frame worker, default DUCC |    0.612 |        0.615 |          0.000 |    1,079,348 |         1 |
| F320   | 1 frame worker, explicit 1   |    0.615 |        0.618 |          0.000 |    1,080,308 |         1 |
| F320   | 4 frame workers × 4 DUCC     |    0.127 |        0.706 |          0.015 |    1,078,708 |         4 |
| F320   | current Dask 4 × 1           |    0.377 |        0.909 |          0.018 |    1,080,056 |         1 |

The pure spectral F320 result is less than one second for all 124 frames. It
therefore explains far below one percent of the original approximately
456.7-second F320 frame stage. DUCC0 is not the bottleneck.

## TRACK spline structure

The checked-out TRACK source is tag `TRACK-1.5.4`, commit
`6ded301a5f5183d73e5b49c16019024b9a53eff7`. The benchmark's `RUNDATIN`
selects `tf=7`, whose `threshold.c` path calls global `surfit()` followed by
`non_lin_opt()`. `surfit.c` selects the rectangular `smoopy_c()` path, and
`smoopy_setup.c` initializes the configured frame region. The object-local
`object_smint.c` path is used by TRACK's separate `tf=8` workflow, not by this
benchmark. The PST full-frame rectangular surface is therefore retained for
the benchmark rather than replaced with a local approximation.

## Experiment scorecard

| Experiment                                |            F320 frame |               January F320→F320 |         Speedup vs prior | Exact output | Decision                                      |
| ----------------------------------------- | --------------------: | ------------------------------: | -----------------------: | ------------ | --------------------------------------------- |
| pre-optimization baseline                 |               3.707 s | 459.05 s total / 456.70 s frame |                    1.00x | reference    | baseline                                      |
| vectorized FITPACK coefficient extraction |  0.037 s spline build |         measured in combination |        3.4x spline build | yes          | retained                                      |
| packed Numba rectangular detector         | 0.058 s complete call |         measured in combination | about 54x total workflow | yes          | retained                                      |
| cached grid order/endpoint metadata       | 0.058 s complete call |         measured in combination |    small additional gain | yes          | retained                                      |
| local GDFP changes                        |         not attempted |                   not attempted |                        — | —            | rejected by scope; already small              |
| full-grid spline replacement              |         not attempted |                   not attempted |                        — | —            | rejected; TRACK structure requires validation |

All retained changes were checked with exact candidate-array equality against the
Python reference on constructed signed/unsigned frames and a real January
F320 frame. Complete real-frame `HodgesCenterFrame` coordinates, values,
diagnostics, and statuses were exactly equal. The final January runs below
also produced the accepted baseline TrackJSON hashes.

## Final January validation

The normal Dask benchmark used four frame workers, one SHT thread per frame,
two MGE workers, the existing MGE=3 setting, and the unchanged scientific
T6-42/SMOOPY settings.

| Case              | Source/graph setup (s) | Frame stage (s) |      MGE (s) | Merge/splice (s) | TrackJSON write (s) |    Total (s) | Tracks / points | SHA prefix     |
| ----------------- | ---------------------: | --------------: | -----------: | ---------------: | ------------------: | -----------: | --------------: | -------------- |
| F320→T42 January  |           0.207 median |    0.512 median | 1.583 median |     0.091 median |        0.043 median | 2.446 median |     718 / 5,145 | `190fad7d5220` |
| F320→F320 January |           0.206 median |    6.178 median | 1.997 median |     0.097 median |        0.045 median | 8.531 median |     789 / 5,549 | `7beafdf17803` |

The three native-F320 total times were 8.583, 8.377, and 8.531 s; the three
T42 totals were 2.446, 2.473, and 2.433 s. The original accepted PST values
were about 7.01 s for T42 and 459.05 s for F320, with the same complete output
hashes. The optimized result is therefore approximately 54x faster for the
native-F320 January workflow and approximately 2.9x faster for T42.

An independent `/usr/bin/time` sample of the same normal-Dask command recorded
process-level user/system/RSS values of 7.63/0.52 s and 320,616 KB for T42,
and 14.62/1.45 s and 499,612 KB for F320. Those process-level values include
Python startup and cache/compilation effects; the table above is the more
stable workflow-boundary median.

The measured original F320 frame-stage budget was approximately 456.7 s. The
component profile attributes it primarily to the Python detector (about 98%
of the old single-frame detector call, dominated by propagation), with full
spline construction contributing about 3% and GDFP less than 0.1%; the small
overlap in separately timed medians means these percentages are approximate.
Before the fixed-grid continuation, FITPACK construction was the remaining
dominant single-frame operation, with GDFP a distant second and SHT negligible.

## Worker recommendation and next benchmark

The final small F320 January matrix was:

| Frame workers | SHT threads | Frame stage (s) |     Total (s) |        Max RSS (KB) |
| ------------: | ----------: | --------------: | ------------: | ------------------: |
|             1 |          16 |           8.998 |        11.375 |             308,196 |
|             2 |           8 |           6.691 |         9.046 |             358,088 |
|             4 |           4 |           6.025 |         8.358 |             484,896 |
|             8 |           2 |          14.251 |        16.676 |             742,728 |
|            16 |           1 |   26.941–48.713 | 29.320–51.107 | 1,182,100–1,195,924 |

Four frame workers × four SHT threads was the fastest measured arrangement on
this host. It is a benchmark recommendation only; public defaults remain
unchanged. The deterioration at eight and sixteen concurrent full-grid
FITPACK builds is consistent with memory-bandwidth and working-set pressure.

The January gate is now satisfied and a full-year F320→F320 benchmark is
justified. It was not run as part of this focused optimization pass; the
remaining cost is source-compatible full-grid frame work, with the rectangular
spline system now cached.
