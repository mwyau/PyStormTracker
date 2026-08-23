# MGE optimization evidence

Status: measured on 2026-08-19 in the working tree rooted at
`534429158efa99e7f62df4e34ceb7bae8e0cf2a2`, before the final optimization
commit. The benchmark host was an AMD Ryzen 9 5950X with 16 logical CPUs,
CPython 3.14.4, NumPy 2.5.2, and Numba 0.67.0. All native thread variables
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, and `DUCC0_NUM_THREADS`) were fixed at `1`.

## Scientific choice of MGE bound

The controlled shared-detection January F320-to-T42 experiment used the same
three overlapping 62-frame segments for both settings:

| `mge_max_iterations` | serial segmented link | tracks / points |
| -------------------: | --------------------: | --------------: |
|                    3 |               58.52 s |     718 / 5,145 |
|                   10 |              205.28 s |     718 / 5,145 |

The packed membership and coordinate arrays were not exactly equal between the
two settings, so the 10-iteration result is not an interchangeable faster
implementation of the 3-iteration result. The TRACK 1.5.4 driver sets
`tot_term=3` and permits the final outer round to be forward-only. The
benchmark therefore uses `mge_max_iterations=3`; the value is a source-specific
algorithmic bound, not a performance-only tuning choice. The implementation
and provenance are documented in [`docs/hodges.md`](../../docs/hodges.md), with
the primary method described by [Hodges (1999)](https://journals.ametsoc.org/abstract/journals/mwre/127/6/1520-0493_1999_127_1362_acfft_2.0.co_2.xml).

The warmed profiler was also run on the same already-computed 124-frame
detections as one unsegmented link to collect the requested control-flow
counters. MGE=3 entered all 3 configured outer iterations, with 13 forward and
7 backward sweeps, 302 accepted exchanges, and 604 `track_fail` calls. MGE=10
entered all 10 configured outer iterations, with 41 forward and 28 backward
sweeps, 855 accepted exchanges, and 1,710 `track_fail` calls. Neither run
terminated before its configured bound. These unsegmented timings were 7.28 s
and 25.66 s respectively; they are separate from the segmented comparison
above. Both produced 718 tracks and 5,145 points, but `Tracks` equality and the
packed timestamp/coordinate arrays were false.

## Candidate experiments

All comparisons below held detections, segment plans, constraints, and source
ordering constant. Array equality means exact `numpy.array_equal` equality of
the packed track arrays and aligned variables unless stated otherwise.

| candidate                                  | measurement                                                                                                                                                                          | result                                                                                                                           | decision                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Row-population cache in `_mge_iteration`   | First 62-frame segment, MGE=3: 22.8966 s to 1.0454 s; exact output digest `8a009ff9ec...fd7`                                                                                         | Removes repeated `count_nonzero` scans from the ordered pair loop                                                                | **retained**                      |
| Native feature prefilter                   | Python 0.1300–0.1326 s; warmed Numba 0.00872–0.00892 s on January detections; exact retained features and diagnostics                                                                | Preserves next-frame-first, previous-frame fallback, inclusive boundary, endpoint-average `dmax`, and source order               | **retained**                      |
| Native MGE initialization                  | Python 0.05121–0.05182 s; warmed Numba 0.00310–0.00313 s; exact workspace                                                                                                            | Preserves source-order nearest-candidate ties (`<=`, later candidate wins), displacement rejection, and paired real/phantom rows | **retained**                      |
| Cached geometry arrays                     | First 62-frame segment, MGE=3: 1.3477 s versus 1.0454 s with the row cache alone; exact output                                                                                       | Extra indexing and cached representation cost more than it saves                                                                 | rejected                          |
| Native adaptive split/filter preprocessing | Isolated directional preprocessing fell from about 0.014 s to 0.00053 s, but warmed first-segment linking was 1.0497 s versus 1.0506 s and full January was 2.8066 s versus 2.7729 s | The local preprocessing is faster, but it is too small a fraction of the ordered exchange work to justify the extra path         | rejected                          |
| Complete native directional loop           | Exact packed arrays and variables; full January improvement was about 1.2% after warmup and about 0.36% for two concurrent large segments                                            | Adaptive preprocessing still requires integration around the native loop; measured end-to-end gain is not material               | rejected                          |
| Scratch-array reuse                        | First 62-frame segment: 0.8632–0.8650 s reused versus 0.8682–0.8713 s freshly allocated                                                                                              | Difference is below useful measurement resolution for this workload                                                              | rejected                          |
| `prange` or parallel Numba pair loop       | Not attempted                                                                                                                                                                        | The pair loop has greedy source order, trial mutations, and shared best-swap state                                               | prohibited by algorithm semantics |

The retained native paths are in `pystormtracker.hodges.mge`; the Python
implementations remain available as reference methods for focused equivalence
tests. The ordinary tracker has no profiling output or per-pair logging.

## Profiler

The reproducible helper is
[`scripts/profile_mge.py`](scripts/profile_mge.py). It warms the Numba kernels,
then profiles a serial link with optional counters disabled in normal tracking:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 DUCC0_NUM_THREADS=1 \
uv run python benchmarks/track_comparison/scripts/profile_mge.py \
  --case f320_to_t42_january --frames 62 --output /tmp/profile_mge_62.json
```

The January profile used one unsegmented 62-frame segment, so its timing is a
microbenchmark of the linker rather than the segmented end-to-end benchmark.
It recorded:

```text
raw features                 2,933
retained features            2,507
workspace rows               712 (356 real rows; 356 phantom rows; 2,507 real cells; 41,637 phantom cells)
outer iterations             3
directional stages           5
forward / backward sweeps    7 / 5
ordered pair evaluations     213,166,800
accepted exchanges           105
track_fail calls             210
split calls / added rows     6 / 0
constraint-filter calls      5
link wall time               0.9896 s (optional counters enabled)
prefilter / workspace        0.0089 s / 0.0037 s
adaptive split / filter      0.0129 s / 0.0384 s
forward / backward stages    0.4913 s / 0.3817 s
final split / materialize    0.0026 s / 0.0439 s
output                       384 tracks / 2,507 points
```

## Final scaling measurements

These runs used Dask, four frame workers, one SHT thread, `segment_frames=62`,
MGE=3, and the fixed native-thread limits above. The resolved worker counts
were recorded from the run metadata. Every row within a case produced the
same TrackJSON SHA.

### F320 to T42, January, 124 frames

| requested MGE workers | resolved | total wall | frame stage | MGE segment stage | output SHA prefix |
| --------------------: | -------: | ---------: | ----------: | ----------------: | ----------------- |
|                     1 |        1 |     7.91 s |      5.04 s |            2.53 s | `190fad7d5220`    |
|                     2 |        2 |     7.01 s |      5.04 s |            1.62 s | `190fad7d5220`    |
|                     4 |        4 |     6.96 s |      5.01 s |            1.60 s | `190fad7d5220`    |
|                    16 |       16 |     6.93 s |      4.98 s |            1.61 s | `190fad7d5220`    |

Each run produced 718 tracks and 5,145 points. The serial gate was 29.29 s
end-to-end and produced the same full TrackJSON SHA as the Dask MGE=2 run:
`190fad7d5220e832fa6df7cb759240ea1377fabd29832dcda6615e5b663ae34d`.

### F320 to T42, full year, 1,464 frames

| requested MGE workers | resolved | total wall | frame stage | MGE segment stage | output SHA prefix  |
| --------------------: | -------: | ---------: | ----------: | ----------------: | ------------------ |
|                     1 |        1 |    86.81 s |     55.99 s |           28.70 s | `df1d78e00f9e3944` |
|                     2 |        2 |    72.06 s |     55.17 s |           14.81 s | `df1d78e00f9e3944` |
|                     4 |        4 |    65.52 s |     54.97 s |            8.38 s | `df1d78e00f9e3944` |
|                    16 |       16 |    63.49 s |     55.11 s |            6.25 s | `df1d78e00f9e3944` |

Each run produced 7,859 tracks and 60,878 points. The frame stage dominates at
the higher worker counts; the MGE segment stage continues to benefit through
16 workers but is no longer the end-to-end bottleneck.

## Final validation matrix

These are final-production-path measurements after the retained changes. The
isolated 62-frame value is a warmed serial linker call and includes link setup,
MGE, final splitting, and packed materialization; it does not include field
reading, SHT, or detection. The Dask values use the same fixed native-thread
limits as the scaling runs.

| validation case                           | total wall | frame stage |        MGE stage |    merge / write | resolved workers (frame / SHT / MGE) | tracks / points |
| ----------------------------------------- | ---------: | ----------: | ---------------: | ---------------: | ------------------------------------ | --------------: |
| January T42, isolated 62-frame link       |   0.9420 s |           — | included in link | included in link | serial / 1 / serial                  |     384 / 2,507 |
| F320 → T42 January, 124 frames            |   7.0086 s |    5.0374 s |         1.6227 s |         0.1346 s | 4 / 1 / 2                            |     718 / 5,145 |
| F320 → F320 January, 124 frames           | 459.0505 s |  456.6955 s |         1.9979 s |         0.1430 s | 4 / 1 / 2                            |     789 / 5,549 |
| F320 → T42 full year, 1,464 frames, MGE=2 |   72.057 s |    55.168 s |         14.809 s |           1.10 s | 4 / 1 / 2                            |  7,859 / 60,878 |

The isolated 62-frame samples were `(0.9485, 0.9420, 0.9408)` seconds wall
and `(0.9481, 0.9415, 0.9406)` seconds process CPU; the table reports the
wall/CPU medians. The F320-grid January result had TrackJSON SHA
`7beafdf17803d87c1e158ddfc797790f0e6b7bb747ada94bb11ae87b45529e26`.

## Equivalence and validation

- The actual January native prefilter/initialization linker output compared
  exactly equal to a linker forced to use the retained Python reference
  methods: `Tracks.__eq__` was true for 718 tracks and 5,145 points.
- Dask and serial January TrackJSON outputs were byte-identical.
- Focused Hodges tests passed after the counter and native-path changes:
  `uv run pytest -q tests/unit/hodges/test_mge.py tests/unit/hodges/test_linker.py`.
- The counter array is passed only by the benchmark profiler; normal tracker
  calls use the zero-length disabled path and retain the same greedy serial
  pair ordering. No inner pair parallelism was introduced.

The repository references `docs/development/track-1.5.4-source-map.md` in its
checkout instructions, but that file is not present in this checkout. The
source-specific claims above were checked against the available
[`docs/hodges.md`](../../docs/hodges.md) links and the primary Hodges paper;
this missing local map is an existing repository/documentation gap, not a
substitute for scientific validation.
