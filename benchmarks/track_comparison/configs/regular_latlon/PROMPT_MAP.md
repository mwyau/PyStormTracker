# Regular-grid spectral prompt map

The authoritative line-by-line decoding is
[`INPUT_SEMANTICS.md`](../../INPUT_SEMANTICS.md). This short note records the
source decisions specific to the regular-grid streams.

## Source branch

TRACK option 4 is spatial spectral filtering. TRACK 1.5.4 dispatches:

```text
0  least-squares spherical-harmonic decomposition
1  fast spectral transform
2  limited-area DCT
```

The regular streams select `0`. `src/spectral_filter.c` evaluates a spherical
basis on the supplied regular longitude/latitude coordinates, forms the least-
squares normal equations, and reconstructs a new quadratic Gaussian T42 grid.
The fast path is not used for these ordinary regular-latitude/longitude files.

## Exact source-file prefix

The first 19 answers are input initialization, not spectral-band settings:

|  Lines | Answers                   | Meaning                                                                                   |
| -----: | ------------------------- | ----------------------------------------------------------------------------------------- |
|   1--2 | `n`, `0`                  | No country mask; no existing initialization.                                              |
|   3--7 | `4`, `n`, `1`, `y`, `msl` | NetCDF; no summary; identify by names; COARDS; select `msl`.                              |
|  8--11 | `n`, `y`, `y`, `y`        | Do not translate; retain equator, Southern pole, and Northern pole.                       |
| 12--14 | `n`, `g`, `n`             | Do not make the source grid periodic; geodesic norm; default Plate Carrée interpretation. |
| 15--18 | `1 NX 1 NY`               | Full source-grid search rectangle.                                                        |
|     19 | `y`                       | Enter the analysis menu.                                                                  |

`NX,NY` are `144,73` or `1440,721`. The field name at line 7 is required
because the regular ERA5 files also contain `number` and `expver`; the F320
file has only `msl` and therefore auto-selects its sole field.

## Exact spectral tail

Lines 20--42 are the same logical answers for both resolutions:

```text
20  4             spatial spectral filtering
21  1             first frame
22  1             every frame
23  1000000       EOF-bounded end sentinel
24  0             least-squares method
25  0             full decomposition
26  42            source truncation
27  0             no memory-mapped storage
28  y             new Gaussian output grid
29  1             create, rather than read, the grid
30  42            new-grid truncation
31  n             omit the output longitude wraparound
32  2             two bands
33  0             first boundary
34  5             second boundary
35  42            third boundary
36  1             do not mask band 1
37  1             do not mask band 2
38  y             Hoskins filter
39  0.1           Hoskins cutoff
40  y             filter band 1
41  y             filter band 2
42  n             no large-value restriction
```

The resulting bands are T0--5 and T6--42. The checked-in streams contain two
additional trailing `n` answers (lines 43--44); TRACK's completed global
least-squares path does not read them. They are retained as part of the
validated transcript and do not affect the generated field.

The source file's `valid_time` coordinate is int64. TRACK 1.5.4's NetCDF
reader accepts short, int, float, and double data types but rejects int64 time
coordinates. The benchmark therefore runs a compatibility view that changes
only the time coordinate to int32 hours since 1900; `msl`, latitude, longitude,
and their values remain the same. The original-file failure and this
compatibility step are part of the benchmark record.
