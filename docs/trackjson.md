# TrackJSON v1.0

TrackJSON/1.0 is PyStormTracker's native JSON representation of immutable packed
`Tracks`. It is provisional and is not a compatibility promise for released
files.

## Canonical structure

The root object contains the required members `format`, `metadata`, `index`,
and `data`. It may also contain `stats`.

```json
{
  "format": "TrackJSON/1.0",
  "metadata": {
    "primary_var": "msl",
    "mode": "min",
    "units": {
      "msl": "Pa"
    },
    "time": {
      "units": "milliseconds since 1970-01-01 00:00:00",
      "calendar": "proleptic_gregorian"
    },
    "bounds": {
      "south": 0.0,
      "north": 90.0,
      "west": 120.0,
      "east": -100.0
    }
  },
  "index": {
    "ids": [
      10,
      20,
      35
    ],
    "offsets": [
      0,
      4,
      7,
      12
    ]
  },
  "data": {
    "times": [],
    "lats": [],
    "lons": [],
    "variables": {
      "msl": []
    }
  },
  "stats": {
    "version": 1,
    "point_count": [],
    "start_time": [],
    "end_time": [],
    "duration_hours": [],
    "start_lat": [],
    "start_lon": [],
    "end_lat": [],
    "end_lon": [],
    "south_lat": [],
    "north_lat": [],
    "west_lon": [],
    "east_lon": [],
    "antimeridian_wrap": [],
    "peak_time": [],
    "peak_lat": [],
    "peak_lon": [],
    "peak_value": [],
    "path_length_km": [],
    "displacement_km": []
  }
}
```

`stats` is optional and is omitted when unavailable or when the writer is
called with `include_stats=False`. Optional `metadata.bounds` is omitted when
absent. Explicit `null` is not valid for either optional object.

`index.ids` has shape `(T,)`. `index.offsets` is the complete boundary buffer
with shape `(T + 1,)`; it starts at zero, ends at `N`, and is strictly
increasing when `T > 0`. Track `i` is always
`slice(offsets[i], offsets[i + 1])`. The valid empty index is `ids: []` and
`offsets: [0]`. All point columns and variable columns have length `N`.

`metadata.units` has exactly the same keys as `data.variables`, and
`metadata.primary_var` is one of those keys even for an empty document.
`metadata.mode` is the concrete value `min` or `max`.

`Tracks` contains only canonical trajectory data and metadata. The optional
wire `stats` member is a derived cache: it does not define membership and is
never retained by the core model. Default writes omit it. Default reads decode
and structurally validate it if present, then discard it. `verify_stats=True`
recomputes every field, reports the first mismatch, discards the supplied and
computed statistics, and returns canonical `Tracks`; if no stats member is
present, verification succeeds because there is nothing to verify.

The selected source variable name is preserved exactly. Preprocessing is
recorded in the optional `metadata.processing` sequence rather than encoded in
temporary variable names. Each entry contains an `operation`, an `enabled`
flag, and JSON-scalar `parameters`.

For tracking inputs, both `lmin` and `lmax` must be supplied to record a
`spectral_filter` operation; omitting both means that no optional spectral
filter occurred. `taper_points` is independent. A projection or HEALPix
conversion may record a `regrid` operation with its derived
`transform_lmax` even when no optional filter occurred. The transform bandwidth
must not be interpreted as a user-requested filter.

Recognized pressure variables are stored in pascals and recognized vorticity
variables in inverse seconds. Source values and detection thresholds are
converted together before detection. Custom variable units are preserved when
declared; an undeclared custom unit is recorded as `"1"`.

## Coordinate and time conventions

These are conventions of TrackJSON/1.0, not repeated per-document metadata:

- Latitude is degrees north in `[-90, 90]`.
- Longitude is degrees east, normalized to `[-180, 180)`; `180` is therefore
  represented as `-180` in point data.
- `metadata.time.units` is required and must be exactly
  `milliseconds since 1970-01-01 00:00:00`.
- `metadata.time.calendar` is required. `data.times` and the time fields in
  `stats` are signed integer millisecond offsets under that calendar. They
  are not ISO strings, `yyyymmdd` values, or generic Unix timestamps.
- TrackJSON uses only the canonical calendar `proleptic_gregorian`. Source
  metadata values `standard`, `gregorian`, and `proleptic_gregorian` are
  canonicalized to it. Explicitly declared `standard` dates before
  `1582-10-15` are rejected because mixed Julian/Gregorian conversion is not
  implemented.
- The packed time range is restricted to signed JavaScript safe integers:
  `-9007199254740991 <= time <= 9007199254740991`. This preserves exact
  values when the standalone explorer parses JSON numbers as JavaScript
  `Number`.
- Python `datetime` and NumPy `datetime64` input without source calendar
  metadata uses `proleptic_gregorian`. CF/netCDF input uses its declared
  calendar and defaults to `proleptic_gregorian` when the attribute is absent.
  All finalized `Tracks` objects use the canonical calendar name.
- Coordinates cannot be null, NaN, or infinite.
- Variable NaN values are encoded as JSON `null`; unavailable peak fields are
  also encoded as `null`.

TrackJSON v1 explicitly rejects `360_day`, `noleap`, `365_day`, `all_leap`,
`366_day`, `julian`, `utc`, `tai`, `none`, custom calendars, explicit
month-length calendars, and other CF 1.13 calendar facilities. Broader CF
calendar support is deferred; unsupported calendars are not silently
reinterpreted.

TrackJSON serializes finite floating-point wire values with up to 15 significant
decimal digits to suppress insignificant binary floating-point tails. This
preserves scientific precision across magnitudes but does not promise
bit-identical float64 round trips.

The document does not repeat longitude-convention, longitude-range,
longitude-bound, latitude-range, or time-unit members.

## Declared spatial bounds

`metadata.bounds` is an optional declared tracking, filtering, analysis, or
visualization domain. It is not required to be the pointwise extrema of every
complete trajectory. A regional selection may retain a complete trajectory
that intersects the domain.

`bounds` contains `south`, `north`, `west`, and `east`:

- `-90 <= south <= north <= 90`.
- `-180 <= west <= 180` and `-180 <= east <= 180`.
- `west < east` is an ordinary interval.
- `west > east` crosses the antimeridian and means
  `[west, 180] union [-180, east]`.
- `west=-180, east=180` is the complete global longitude domain.
- Other `west == east` pairs are ambiguous and invalid.

The bounds do not contain an `antimeridian_wrap` member. That relationship is
defined by `west > east`. Bounds are preserved by ordinary subsets and are
left absent when a source format has no domain metadata. The explorer uses
declared bounds for its initial view and derives a view from point data when
they are absent. It does not provide the later full bounding-box filtering UI.

## Precomputed per-track statistics

`stats` contains derived, optional precomputed per-track statistics. It does
not define trajectory membership. Every array has length `T`, and row `i`
corresponds to `index.ids[i]`. Version 1 stores actual metrics only; it stores
no threshold-result flags. Application thresholds such as 48 hours and
1000 km remain configurable behavior.

| Field               | JSON type     | Shape  | Unit                        | Nullable | Definition                                                     |
| ------------------- | ------------- | ------ | --------------------------- | -------- | -------------------------------------------------------------- |
| `version`           | integer       | scalar | —                           | no       | Statistics format version, currently `1`.                      |
| `point_count`       | integer array | `(T,)` | points                      | no       | `diff(index.offsets)` for each track.                          |
| `start_time`        | integer array | `(T,)` | CF ms under `metadata.time` | no       | Time of the first point.                                       |
| `end_time`          | integer array | `(T,)` | CF ms under `metadata.time` | no       | Time of the final point.                                       |
| `duration_hours`    | number array  | `(T,)` | hours                       | no       | `(end_time - start_time) / 3,600,000`.                         |
| `start_lat`         | number array  | `(T,)` | degrees north               | no       | Latitude of the first point.                                   |
| `start_lon`         | number array  | `(T,)` | degrees east                | no       | Signed longitude of the first point.                           |
| `end_lat`           | number array  | `(T,)` | degrees north               | no       | Latitude of the final point.                                   |
| `end_lon`           | number array  | `(T,)` | degrees east                | no       | Signed longitude of the final point.                           |
| `south_lat`         | number array  | `(T,)` | degrees north               | no       | Minimum track latitude.                                        |
| `north_lat`         | number array  | `(T,)` | degrees north               | no       | Maximum track latitude.                                        |
| `west_lon`          | number array  | `(T,)` | degrees east                | no       | Start edge of the shortest longitude arc containing the track. |
| `east_lon`          | number array  | `(T,)` | degrees east                | no       | End edge of the shortest longitude arc containing the track.   |
| `antimeridian_wrap` | boolean array | `(T,)` | —                           | no       | Whether that shortest arc crosses the antimeridian.            |
| `peak_time`         | integer array | `(T,)` | CF ms under `metadata.time` | yes      | Time of the finite primary-variable extremum.                  |
| `peak_lat`          | number array  | `(T,)` | degrees north               | yes      | Latitude at the primary-variable extremum.                     |
| `peak_lon`          | number array  | `(T,)` | degrees east                | yes      | Signed longitude at the primary-variable extremum.             |
| `peak_value`        | number array  | `(T,)` | primary-variable unit       | yes      | Minimum or maximum value according to `metadata.mode`.         |
| `path_length_km`    | number array  | `(T,)` | km                          | no       | Cumulative great-circle distance between consecutive points.   |
| `displacement_km`   | number array  | `(T,)` | km                          | no       | Great-circle distance from the first to final point.           |

Peak fields are either all present or all null for a row. Missing primary
variable values are ignored; a track with no finite primary value has a
missing peak. Ties use the first occurrence. A one-point track has zero path
length and displacement; for two points, path length equals displacement.
Stats inherit the document’s single time encoding; they do not repeat units or
calendar metadata.

## Schema and implementation

`msgspec` wire structs are the typed source of truth for the wire structure.
The committed `src/pystormtracker/schemas/trackjson.schema.json` is generated
from those structs. Runtime reads use typed `msgspec` decoding followed by
semantic cross-field validation; runtime JSON Schema validation is not used.

Regenerate the committed schema manually with:

```bash
uv run python scripts/generate_trackjson_schema.py
```

CI checks for drift without rewriting the file:

```bash
uv run python scripts/generate_trackjson_schema.py --check
```

The writer constructs typed wire objects directly and uses
`encode_trackjson()`. There is one reusable module-level encoder and decoder.

## Integration test data

Integration data are maintained separately from the software checkout. The
current checkout retains one December 2025, 2.5-degree ERA5 MSL
input at `tests/data/era5/era5_msl_2025-12_2.5x2.5.nc`; it does not retain a
trajectory sample. TrackJSON unit tests construct small synthetic `Tracks`
values and exercise the wire contract without external data.
