# TrackJSON v1.0

TrackJSON is PyStormTracker's native trajectory format. It stores coordinates
and meteorological variables as separator-delimited structure-of-arrays (SoA)
data, making all point values available without a nested object per point.

```bash
stormtracker track -i input.nc -v msl -o tracks.trackjson
```

`read_json()` and `write_json()` use TrackJSON v1.0. The formal
machine-readable contract is available at
<https://raw.githubusercontent.com/mwyau/PyStormTracker/main/schema/trackjson.schema.json>
and uses JSON Schema Draft 2020-12.

## Document structure

- `format` is `"TrackJSON/1.0"`.
- `metadata.primary_var` names the variable used to track features, for example
  `msl` or `vo`.
- `metadata.mode` is `min` for minima or `max` for maxima.
- `metadata.units` maps variable names to physical-unit strings when known.
- `data` holds flat `lat`, `lon`, `time`, and `variables` arrays. A `null`
  value separates tracks.
- `tracks` records the point-array range for each trajectory and optional
  summary fields.

## Global time and spatial bounds

`metadata.bounds` contains `min_time`, `max_time`, `min_lat`, `max_lat`,
`min_lon`, and `max_lon`. Consumers can use these precomputed values to set
time controls and map extents without scanning every point array.

## Per-track acceleration fields

Each item in `tracks` must contain `track_id`, `start`, and `end`. It can also
provide these optional summaries:

- Genesis: `start_lat`, `start_lon`, `start_time`
- Lysis: `end_lat`, `end_lon`, `end_time`
- Primary-variable peak: `peak_lat`, `peak_lon`, `peak_time`, `peak_value`
- Lifetime: `duration_hours`

The schema permits additional properties in metadata and track objects, so
applications may add fields such as `basin`, `category`, or `model`.

## Example

```json
{
  "format": "TrackJSON/1.0",
  "metadata": {
    "primary_var": "msl",
    "mode": "min",
    "units": {"msl": "Pa", "vo": "s^-1"},
    "bounds": {
      "min_time": 1577836800000,
      "max_time": 1577923200000,
      "min_lat": -60.0,
      "max_lat": 60.0,
      "min_lon": -180.0,
      "max_lon": 180.0
    }
  },
  "data": {
    "lat": [10.0, 11.0, null, 20.0, 21.0],
    "lon": [50.0, 52.0, null, 60.0, 62.0],
    "time": [1577836800000, 1577858400000, null, 1577836800000, 1577858400000],
    "variables": {
      "msl": [101000.0, 99000.0, null, 101200.0, 99200.0],
      "vo": [0.00012, 0.00025, null, 0.00011, 0.00021]
    }
  },
  "tracks": [
    {
      "track_id": 1,
      "start": 0,
      "end": 1,
      "start_lat": 10.0,
      "start_lon": 50.0,
      "start_time": 1577836800000,
      "end_lat": 11.0,
      "end_lon": 52.0,
      "end_time": 1577858400000,
      "peak_lat": 11.0,
      "peak_lon": 52.0,
      "peak_time": 1577858400000,
      "peak_value": 99000.0,
      "duration_hours": 6.0
    }
  ]
}
```
