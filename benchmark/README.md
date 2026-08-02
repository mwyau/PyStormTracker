# PyStormTracker Benchmarks

This directory keeps executable benchmark scripts in `scripts/` and historical
benchmark outputs at the directory root.

## Format I/O benchmark

`scripts/benchmark_json_geojson.py` generates deterministic synthetic trajectories
and compares TrackJSON v1.0 with GeoJSON serialization size, compressed size,
JSON parse time, and NumPy-array construction time.

```bash
uv run python benchmark/scripts/benchmark_json_geojson.py
```

Its output depends on the Python and NumPy versions and the host hardware.
Record that environment with any benchmark result used for comparison.
