# Implementation Plan: TrackJSON v1.0 Schema Iteration, GeoJSON I/O & IO Facade Architecture

## Overview
This plan formalizes PyStormTracker's native JSON format as **TrackJSON v1.0**, generalizing the hardcoded `"strength"` key into a multi-variable SoA structure (`points.variables`), adding optional GIS-standard **GeoJSON** import/export support, centralizing format inference and IO routing into `src/pystormtracker/io/format.py`, refactoring the `benchmark/` directory structure, and updating all CLI commands and documentation to default to TrackJSON.

---

## User Review Required

> [!IMPORTANT]
> **TrackJSON v1.0 Schema Design Iteration:**
> 1. **Replacing Hardcoded `"strength"`:** Instead of assuming a single hardcoded `"strength"` key, TrackJSON v1.0 introduces:
>    - `metadata.primary_var`: Explicit name of the primary tracking variable (e.g. `"msl"`, `"vo"`, `"u10"`).
>    - `metadata.intensity_mode`: `"min"` (for pressure lows) or `"max"` (for vorticity/wind peaks).
>    - `metadata.variable_units`: Dictionary mapping variable names to physical units (e.g. `{"msl": "Pa", "vo": "s^-1"}`).
>    - `points.variables`: Map of variable names to flat SoA arrays (`{"msl": [...], "vo": [...]}`).
>    - `track.peak_values` & `track.mean_values`: Summary dictionaries per track replacing scalar `track.strength`.
>    - Backward compatibility: `points.strength` aliases `points.variables[primary_var]` for legacy WebGL viewers.
> 2. **Benchmark Directory Refactoring:**
>    Organize `benchmark/` into clean functional subdirectories:
>    ```text
>    benchmark/
>    ├── io/
>    │   └── benchmark_format_io.py    (TrackJSON vs GeoJSON multi-scale benchmark)
>    ├── tracking/
>    │   ├── run_benchmark_detailed.py (Tracker runtime benchmarks)
>    │   └── generate_stacked_charts.py
>    ├── data/
>    │   ├── benchmark_detailed_v0.3.3.json
>    │   └── benchmark_detailed_v0.4.0.json
>    └── README.md                      (Comprehensive benchmark report & guide)
>    ```

---

## TrackJSON v1.0 Schema Specification (High-Performance & Extensible)

```json
{
  "format": "TrackJSON/1.0",
  "metadata": {
    "primary_var": "msl",
    "mode": "min",
    "units": {
      "msl": "Pa",
      "vo": "s^-1"
    },
    "bounds": {
      "min_time": 1577836800000,
      "max_time": 1577923200000,
      "min_lat": -60.0,
      "max_lat": 60.0,
      "min_lon": -180.0,
      "max_lon": 180.0
    }
  },
  "points": {
    "lat": [10.0, 11.0, null, 20.0, 21.0],
    "lon": [50.0, 52.0, null, 60.0, 62.0],
    "time": [1577836800000, 1577858400000, null],
    "variables": {
      "msl": [101000.0, 99000.0, null, 101200.0, 99200.0],
      "vo": [1.2e-4, 2.5e-4, null, 1.1e-4, 2.1e-4]
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

### JSON Schema Required vs. Optional Fields

- **Required Track Item Fields:** `["track_id", "start", "end"]`
- **Optional Visualization Acceleration Fields:** `start_lat`, `start_lon`, `start_time`, `end_lat`, `end_lon`, `end_time`, `peak_lat`, `peak_lon`, `peak_time`, `peak_value`, `duration_hours`.
- **Duck-Typing / Open Extensibility:** Schema defines `additionalProperties: true` for `metadata` and `track` objects, allowing users and scripts to attach custom properties (e.g. `"basin"`, `"category"`, `"model"`).

---

## Proposed Changes

### 1. Refactor Benchmark Directory (`benchmark/`)

#### [MODIFY] `benchmark/`
- Reorganize files into `benchmark/io/`, `benchmark/tracking/`, and `benchmark/data/`.
- Add `benchmark/io/benchmark_format_io.py` and `benchmark/README.md`.

---

### 2. Schema Specification & Serialization (`src/pystormtracker/io/json.py`)

#### [NEW] `docs/schemas/trackjson.schema.json`
Save the formal Draft 2020-12 JSON Schema for TrackJSON v1.0.

#### [MODIFY] `src/pystormtracker/io/json.py`
- Implement generic TrackJSON v1.0 serializer supporting `primary_var`, `variable_units`, `points.variables`, `track.peak_values`, and `track.mean_values`.
- Reconstruct all variables seamlessly in `read_json`.

---

### 3. GeoJSON I/O Support (`src/pystormtracker/io/geojson.py`)

#### [NEW] `src/pystormtracker/io/geojson.py`
- `write_geojson(tracks, outfile)` & `read_geojson(infile)` for GIS software interoperability (`format="geojson"`).

---

### 4. Centralized IO Router (`src/pystormtracker/io/format.py`)

#### [NEW] `src/pystormtracker/io/format.py`
Centralized IO router (`load_tracks`, `save_tracks`, `infer_format`).

---

### 5. CLI & Documentation Updates

#### [MODIFY] `src/pystormtracker/track.py`, `convert.py`, `compare.py`, `sample.py`
Default all subcommands to TrackJSON. Remove `generate_html` / `--split` options.

#### [MODIFY] `docs/cli.md`, `docs/api.md`, `README.md`, `docs/architecture.md`
Update CLI reference, Python API guides, and architecture documentation.

