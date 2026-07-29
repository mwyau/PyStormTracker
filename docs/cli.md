# CLI Reference

PyStormTracker provides a unified command-line interface via the `stormtracker` command.

## `stormtracker track`

Runs the core storm tracking algorithm.

**Usage:**
```bash
stormtracker track -i input.nc -v vo -o tracks.txt -m max -a hodges
```

**Key Arguments:**
- `-i`, `--input`: Input NetCDF file.
- `-v`, `--var`: Variable to track (e.g., `vo`, `msl`).
- `-o`, `--output`: Output track file.
- `-a`, `--algorithm`: Tracking algorithm (`simple` or `hodges`).
- `-m`, `--mode`: Detection mode (`min` for SLP, `max` for vorticity).
- `--map-proj`: Map projection for detection (`global`, `nh_stereo`, `sh_stereo`, `healpix`).

## `stormtracker sample`

Samples variables from a NetCDF dataset along existing storm tracks.

**Usage:**
```bash
# Sample precipitation within a 500km radius of storm centers
stormtracker sample -i tracks.json -d precip.nc -v pr -o tracks_with_pr.json -m mean -r 500
```

**Key Arguments:**
- `-i`, `--input`: Input track file (JSON).
- `-d`, `--data`: Input NetCDF data file to sample from.
- `-v`, `--var`: Variable name in the NetCDF file.
- `-m`, `--method`: Sampling method (`nearest`, `bilinear`, `mean`, `max`, `min`).
- `-r`, `--radius`: Radius in km for spatial methods (`mean`, `max`, `min`).

## `stormtracker compare`

Matches tracks from a comparison set to a reference set based on spatial proximity and temporal overlap.

**Usage:**
```bash
stormtracker compare --ref era5.json --comp model.json --max-dist 200 --json
```

**Key Arguments:**
- `--ref`: Reference track file.
- `--comp`: Comparison track file.
- `--max-dist`: Maximum mean geodetic distance (km) allowed for a match (default 440).
- `--min-overlap`: Minimum overlap ratio required (default 0.1).
- `-o`, `--output`: Output filtered comparison track file.

## `stormtracker convert`

Converts PyStormTracker data between formats and generates interactive HTML visualizations.

**Usage:**
```bash
# Convert IMILAST to JSON
stormtracker convert -i tracks.txt -o tracks.json -f imilast -F json

# Generate interactive HTML explorer
stormtracker convert -i tracks.json -o explorer.html -f json -F html
```

**Key Arguments:**
- `-i`, `--input`: Input file path.
- `-o`, `--output`: Output file path.
- `-f`, `--in-format`: Input format (`imilast`, `json`).
- `-F`, `--out-format`: Output format (`imilast`, `hodges`, `json`, `html`).
