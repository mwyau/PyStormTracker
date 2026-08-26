# Testing

The test categories have distinct scientific meanings:

- **UNIT** (`tests/unit/`): Analytic, synthetic, and local behavior exercised
  offline.
- **INTEGRATION** (`tests/integration/`): Multiple current PyStormTracker
  components exercised together on real data and evaluated with
  internal/current scientific expectations, rather than primarily against
  another implementation.
- **PARITY** (`tests/parity/`): A current PyStormTracker run from an external
  input file through its normal public pipeline to final trajectories,
  compared with a static final trajectory output from another implementation
  or historical PyStormTracker version. Numerical parity is also allowed for
  a bounded package numerical component compared with static NCL/Spherepack
  output; it must not be confused with trajectory parity or source-stage
  replay.

The main checkout has one historical v0.0.2 trajectory-parity test, one bundled
2.5-degree T5-42 NCL spectral numerical-parity case, and broader external-data
comparisons. The completed 2024 TRACK 1.5.4 comparison covers F320 → T42 and
F320 → F320 full-year 2024 ERA5 MSLP, runtime measurements, raw trajectories,
and RSPLICE-filtered trajectories. The full-year filtered one-to-one F1 is
0.997 in both cases. It is not a package parity test; TRACK source-stage
probes, MGE replay, manifests, and source-stage outputs are not part of the
package test suite.

The orthogonal markers are `integration`, `parity`, `data`, and `slow`.
`data` means that scientific or reference data are required but not bundled
with the source distribution; reading a bundled file does not imply `data`.
`slow` means computationally expensive and is independent of test category or
data ownership. Useful selections include `-m "not data"`,
`-m "integration and not data"`, `-m "parity and not data"`, `-m data`, and
`-m "not slow"`.

## Local commands

The default command is the fast unit suite:

```bash
uv run pytest
```

Run current local integration tests explicitly:

```bash
uv run pytest tests/integration -m "not slow and not data"
```

Run bundled numerical parity tests and any future bundled trajectory parity
with `uv run pytest tests/parity -m "not slow and not data"`. This currently
executes the one local NCL spectral case. Run the legacy trajectory and
external NCL/Spherepack cases when the pinned Data tag and exact external
assets are available.

Tests marked `data` exercise GRIB, reduced-Gaussian, remote-Zarr, or legacy
reference contracts owned by the sibling `PyStormTracker-Data` repository.
Run those tests explicitly when the pinned Data tag and its exact release
assets are available.

Scientific validation is separate from package parity. Source-stage
reproduction, TRACK internals, NCL/Spherepack reference-generation methodology,
reconciliation experiments, and scientific-validation evidence are outside the
package test suite. The main package tests consume only the one compact static
NCL/Spherepack output committed under `tests/data/ncl/`; broader reference data
are owned by `PyStormTracker-Data`.

Use `--durations=30 --durations-min=0.5` when auditing runtime concentration.
Routine Dask backend-equivalence tests use four workers. Worker-count scaling
belongs in benchmarks unless a specific concurrency defect requires another
case.

Ordinary temporal integration tests select the explicit December 2025 period,
from `2025-12-01T00:00` through `2025-12-31T18:00` (124 six-hourly frames).
Bounded numerical and synthetic tests retain only the smallest input needed
for their scientific assertion.

## Scientific assertions and reference data

Prefer analytic solutions, exact mathematical constructions, physical or
mathematical invariants, and identified external comparisons in that order.
Numerical assertions set both `rtol` and `atol` explicitly; comments explain
non-obvious tolerances using floating-point behavior, source precision,
interpolation convergence, physical scale, or the retained comparison.

Parity reference data identify their source and version where that context is
scientifically relevant. Small bundled NCL outputs are consumed directly by
the numerical-parity tests and do not require a manifest or checksum registry.
If a parity job promises that committed reference data are present, missing
data fail the test rather than becoming a successful skip.
