# Testing

The test categories have distinct scientific meanings:

- **UNIT** (`tests/unit/`): Analytic, synthetic, and local behavior exercised
  offline.
- **INTEGRATION** (`tests/integration/`): Multiple PyStormTracker components
  exercised together on real data and checked against PyStormTracker behavior.
- **PARITY** (`tests/parity/`): A PyStormTracker run from an external input
  through the public workflow to final trajectories, compared with static final
  trajectory output from another implementation or an earlier PyStormTracker
  version. Numerical parity may also compare a bounded numerical component with
  static NCL/Spherepack output; this is distinct from trajectory parity and
  source-stage replay.

The repository includes one v0.0.2 trajectory-parity case, one bundled
2.5-degree T5-42 NCL scalar spectral parity case, and broader external-data
comparisons. The completed 2024 TRACK 1.5.4 comparison covers F320 → T42 and
F320 → F320 full-year 2024 ERA5 mean sea-level pressure, runtime measurements,
raw trajectories, and RSPLICE-filtered trajectories. This TRACK comparison is
part of the scientific validation record; source-stage probes, MGE replay,
manifests, and source-stage outputs are maintained with the validation work.

The orthogonal markers are `integration`, `parity`, `data`, and `slow`.
`data` marks tests that require scientific or reference data outside the source
distribution. `slow` means computationally expensive and is independent of test
category or data location. Useful selections include `-m "not data"`,
`-m "integration and not data"`, `-m "parity and not data"`, `-m data`, and
`-m "not slow"`.

## Local commands

The default command runs the fast unit suite:

```bash
uv run pytest
```

Run bundled integration tests explicitly:

```bash
uv run pytest tests/integration -m "not slow and not data"
```

Run bundled parity tests with:

```bash
uv run pytest tests/parity -m "not slow and not data"
```

This executes the bundled NCL scalar spectral case.

Tests marked `data` use GRIB, reduced-Gaussian, remote-Zarr, or other external
reference data stored in the sibling `PyStormTracker-Data` repository. The
pinned `v0.2.0-data` release contains the ERA5 inputs used by the current
integration and parity tests.

Scientific validation covers TRACK source-stage reproduction, TRACK internals,
NCL/Spherepack reference generation, reconciliation experiments, and other
scientific-validation evidence. Package parity uses compact static references,
including the NCL T5-42 spectral output under `tests/data/ncl/`; broader
reference data are stored in `PyStormTracker-Data`.

Use `--durations=30 --durations-min=0.5` to identify slow tests. Routine Dask
backend-equivalence tests use four workers. Worker-count scaling belongs in
benchmarks unless a specific concurrency defect requires another case.

Ordinary temporal integration tests select the explicit December 2025 period,
from `2025-12-01T00:00` through `2025-12-31T18:00` (124 six-hourly frames).
Bounded numerical and synthetic tests use only the input needed for their
scientific assertion.

## Scientific assertions and reference data

Prefer analytic solutions, exact mathematical constructions, physical or
mathematical invariants, and identified external comparisons in that order.
Numerical assertions set both `rtol` and `atol` explicitly; comments explain
non-obvious tolerances using floating-point behavior, source precision,
interpolation convergence, physical scale, or the reference comparison.

Parity reference data identify their source and version where that context is
scientifically relevant. Small bundled NCL outputs are consumed directly by
the numerical-parity tests and do not require a manifest or checksum registry.
Required committed parity data must be present; missing data fail the test.
