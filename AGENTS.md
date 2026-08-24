# PyStormTracker Repository Instructions

These are high-level invariants for automated changes. Detailed design, API,
algorithm, testing, performance, and CI guidance belongs in `docs/`.

## Read first

- Inspect the actual checkout before editing: current branch/HEAD, working-tree
  status, and relevant diff. Do not assume an old prompt, report, or handoff
  still describes the current code, and preserve unrelated work.
- Read `docs/architecture.md` before changing data models, tracker interfaces,
  preprocessing, parallel execution, packaging, testing, or CI/CD.
- Read `docs/hodges.md` before changing Hodges detection, feature-point
  location, trajectory linking, optimization, or adaptive constraints.
- Read `docs/development/track-1.5.4-source-map.md` before changing behavior
  attributed to TRACK 1.5.4. Verify TRACK behavior from the cited source rather
  than inferring it from PyStormTracker.
- Read `docs/trackjson.md` before changing packed track models, canonical time,
  TrackJSON serialization, metadata, statistics, or schema behavior.
- Read the relevant roadmap/documentation before changing documented behavior.
- For literature-derived methods, check the original cited source and verify
  bibliographic metadata against a primary publisher or equivalent authority.

## Scientific correctness and terminology

- Preserve algorithm-specific behavior, defaults, thresholds, output formats,
  and supported backends unless the requested change requires otherwise.
- TRACK 1.5.4 is the implementation baseline for TRACK parity. For Hodges
  methods, prefer the published terminology of Hodges (1994, 1995, 1999), such
  as objects, feature points, trajectories, adaptive constraints, regional
  upper-bound displacements, and adaptive track smoothness constraint.
- Treat TRACK as an implementation/parity reference, not scientific ground
  truth. Clearly distinguish TRACK behavior, published methods, and
  PyStormTracker extensions.
- Do not replace a cited method with an approximation while retaining its name,
  and do not silently fall back from one scientific method to another.
- Feature-point refinement must remain associated with the detected feature and
  its documented local domain unless the method explicitly defines a global
  search.
- Equivalent cyclic longitude representations must describe the same physical
  field. Reorder coordinates and data together and do not introduce
  seam-dependent scientific results except where a documented compatibility
  path intentionally reproduces source behavior.
- Do not name scientific test/reference data an `oracle` or `fixture`; use the
  specific scientific/source term, such as analytic field, analytic solution,
  reference data, reference output, TRACK output, or NCL/Spherepack output.
  Pytest's `@pytest.fixture` mechanism is unaffected.
- Do not claim parity, accuracy, performance, superiority, or external
  validation without identifying the comparison and evidence.
- Do not weaken scientific assertions, tolerances, or retained reference
  results merely to make a failing test pass. Investigate the cause first.
- Do not duplicate scientific defaults where they can be obtained from their
  authoritative owning constant.

## Data models and formats

- Preserve the packed, array-backed `Tracks` architecture and immutable
  finalized output. Do not add persistent object-per-center or
  object-per-point storage to performance-critical paths.
- Preserve released TrackJSON wire contracts independently of Python API
  refactors.
- Keep canonical track time in the representation documented by
  `docs/trackjson.md`. Never guess that an ambiguous numeric external time or
  TRACK frame index is Unix time.
- Preserve source variable identity. Normalize values, thresholds, and units
  consistently, and keep stored values and metadata units consistent.
- Record only preprocessing that actually occurred. Keep optional scientific
  filtering distinct from transform bandwidth required internally for
  projection or regridding.

## Parallel execution and performance

- Serial, Dask, and MPI execution must preserve the same scientific semantics
  and deterministic canonical results for the tested scope.
- Keep Dask execution lazy from supported source I/O through preprocessing into
  frame-level work. Do not materialize the complete time series merely to
  construct Dask tasks.
- Compute each unique source frame once. Overlapping Hodges/HEALPix MGE
  segments must reuse frame results rather than repeat detection/refinement.
- MPI must assign work before heavy processing, perform rank-local
  loading/preprocessing, avoid nested Dask schedulers, and gather results in
  deterministic source order.
- Do not add nested thread pools, Numba parallel regions, or other inner
  parallelism beneath Dask/MPI without profiling and explicit control of
  oversubscription.
- Keep scientific segmentation parameters separate from I/O/task scheduling.
  Benchmark representative workloads and measure the actual bottleneck before
  redesigning execution or making scaling claims.

## Tests

- Unit tests exercise one function, numerical primitive, or tightly bounded
  class contract using deterministic analytic or constructed inputs.
- Integration tests exercise multiple current PyStormTracker components, real
  I/O, or execution backends. Parity tests compare against TRACK, a historical
  PyStormTracker version, NCL/Spherepack, or another identified implementation.
- `slow` describes runtime cost and is independent of unit/integration/parity.
- Do not add tests merely to increase coverage. A regression test should
  reproduce the numerical, scientific, API, or behavioral failure being
  protected rather than implementation details.
- Routine parallel-result equivalence tests use exactly 4 workers. Other worker
  counts require a specific concurrency defect; performance scaling belongs in
  benchmarks.
- Prefer scientifically meaningful expected values, analytic properties, and
  explicit tolerances justified by the method or comparison.
- Backend-equivalence tests should compare complete canonical results, not only
  counts or summary statistics.
- Parity comparisons must hold input data, preprocessing, configuration, and
  comparison population constant before attributing differences to a method.
- Avoid uncontrolled network dependencies. Required reference data missing from
  a parity CI job is a failure, not a successful skip.
- Fix project warnings at their source rather than adding broad warning
  suppressions.

## Engineering and change control

- Use `uv` for dependency management, environment synchronization, builds, and
  development commands.
- Keep one authoritative owner for each scientific default, closed domain type,
  numerical primitive, and shared algorithm. Avoid generic constants/types/
  kernels dumping grounds when a concept has a clear owner.
- Provide explicit type annotations; do not introduce `Any`, casts,
  compatibility aliases, or shims merely to bypass typing or preserve an
  unreleased API.
- Expose scientifically meaningful choices in public APIs. Keep implementation
  switches private unless intentionally part of the user contract.
- Library modules use module loggers and do not configure global logging.
  CLI logging/progress must not participate in scientific scheduling or results.
- Use direct technical and scientific prose. Distinguish implemented behavior,
  repository-tested behavior, external validation, and planned work.
- Do not retain agent prompts, temporary plans, progress transcripts, or large
  generated validation artifacts as permanent project documentation. Preserve
  reproducibility code, compact evidence, and required reference data instead.
- Keep changes limited to the requested task and update the relevant `docs/`
  when behavior, API contracts, or scientific rationale changes.
- Do not stage, commit, push, publish, rewrite history, or create releases
  unless explicitly requested.
