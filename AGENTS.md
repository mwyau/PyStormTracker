# PyStormTracker Repository Instructions

These instructions apply to automated changes in this repository.

## References

- Read `docs/architecture.md` before changing data models, tracker interfaces, preprocessing, parallel execution, testing, packaging, or CI/CD.
- Read `docs/hodges.md` before changing Hodges detection, linking, optimization, constraints, or sub-grid refinement.
- Read the relevant `docs/roadmap.md` documents before changing documented behavior.

## Scientific and behavioral requirements

- Preserve algorithm-specific behavior, defaults, thresholds, output formats, and supported backends unless the requested change requires otherwise.
- Maintain documented TRACK/Hodges parity. Do not replace the original method with an approximation. Alternative methods must be explicit options.
- Parallel implementations must reproduce deterministic serial results for the tested scope.
- Do not claim parity, accuracy, performance, or external validation without identifying the comparison and evidence.

## Engineering requirements

- Use `uv` for dependency management, environment synchronization, builds, and development commands.
- Keep numerical kernels vectorized or Numba-compiled where required by the architecture.
- Do not introduce persistent object-per-center or object-per-point storage in performance-critical paths.
- Provide explicit type annotations. Do not introduce `Any`.
- Keep tracker implementations compatible with the shared protocol, including `**kwargs` where required for cross-algorithm calls.
- Use `DataLoader` and coordinate-aware Xarray paths for supported meteorological inputs.
- Add or update focused tests for behavioral changes. Preserve historical fixtures and tolerances unless the change corrects them explicitly.

## Documentation and change control

- Use direct technical and scientific prose. Preserve established terminology and define acronyms when first used.
- Avoid promotional language, buzzwords, unsupported adjectives, and claims broader than the available evidence.
- Distinguish implemented behavior, repository-tested behavior, external validation, and planned work.
- Keep changes limited to the requested task. Preserve design rationale unless it is obsolete and the replacement is documented.
- Do not stage, commit, push, publish, or create releases unless explicitly requested.
