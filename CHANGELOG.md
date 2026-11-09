# Changelog

## Post-refactor cleanup — 2026-11-09

- Record the intentional removal of 345 CSVs and eight PNGs from three obsolete
  result trees in `1b2344a`; retain the curated `final_results/` and source data.
- Scope the original preservation evidence to refactor commit `89466f0`.
- Remove the incomplete `thesis_tex` Git link and ignore the separately
  maintained local manuscript checkout, preserving its contents and Git metadata.

## 0.1.0 — research codebase refactor (2026-11-09)

- Separate connectome IO, topology, weight controls, and spectral normalization.
- Separate simulation configuration, sampling, variant evaluation, and CSV storage.
- Split reservoir dynamics and KR/GR stream protocols while retaining imports.
- Split plotting into cohesive feature modules; share exact aggregation/path helpers.
- Preserve legacy commands, labels, seeds, grids, metric definitions, schemas, and grouping.
- Restore the optional weight-performance plot's disabled exporter and remove its
  pre-existing undefined-variable crash; scientific calculations are unchanged.
- Add deterministic CPU baselines, invariant/controlled-example tests, CLI and
  rendering checks, a reusable plotting-validation driver, and package metadata.
- Correct effective-rank, ridge-solver, sign-control, and workflow documentation.
- Record historical scientific issues for separate corrections; preserve archived
  source data/results/figures and all `thesis_tex/` contents.

Review and validation details: [docs/REFACTOR_LOG.md](docs/REFACTOR_LOG.md).
