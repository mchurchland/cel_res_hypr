# Actor–Critic iteration log

## 1. Audit and numerical baseline — ACCEPT

**Actor:** Recorded the original module/directory inventory, reviewed entry
points and scientific protocols, created the refactoring branch, and captured
a seeded 54-row CPU baseline before scientific implementation changes.

**Validation:** Original unittest and pytest suites: 15 passed. Baseline plus
original suite: 16 passed. Original synthetic graph, configuration, trajectory,
schema, and numerical outputs are archived in `docs/baseline/`.

**Critic:** Independently inspected the new files and reran all 16 tests
(2.21 s). JSON and CSV fixtures agree exactly. No scientific source edits.
Low: add serialized CSV read-back checks in the persistence iteration; document
the smoke tool's development dependencies or make optional metadata optional.
Both are tracked for later iterations.

**Decision:** ACCEPT. Historical scientific issues are recorded in `AUDIT.md`
and will receive regression tests rather than numerical corrections.

## 2. Connectome and network construction — ACCEPT

**Actor:** Extracted connectome loading, directed topology, weight/sign controls,
spectral normalization, construction, and global seeding into `cel_res_hypr`.
Retained every original utility function through explicit `util.util` exports.
Corrected the builder's obsolete return description and the misleading
power-iteration implication of the radius helper's name.

**Validation:** 32 CPU tests passed, including graph degrees/edge counts,
weight/sign invariants, loading, seeded builds, both normalizations, and the
54-row baseline. All 31 functions have identical executable ASTs after
excluding docstrings. Legacy CLI help remains operational.

**Critic:** Independently ran 32 tests (2.66 s), checked AST identity, exports,
and acyclic dependencies. No blocking findings. Follow-ups: clarify the
`original_radius`/`None` convention and float32 edge-mask return; document/test
the unused historical row-indexing defect in `_shuffle_ce_weights_except_1`.

**Decision:** ACCEPT. Scientific expressions and RNG calls are unchanged.

## 3. Experiments, dynamics, protocols, and persistence — ACCEPT after revision

**Actor:** Separated configuration, evaluation, variant dispatch, grid/subset
sampling, and CSV storage. Kept `reservoir_variants` exports and CLI helpers.
Extracted dynamics and input-stream protocols while retaining `run_one`'s
patchable imports. Consolidated repeated dispatch into one evaluation call.
Corrected effective-rank, IPC-solver, builder, and variant documentation.

**Validation:** 47 CPU tests passed, including analytical trajectories,
single-delay MC/IPC recovery, ridge epsilon, rank distributions, seed/subset
invariants, CSV read-back/selective updates, and an actual legacy CLI run.
All 54 baseline rows remain exactly equal. CLI help is unchanged. Executable
function ASTs match the original except the explicitly consolidated dispatch
and corrected loader error text (return annotations/docstrings excluded).

**Critic:** Initial decision REVISE: relocated descriptions retained informal
and scientifically inaccurate prose. Actor replaced all descriptions with
technical definitions, preserving keys/labels/order. Critic independently
reran 47 tests (4.42 s) and accepted. The historical CLI fraction, ER
probability, dtype, row-shuffle, and rank-roundoff issues are documented and
guarded by tests. Optional package metadata no longer prevents a smoke run.

**Decision:** ACCEPT. No scientific numerical correction was applied.

## 4. Plotting and result aggregation — ACCEPT

**Actor:** Split plotting into styles, input handling, comparison statistics,
tables, rendering, contours, 3D performance, sign fractions/normalization,
weight controls, radius summaries, and triads. Preserved all original function
exports and mutable legacy mode-filter behavior. Moved aggregation into its
own module and shared exact path/CV/grid-averaging duplicates. Retained the
distinct plotting and merge column policies. Repaired the optional
weight-performance exporter's pre-existing undefined `saved_paths` crash by
restoring its commented-out export call. Added working script/module merge
imports and a regular `util` package to prevent name shadowing.

**Validation:** 55 CPU tests passed. Actual contour/triptych and repaired
exporter images were generated. Merge script/module outputs agree exactly;
plotting CLI help is unchanged. All 23 PNG/CSV/LaTeX artifacts from eight
original plotting paths match byte-for-byte (50-DPI test exports). Hashes are
recorded in `baseline/plot_comparison.json`. Plot function ASTs are identical
apart from the explicit exporter repair and exact shared helper aliases.

**Critic:** Independently reran 55 tests (6.05 s), checked exports/backend/filter
compatibility, inspected the exporter repair, and verified all 23 artifact
bytes and hashes. Low follow-ups: correct merge-output and inferred-repeat
documentation; retain a reusable plotting comparison driver.

**Decision:** ACCEPT. No metric, grouping, contour, or statistical calculation
was changed; no archived figures were overwritten.
