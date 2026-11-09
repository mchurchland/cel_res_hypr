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
