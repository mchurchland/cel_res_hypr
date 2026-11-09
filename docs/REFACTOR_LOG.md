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
