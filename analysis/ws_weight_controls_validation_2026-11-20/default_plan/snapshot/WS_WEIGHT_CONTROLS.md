# Paired binary and half-normal WS controls

Submit from the Thesis repository root on Orca:

```bash
sbatch r_ws_binary_halfnormal.sbatch
```

Only **WS binary** and **WS half-normal** are simulated. Defaults match completed
run 271871: 100 paired realizations, 0–100% inhibitory edges in 10% steps,
24 radius/leak/input settings, zero bias, three normalization protocols, and
MC, NARMA10 and finite-basis IPC. This adds **158,400 rows** and 316,800 reservoir
trajectories; IPC reuses the MC trajectory. Existing ten-model simulations are
not repeated. Preflight uses a few reference evaluations for validation.

The new models use the exact cached directed WS-like topology and sign mask
from run 271871, including its fixed rewiring probability .1. WS binary uses
magnitude one. WS half-normal reuses the exact absolute-Gaussian magnitude draws
already used by the CE/ER half-normal controls, with the existing WS parent-weight
permutation. Binary and half-normal signed multisets are checked against their
matching parent controls at every requested fraction and at the fixed-reference
fraction. Raw-matrix input projections, task streams and CE scaling references
come from validated parent caches.

Own-radius scaling uses each new model's own radius. Fixed-reference scaling
uses its radius at the parent's known-edge reference fraction. Shared-CE scaling
uses the parent's CE multiplier at the current fraction. Comparisons therefore
retain the same meanings as the completed sweep.

The current readouts and IPC definition are unchanged so new and old controls
remain comparable. The previously observed IPC regime-switching/generalization
failure mode remains possible. Raw signed estimates are retained; adding these
models does not resolve that issue or silently apply a different regularizer.

The completed parent run must remain accessible, including its manifest,
merged CSV and checksum, and network/task caches. Override `PARENT_RUN_DIR`
if stored elsewhere. Incompatible grids, tasks, IPC bases or source hashes fail
before submission. `TOTAL_REPEATS` may be reduced to use the first paired
realizations; the analysis uses that same subset of the parent.

```bash
TOTAL_REPEATS=20 ARRAY_TASKS=10 sbatch r_ws_binary_halfnormal.sbatch
```

GPU preflight gates the GPU simulation array; merge and analysis depend on the
whole array succeeding. Simulation tasks each request one GPU. Defaults retain
20 array tasks and the existing 12-hour per-task limit, not a measured runtime
prediction. No packages are installed inside jobs.

New raw results:

```
results/ws_binary_halfnormal/ws_binary_halfnormal_<submission-job-ID>/
```

The merged `architecture_mc_narma.ALL.csv` contains only the two new families.
Automatic analysis loads the matching existing parent rows as well, producing
comparisons and PNG duel curves for all twelve families without rerunning the
old models or modifying their results. Its report/manifest record both sources.

Analysis:

```
analysis/ws_binary_halfnormal/ws_binary_halfnormal_<submission-job-ID>/
```

The parent dataset is filtered to the same fractions, grid, normalizations and
realization IDs. Parent and new family names must be disjoint, and every
configuration must be complete and unique. Mean performance is supporting
context; the primary comparison remains fraction of matched settings won.

Use a new run ID for changed settings. Resume unchanged runs with the same
RUN_ID. Existing snapshots and outputs are preserved; already queued jobs retain
their frozen source snapshots.
