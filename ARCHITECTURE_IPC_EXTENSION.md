# Full E/I sweep: upper-half extension and lower-half IPC backfill

From the Thesis repository root on Orca:

```bash
sbatch r_architecture_mc_narma_ipc_full.sbatch
```

The setup job queues the existing GPU preflight, GPU array and dependent
merge/analysis. Simulation jobs each request one GPU; there is no CPU simulation
array. The setup job itself only prepares and submits the GPU jobs.

The sweep uses all ten models, all three normalization protocols, the same
24 radius/leak/input settings, zero bias, and 100 paired realizations with the
same base seed as run 271846. Fractions are **0%, 10%, 20%, …, 100%**.

| Fraction | MC and NARMA | IPC |
|---|---|---|
| 0–50% | Reuse existing scores from run 271846 | Compute from regenerated MC trajectories |
| 60–100% | Compute new scores | Compute from those MC trajectories |

**50% is already covered for MC/NARMA and is not rerun in the production
array.** A few 50% conditions are recalculated during GPU preflight to verify
compatibility. Existing raw matrices, input projections, scaling factors and
task streams are validated before reusing parent scores. IPC requires the
lower-half MC state trajectories, which were not saved in the earlier results;
those trajectories are regenerated, but MC readout fitting and NARMA simulation
are skipped for the reused fractions.

The default merged dataset has **792,000 rows**: 432,000 rows with reused
MC/NARMA scores and new IPC, plus 360,000 rows with all-new scores. Production
uses 792,000 MC trajectories and 360,000 NARMA trajectories, rather than
rerunning both tasks over the entire grid. Each reused row records its parent
experiment ID and parent-shard checksum. Source files are not changed.

By default the parent is
`results/architecture_mc_narma/architecture_mc_narma_271846`.
Set `PARENT_RUN_DIR` if it is stored elsewhere. The parent manifest, shards and
task/network caches must remain available. The configuration and model/task/data
source hashes must match. Paired input streams and model component seeds are
retained. Validated parent input/target streams are loaded from its caches for
all fractions, preserving the exact previously accepted sequences. No STP is added.

The old filename `r_architecture_mc_narma_ipc_50_100.sbatch` remains a compatibility
entry point and now launches this same full-range sweep.

MC/NARMA definitions, lengths and readouts are unchanged. IPC reuses the MC
trajectory, so it does not require another reservoir simulation. New results
have an explicit IPC protocol identifier and their own frozen source snapshot.
The new IPC measure is **not** interchangeable with the archived restricted
single-delay, odd-order IPC score.

## What IPC means in this run

The finite basis contains **all 285 nonconstant products** of normalized
Legendre polynomials with total degree 1–3 across input lags 1–10. This includes
cross-delay products, rather than only functions of one delayed input. It
excludes the instantaneous input and degrees/delays outside the configured
basis. Input is iid Uniform[-1,1]. SciPy supplies polynomial evaluation and
scikit-learn supplies the SVD Ridge readout.

For each target z and held-out prediction zhat, report

```
capacity_estimate = 1 - mean((z - zhat)**2) / mean(z**2)
IPC = sum(capacity_estimate over the configured basis)
```

This follows the normalized reconstruction-error formulation of
[Dambre et al. (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3400147/), adapted
to held-out evaluation and validation-selected ridge regression with an
intercept. It is a **finite-basis regularized estimate**, not a claim to full
population IPC. Finite held-out estimates can be negative. Negative estimates
are retained in `IPC`; they are not silently clipped or thresholded.

Outputs include degree-1/linear and nonlinear sums, each degree separately,
single-delay and cross-delay sums, the training-mean predictor baseline and
excess over that baseline. `IPC_positive_part` is a separately named diagnostic
that clips negative terms; its positive noise floor is not the primary outcome.
The theoretical nonnegative population capacity and dimension bound should not
be applied uncritically to this finite held-out estimator. Basis specification,
readout settings and provenance are recorded in the manifest.

Preflight tests the basis, timing alignment and known synthetic capacities,
and includes IPC in the actual CPU/GPU scoring comparison. Its representative
conditions cover the first, middle and last requested fractions (0%, 50%, 100%
by default). Failed preflight prevents the production array from starting.

## Results and figures

Results:

```
results/architecture_mc_narma_ipc_full/architecture_full_ipc_<submission-job-ID>/
```

The merged CSV keeps the existing filename `architecture_mc_narma.ALL.csv` and
contains the additional IPC columns. `shards/` contains completed per-condition
files, and `preflight/` contains the GPU checks.

Analysis (PNG, CSV, Markdown):

```
analysis/architecture_mc_narma_ipc_full/architecture_full_ipc_<submission-job-ID>/
```

The `duel_plots_by_model/` subfolder provides separate plots for all nine controls, with MC/NARMA/IPC and input-specific panels. IPC gets the same mean curves, matched-setting win curves, input-conditioned
comparisons and hyperparameter variance decomposition as MC/NARMA. Win fractions
remain the primary comparison of breadth across the fixed grid.

**All three metrics now cover 0–100% in the new merged dataset.** The lower-half
MC/NARMA values are inherited exactly from validated parent shards; their
provenance is explicit in `MC_NARMA_reused`, `MC_NARMA_source_experiment_id`
and `MC_NARMA_parent_shard_sha256`. IPC uses the same configured basis for
every fraction. Do not substitute archived restricted IPC scores.

Merge streams validated shards to disk instead of keeping the full CSV in
memory. Analysis and per-model duel plots include the full E/I domain. The
parent cache comparison and preflight overlap checks guard against mixing
incompatible model/task protocols. Numerical trajectories can still depend on
hardware/library versions; the raw matrices and task streams must match, and
stable-condition overlap scores must agree within 1e-6 before the array starts.

## Adjustable settings

Edit the defaults near the top of the sbatch file, or supply environment values:

```bash
TOTAL_REPEATS=20 ARRAY_TASKS=10 sbatch r_architecture_mc_narma_ipc_full.sbatch
```

`IPC_MAX_DELAY` and `IPC_MAX_DEGREE` control basis size; changing them changes
the IPC estimand, so use a new run ID. There is a 5,000-target guard against an
accidental combinatorial expansion. The default 285-target basis adds readout
work; the GPU array requests 12 hours per task, which is a scheduling limit,
not a measured runtime guarantee.

Resume with the same RUN_ID/settings. Existing shards are hash-checked and
reused. Existing runs, plots and snapshots are preserved. No new dependency
installation is performed inside jobs.
