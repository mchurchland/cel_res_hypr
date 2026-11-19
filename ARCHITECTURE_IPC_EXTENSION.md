# Upper-half E/I extension with IPC

From the Thesis repository root on Orca:

```bash
sbatch r_architecture_mc_narma_ipc_50_100.sbatch
```

The setup job queues the existing GPU preflight, GPU array and dependent
merge/analysis. Simulation jobs each request one GPU; there is no CPU simulation
array. The setup job itself only prepares and submits the GPU jobs.

The extension uses all ten models, all three normalization protocols, the same
24 radius/leak/input settings, zero bias, and 100 paired realizations with the
same base seed as run 271846. Fractions are **50%, 60%, 70%, 80%, 90%, 100%**.
The 50% point overlaps the existing run; 60–100% are the new fractions. By default, preflight checks the overlap against `results/architecture_mc_narma/architecture_mc_narma_271846`: raw matrices/input projections must match, and stable-condition MC/NARMA scores must agree within 1e-6. Set `PARENT_RUN_DIR` if that run is stored elsewhere. Input
streams and model component seeds remain paired. There are 432,000 condition
rows, each with MC, conventional NARMA10 and IPC estimates. No STP is added.

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
conditions cover the first, middle and last requested fractions (50%, 80%, 100%
by default). Failed preflight prevents the production array from starting.

## Results and figures

Results:

```
results/architecture_mc_narma_ipc_50_100/architecture_upper_ipc_<submission-job-ID>/
```

The merged CSV keeps the existing filename `architecture_mc_narma.ALL.csv` and
contains the additional IPC columns. `shards/` contains completed per-condition
files, and `preflight/` contains the GPU checks.

Analysis (PNG, CSV, Markdown):

```
analysis/architecture_mc_narma_ipc_50_100/architecture_upper_ipc_<submission-job-ID>/
```

The `duel_plots_by_model/` subfolder provides separate plots for all nine controls, with MC/NARMA/IPC and input-specific panels. IPC gets the same mean curves, matched-setting win curves, input-conditioned
comparisons and hyperparameter variance decomposition as MC/NARMA. Win fractions
remain the primary comparison of breadth across the fixed grid.

**IPC covers this requested upper half only.** The old 0–40% data do not gain
IPC automatically. Full 0–100% IPC curves require computing those lower-half
trajectories too; do not fill the missing values or substitute archived IPC
scores with different definitions. If wanted later, reuse the launcher with
a direct helper invocation with `--with-ipc --fractions 0 .1 .2 .3 .4` and a new `RUN_ID`/`OUTROOT` (the upper-half wrapper requires its 50% overlap). That invocation also recomputes
MC/NARMA under the same protocol and preserves the old results.

The preflight 50% overlap check validates representative stable conditions before launching; the complete overlap should still be checked before joining old and new MC/NARMA sweeps.
Neither the raw matrices nor the MC/NARMA definitions changed, but numerical
chaotic trajectories can depend on hardware/library versions. Keep the recorded
source hashes, environment, input seeds and raw-network hashes when checking it.

## Adjustable settings

Edit the defaults near the top of the sbatch file, or supply environment values:

```bash
TOTAL_REPEATS=20 ARRAY_TASKS=10 sbatch r_architecture_mc_narma_ipc_50_100.sbatch
```

`IPC_MAX_DELAY` and `IPC_MAX_DEGREE` control basis size; changing them changes
the IPC estimand, so use a new run ID. There is a 5,000-target guard against an
accidental combinatorial expansion. The default 285-target basis adds readout
work; the GPU array requests 12 hours per task, which is a scheduling limit,
not a measured runtime guarantee.

Resume with the same RUN_ID/settings. Existing shards are hash-checked and
reused. Existing runs, plots and snapshots are preserved. No new dependency
installation is performed inside jobs.
