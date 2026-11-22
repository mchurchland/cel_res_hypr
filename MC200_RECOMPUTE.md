# MC200 extension

Run on Orca from the repository root: `sbatch r_mc200.sbatch`.
The controller submits a GPU preflight, 40 one-GPU workers, and a dependent merge.
Orca has fewer than 40 physical GPU nodes; workers may share a node's separate GPUs.

All 12 architecture variants and 10 ER distribution variants, all original scaling
branches, 100 paired realizations, 11 E/I fractions (0–100% in 10% steps), and
24 hyperparameter combinations are included. Exactly identical baseline/scaling
conditions share a newly computed curve after bitwise matrix, projection and
scaling checks. No NARMA or IPC simulation/readout is performed.

The primary MC score sums squared held-out Pearson correlations for delays 1–200.
The held-out segment increases from 500 to 5,000 time steps to reduce accumulated
finite-test chance correlation. Washout 500, train 1,500, validation 500, ridge grid,
float64 recurrence and all original input prefixes remain unchanged. Readouts use
the existing audited sklearn Ridge implementation, selecting each lag's penalty
only on validation data. The horizon pilot is in
`analysis/mc_horizon_local_pilot_2026-11-22/long_test_validation/`.
MC200 is a finite horizon estimate, not a demonstrated infinite-delay capacity.

Results: `results/mc200_recompute/mc200_<controller-job-id>/`.
Compressed NumPy checkpoints retain all 200 lag scores and chosen penalty indices;
`mc200.ALL.csv.gz` contains scalar sums at horizons 30, 60, 100, 150, and 200.
The old and new sums use different test lengths and must be labeled accordingly.
Metadata preserves source shard hashes, raw/input hashes, aliases and frozen code.
No states are saved. Checkpoint checksums make interrupted jobs resumable.
The original 500-sample test prefix is compared with archived 60-lag curves;
numerical trajectory differences in unstable regimes are recorded, not filtered.

The three archived parent runs must retain `manifest.json`, `cache/`, and `shards/`
on Orca until this run has finished. Other archived output can be removed after a
verified local backup. Keep local NARMA/IPC results for joining by suite/family,
realization, fraction, normalization and hyperparameters.
