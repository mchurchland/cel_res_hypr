# Unified architecture sweep

Protocol: `architecture-mc-narma-ipc-v3-delayed-legendre`. 950,400 paired observations; 100 realizations; 24 grid settings.

The primary comparison is CE empirical versus each control at identical hyperparameters. MC is the sum of held-out squared correlations over delays 1–30; larger is better. NARMA10 NMSE is test MSE divided by test-target variance; smaller is better. Readout penalties use separate validation data. Mean curves average every grid setting; winning fractions compare realization-averaged scores at each configuration. They are different summaries.

Intervals resample whole paired realizations jointly across models, fractions and configurations (1000 draws, seed 20261119); they are pointwise, descriptive intervals. The fixed grid is not a random hyperparameter population. One-realization intervals are undefined.

`hyperparameter_variance_components.csv` contains exact balanced-grid main effects and interactions within each realization. The allocated-share plot splits each interaction equally across its participating parameters; that allocation is descriptive and depends on the grid and score scale. Constant surfaces have undefined percentage shares.

Inspect `configuration_advantages.csv` for the settings driving any crossover and `dynamics_diagnostics.csv` for initialization sensitivity, actual scaled radii, memory beyond the primary cutoff, and large errors. No poor-performing configurations were discarded. NARMA generator-only nonfinite streams were replaced by the prespecified recorded seed rule before network evaluation; `narma_streams.csv` records counts and accepted seeds, and task caches contain rejected seed lists.

Own-radius scaling equalizes current raw radius. Fixed-reference uses each family's radius at the known-edge CE reference fraction, held fixed across the sweep. Shared-CE scaling applies the CE multiplier at the current fraction to all families; it varies across fractions. Only the shared multiplier preserves exact signed-weight matching after scaling. The removed-edge CE control has fewer edges and is not density-matched to the full-CE ER controls. WS is a directed WS-like control. All edge signs are randomized quotas; these are reservoir controls, not inferred biological sign balances.

The protocol uses conventional NARMA10 and a new stabilized readout; it must not be silently pooled with archived NARMA results. Performance differences do not alone isolate eigenmode causality or establish biological optimality.

IPC is the signed sum of held-out 1 - MSE/target-second-moment estimates for 285 orthonormal Legendre-product targets, lags 1–10, total degree 1–3. Cross-delay products are included. Higher is better; finite estimates may be negative and are retained. This is a finite-basis, regularized adaptation of Dambre et al. (2012), not full population IPC. IPC_positive_part is a separately named diagnostic with a positive clipping floor; it is not used for the primary win curves. Linear/nonlinear, cross-delay and mean-predictor diagnostics are included. IPC covers every fraction in this run with the same definition; do not substitute archived restricted IPC scores.

Only WS binary and WS half-normal were newly simulated (158,400 rows). 792,000 existing comparator rows were loaded from /home/milesc/Thesis/results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871/architecture_mc_narma.ALL.csv. The paired analysis covers all 12 families; the new run's merged CSV contains only newly simulated rows. Parent task streams, WS topology/sign masks and half-normal magnitude draws are reused exactly. The current IPC estimator/readout are unchanged, including the known possibility of severe held-out errors after activation-regime changes. No failing rows were removed or clipped.

Primary comparison: fraction of identical hyperparameter settings won at each E/I fraction, after averaging each setting over paired realizations. Grid-averaged performance is supporting context.
