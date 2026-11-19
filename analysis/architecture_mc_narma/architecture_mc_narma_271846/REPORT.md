# Unified architecture sweep

Protocol: `architecture-mc-narma-v2-conventional`. 432,000 paired observations; 100 realizations; 24 grid settings.

The primary comparison is CE empirical versus each control at identical hyperparameters. MC is the sum of held-out squared correlations over delays 1–30; larger is better. NARMA10 NMSE is test MSE divided by test-target variance; smaller is better. Readout penalties use separate validation data. Mean curves average every grid setting; winning fractions compare realization-averaged scores at each configuration. They are different summaries.

Intervals resample whole paired realizations jointly across models, fractions and configurations (1000 draws, seed 20261119); they are pointwise, descriptive intervals. The fixed grid is not a random hyperparameter population. One-realization intervals are undefined.

`hyperparameter_variance_components.csv` contains exact balanced-grid main effects and interactions within each realization. The allocated-share plot splits each interaction equally across its participating parameters; that allocation is descriptive and depends on the grid and score scale. Constant surfaces have undefined percentage shares.

Inspect `configuration_advantages.csv` for the settings driving any crossover and `dynamics_diagnostics.csv` for initialization sensitivity, actual scaled radii, memory beyond the primary cutoff, and large errors. No poor-performing configurations were discarded. NARMA generator-only nonfinite streams were replaced by the prespecified recorded seed rule before network evaluation; `narma_streams.csv` records counts and accepted seeds, and task caches contain rejected seed lists.

Own-radius scaling equalizes current raw radius. Fixed-reference uses each family's radius at the known-edge CE reference fraction, held fixed across the sweep. Shared-CE scaling applies the CE multiplier at the current fraction to all families; it varies across fractions. Only the shared multiplier preserves exact signed-weight matching after scaling. The removed-edge CE control has fewer edges and is not density-matched to the full-CE ER controls. WS is a directed WS-like control. All edge signs are randomized quotas; these are reservoir controls, not inferred biological sign balances.

The protocol uses conventional NARMA10 and a new stabilized readout; it must not be silently pooled with archived NARMA results. Performance differences do not alone isolate eigenmode causality or establish biological optimality.
