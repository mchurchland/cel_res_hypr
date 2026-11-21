# AUC of mean-performance curves

These figures integrate each model's mean performance over inhibitory-edge
fraction, then divide by the observed E/I range. They retain the performance
metric's units. MC and IPC are better when higher; NARMA10 NMSE is better when
lower. The integral uses trapezoidal interpolation over the eleven sampled
fractions from 0 to 1. It is not a percentage of hyperparameter wins.

| Scaling | All tasks | By input scale |
|---|---|---|
| Own radius | [PNG](performance_auc_own_radius.png) | [PNG](performance_auc_by_input_own_radius.png) |
| Fixed reference | [PNG](performance_auc_fixed_reference.png) | [PNG](performance_auc_by_input_fixed_reference.png) |
| Shared CE multiplier | [PNG](performance_auc_shared_ce_scale.png) | [PNG](performance_auc_by_input_shared_ce_scale.png) |

Separate task figures, with the same task scale across normalization panels:

- [MC](performance_auc_MC.png)
- [NARMA10](performance_auc_NARMA10.png)
- [IPC](performance_auc_IPC.png)

All twelve models are included. Full-grid results average 24 equally weighted
settings within each realization; each input-specific result averages six
radius/leak settings. Each realization's mean curve is integrated before
averaging AUCs across the 100 paired realizations. Intervals resample
whole realizations jointly across fractions, models, inputs and tasks:
1000 draws, seed 20261119. No pointwise intervals were integrated.
Input heatmaps show point estimates; their intervals are recorded in the CSV.

IPC uses a symmetric-log display (linear between −1 and 1), retaining severe
negative held-out estimates and their influence on the mean. Input heatmaps use
the same colour normalization across scaling protocols within each metric.
No observations were removed, clipped, or replaced. Curves, weights and task
scores are unchanged. Statistical intervals are descriptive percentile intervals;
they do not establish that one model is significantly better than another.

The CSVs contain 540 mean-AUC estimates and 54,000 realization AUCs. All 108
full-grid point estimates and intervals reproduce the earlier performance-AUC
table; point estimates also match integration of the actual mean-curve table.
Uniform E/I weighting and linear interpolation between observations are explicit
analysis choices. Scores and AUCs have separate units across tasks.
