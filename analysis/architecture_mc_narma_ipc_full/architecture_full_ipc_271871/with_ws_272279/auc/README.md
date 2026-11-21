# Area under the matched-setting win curves

The primary AUC integrates CE's fraction of matched hyperparameter settings won
across the observed inhibitory-edge sweep, divided by the sweep width. Values
run from 0 to 100%. Higher favors CE against the named control; 50% is half of
the integrated fixed-grid/EI domain. Strict wins are determined after averaging
performance across paired realizations. Lower NARMA10 NMSE determines a win.
Exact ties count separately. This is not ROC AUC.

[All controls and normalization protocols](auc_win_fraction_overview.png).

| Scaling | Full grid | By input scale |
|---|---|---|
| Own radius | [PNG](auc_win_fraction_own_radius.png) | [PNG](auc_by_input_own_radius.png) |
| Fixed reference | [PNG](auc_win_fraction_fixed_reference.png) | [PNG](auc_by_input_fixed_reference.png) |
| Shared CE multiplier | [PNG](auc_win_fraction_shared_ce_scale.png) | [PNG](auc_by_input_shared_ce_scale.png) |

The grid has 24 settings and
100 paired realizations; 11 observed E/I fractions
cover 0–100% inhibition.
Trapezoidal integration assumes linear interpolation between observed curve
points and gives endpoints half the weight of equally spaced interior points.
Uniform weighting of E/I and this finite grid is an analysis choice, not an
empirical biological distribution. These AUCs measure breadth of relative wins;
small advantages at low absolute performance count equally to large advantages.

Intervals use 1000 whole-realization bootstrap draws, seed 20261119.
The same resampling counts are shared across models, metrics, fractions and
settings. Each draw recomputes mean performance, strict wins and the complete
curve's AUC. Pointwise curve intervals are not integrated. These are descriptive
percentile intervals, with no multiple-comparison adjustment. The by-input
plots show point estimates; their intervals are in the CSV.

[Supporting mean-performance AUC](auc_mean_performance.png) summarizes score
magnitude on separate MC, NARMA10 and IPC axes. Its intervals resample whole
realizations. It is a distinct estimand; NARMA10 error AUC is better when lower.
The IPC axis uses symmetric-log scaling (linear between −1 and 1) to retain
extreme negative held-out estimates. No IPC estimates were removed or clipped.
All 11 full-grid control comparisons were checked against the
existing plotted win-fraction tables for every task and normalization protocol.

## Additional performance-curve AUC figures

[All nine performance-curve AUC figures and tables](PERFORMANCE_AUC.md): individual tasks, normalization protocols, and input-scale breakdowns.
