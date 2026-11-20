# Matched hyperparameter win curves by control

Every figure compares empirical CE with the named control. Higher means CE wins more of the tested settings; NARMA uses lower error to decide a win. These are win-count curves, not grid-mean performance curves.

The full-grid plots use 24 configurations and the recorded pointwise paired-realization bootstrap intervals. Input-specific panels count 6 radius/leak settings per input; they show descriptive point estimates without intervals. The dotted line is half of the configurations.

[MC overview](overview_MC_all_controls.png) · [NARMA10 overview](overview_NARMA10_all_controls.png) · [IPC overview](overview_IPC_all_controls.png)

| Control | Full grid, all metrics | By input, all metrics |
|---|---|---|
| CE shuffled magnitudes | [PNG](ce_shuffled_full_grid.png) | [PNG](ce_shuffled_by_input.png) |
| CE degree rewired | [PNG](ce_degree_rewired_full_grid.png) | [PNG](ce_degree_rewired_by_input.png) |
| CE binary | [PNG](ce_binary_full_grid.png) | [PNG](ce_binary_by_input.png) |
| CE half-normal | [PNG](ce_halfnormal_full_grid.png) | [PNG](ce_halfnormal_by_input.png) |
| ER exact CE signed weights | [PNG](er_signed_empirical_full_grid.png) | [PNG](er_signed_empirical_by_input.png) |
| ER binary | [PNG](er_binary_full_grid.png) | [PNG](er_binary_by_input.png) |
| ER half-normal | [PNG](er_halfnormal_full_grid.png) | [PNG](er_halfnormal_by_input.png) |
| WS-like exact CE signed weights | [PNG](ws_signed_empirical_full_grid.png) | [PNG](ws_signed_empirical_by_input.png) |
| CE unknown edges removed | [PNG](ce_unknown_removed_full_grid.png) | [PNG](ce_unknown_removed_by_input.png) |

The six inhibitory fractions are observed points; line segments connect them and do not establish an exact crossover. The grid is fixed, and results do not extend automatically to untested settings. Every strict positive seed-mean advantage counts, including a small advantage at low absolute performance.

All existing source tables and figures were preserved. No simulations were run.
