# Updated results including WS binary and WS half-normal

The complete 12-model analysis is in **[with_ws_272279](with_ws_272279/REPORT.md)**.
The previous ten-model analysis remains in this directory unchanged.

The combined dataset contains 950,400 observations: 100 paired realizations,
24 settings, 11 inhibitory-edge fractions, three normalization protocols, and
MC, NARMA10 and IPC. Both original experiments and all signed IPC estimates
are preserved. No simulations were rerun.

| Figures | Link |
|---|---|
| AUC of matched-setting win curves, all scaling protocols | [PNG](with_ws_272279/auc/auc_win_fraction_overview.png) |
| All AUC plots, input-scale breakdowns and tables | [AUC index](with_ws_272279/auc/README.md) |
| Performance-curve AUC by task, normalization and input scale | [Performance AUC index](with_ws_272279/auc/PERFORMANCE_AUC.md) |
| MC matched-setting duel overview | [PNG](with_ws_272279/duel_plots_by_model/overview_MC_all_controls.png) |
| NARMA10 matched-setting duel overview | [PNG](with_ws_272279/duel_plots_by_model/overview_NARMA10_all_controls.png) |
| IPC matched-setting duel overview | [PNG](with_ws_272279/duel_plots_by_model/overview_IPC_all_controls.png) |
| Every model, full grid and by input | [Figure index](with_ws_272279/duel_plots_by_model/README.md) |
| Mean MC, own radius | [PNG](with_ws_272279/mean_MC_own_radius.png) |
| Mean NARMA10, own radius | [PNG](with_ws_272279/mean_NARMA10_test_nmse_own_radius.png) |
| Mean IPC, own radius, readable axis and full-range inset | [PNG](with_ws_272279/mean_IPC_own_radius.png) |
| Hyperparameter contributions, own radius | [PNG](with_ws_272279/variance_contributions_own_radius.png) |

All three scaling protocols have corresponding mean, duel, input-advantage and
hyperparameter-contribution plots. The primary duel outcome counts settings
whose paired realization-mean score favors empirical CE. Higher win fractions
favor CE; NARMA uses lower NMSE to determine a win.

The raw combined CSV and its provenance are stored together inside the original
results folder:

`results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871/with_ws_272279/`

Means, configuration advantages and win fractions were checked against every
raw observation. The cached 1,000 paired-realization bootstrap intervals were
reused, and the figures were regenerated. The known IPC generalization failures
remain present; no adverse observations were removed or clipped.
