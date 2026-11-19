# Which hyperparameter settings does CE lose on?

A loss means the paired seed-averaged advantage is strictly negative: CE−ER for MC/IPC/KR, ER−CE test NMSE for NARMA. These are descriptive counts over fixed grid settings, not significance-filtered losses or independent biological samples. Exact losing tuples, including every E/I fraction and parameter, are in losing_configurations.csv.

The archived analysis includes both CE variants, the 200-realization sign sweep and the separate 100-realization normalization controls. The newer MC/NARMA run uses 100 realizations, 30 settings, fixed zero bias and fractions 0–50%. These grids and normalization experiments are never pooled.

## Input scale at 20% inhibition

| Dataset | CE variant | Scaling | Metric | Input | Losing settings |
|---|---|---|---|---:|---:|
| final_results_sign | cel_matched | spectral_radius | IPC | 0.1 | 4/24 (16.7%) |
| final_results_sign | cel_matched | spectral_radius | IPC | 0.5 | 11/24 (45.8%) |
| final_results_sign | cel_matched | spectral_radius | IPC | 1 | 10/24 (41.7%) |
| final_results_sign | cel_matched | spectral_radius | IPC | 1.5 | 8/24 (33.3%) |
| final_results_sign | cel_matched | spectral_radius | KR | 0.1 | 0/24 (0.0%) |
| final_results_sign | cel_matched | spectral_radius | KR | 0.5 | 0/24 (0.0%) |
| final_results_sign | cel_matched | spectral_radius | KR | 1 | 0/24 (0.0%) |
| final_results_sign | cel_matched | spectral_radius | KR | 1.5 | 0/24 (0.0%) |
| final_results_sign | cel_matched | spectral_radius | MC | 0.1 | 0/24 (0.0%) |
| final_results_sign | cel_matched | spectral_radius | MC | 0.5 | 6/24 (25.0%) |
| final_results_sign | cel_matched | spectral_radius | MC | 1 | 8/24 (33.3%) |
| final_results_sign | cel_matched | spectral_radius | MC | 1.5 | 9/24 (37.5%) |
| narma_271800 | cel_matched | original_radius | MC | 0.001 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | MC | 0.01 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | MC | 0.05 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | MC | 0.1 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | MC | 0.5 | 4/6 (66.7%) |
| narma_271800 | cel_matched | original_radius | NARMA10_test_nmse | 0.001 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | NARMA10_test_nmse | 0.01 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | NARMA10_test_nmse | 0.05 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | NARMA10_test_nmse | 0.1 | 2/6 (33.3%) |
| narma_271800 | cel_matched | original_radius | NARMA10_test_nmse | 0.5 | 4/6 (66.7%) |
| narma_271800 | cel_matched | spectral_radius | MC | 0.001 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | MC | 0.01 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | MC | 0.05 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | MC | 0.1 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | MC | 0.5 | 5/6 (83.3%) |
| narma_271800 | cel_matched | spectral_radius | NARMA10_test_nmse | 0.001 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | NARMA10_test_nmse | 0.01 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | NARMA10_test_nmse | 0.05 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | NARMA10_test_nmse | 0.1 | 0/6 (0.0%) |
| narma_271800 | cel_matched | spectral_radius | NARMA10_test_nmse | 0.5 | 5/6 (83.3%) |

## Main interpretation

- In the archived own-radius MC sweep, CE has no losses at input 0.1 at the four imbalanced fractions (0, 20, 80, 100%). At those fractions, losses concentrate at larger input and leak 1. IPC has exceptions even at input 0.1 and is not monotonically worse with input.
- At 40% and 60% inhibition in that archived sweep, CE loses on almost all MC/IPC settings regardless of input; all completed-sign CE KR settings lose there. Input alone cannot explain the overall pattern.
- In the newer own-radius run, all MC and NARMA losses at 0/10/20% occur at input 0.5; every tested input 0.001–0.1 wins. At 20%, five of six input-0.5 settings lose on both metrics. The sole winner is radius 0.6, leak 0.6.
- Under fixed-reference scaling, low target radius matters: at 20%, every radius-0.6 setting loses on both metrics, across all five inputs and both leaks. Thus the high-input-only explanation does not transfer across normalization protocols.
- At 40% and 50% in the newer own-radius run, ER wins every MC configuration. NARMA losses remain widespread but include exceptions, so conclusions should remain metric-specific.

## Scope and files

- hyperparameter_loss_summary.csv: separate loss percentages for each parameter level at each E/I fraction, within explicit E/I regions, and over each sampled sweep. E/I regions are exploratory summaries: imbalanced <=20% or >=80%; interior strictly between 20% and 80%. In the NARMA run these groups contain 0/10/20% and 30/40/50%, respectively.
- input_leak_loss_summary.csv: joint input/leak loss rates within each E/I region. Marginal rates average the other finite grid levels and do not isolate a biological mechanism.
- all_configuration_comparisons.csv and losing_configurations.csv: exact tuples. final_results sign KR is updated effective rank; normalization-control KR has archival/unverified protocol provenance and stays separate.
- Most_negative_advantages.csv lists up to five losses per metric, fraction, normalization and CE family, sorted within each group. Different metric magnitudes are not comparable.
- Binary-weight ER is the control in these sign/crossover experiments, not a CE-weight-matched ER. Do not generalize these contrasts to the weighted-control study.
- All NARMA test errors are retained. A large CE test error at 40%, input 0.01, radius 1.05, leak 0.6 influences the corresponding mean comparison. Existing pointwise paired intervals are retained where available; no new significance threshold is applied.
- No new simulations, source edits, outlier filtering, PDFs or overwritten previous analyses.
