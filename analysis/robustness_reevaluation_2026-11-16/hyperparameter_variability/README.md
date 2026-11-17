# Variability attributable to the hyperparameter grid

The analysis quantifies the spread shown in the E/I dot plots, separately within every E/I level, architecture family, metric and normalization. It uses the original performance values, never jittered display coordinates.

## Main connectome result

Allocated share of within-grid variance, summarized across E/I levels (%):

| Parameter | MC | IPC | KR |
| --- | ---: | ---: | ---: |
| Spectral radius | 5.7 | 6.5 | 48.7 |
| Leak rate | 10.7 | 18.0 | 29.7 |
| Input scale | 82.7 | 68.7 | 19.9 |
| Bias range | 0.8 | 6.8 | 1.7 |

These percentages split every interaction equally among its participating parameters, so the four shares sum to 100%. This is an explicit allocation convention; a joint effect is not uniquely caused by one parameter. Across E/I levels, component variances are summed and divided by summed total within-level variance. Each E/I level has equal design weight; levels with more grid variance contribute more to the resulting percentage. Variation between E/I levels is excluded.

## What each quantity means

- **Main-effect share:** variance of performance averaged over the other three parameters. It captures an effect that survives averaging; cancellation can hide conditional effects.
- **Total involvement share:** variance from the main effect plus every interaction involving that parameter. These shares overlap and can sum to more than 100%.
- **Allocated share:** main effect plus an equal split of each interaction among its participants; the four shares sum to 100%.
- **Matched-sweep RMS SD:** root mean variance when changing only that parameter, holding the other three fixed, averaged over their settings. It is in the original performance units.
- **Mean matched-sweep range:** mean max−min performance when changing just that parameter with the others fixed. It depends on extremes.

## Design and uncertainty

The grid is 4 spectral-radius targets × 3 leak rates × 4 input scales × 2 bias ranges. Levels are equally weighted as discrete choices. An exact factorial decomposition retains all 15 main/interaction components and verifies that they sum to total variance. Results apply to these sampled ranges; they are not intrinsic parameter rankings independent of the chosen grid. Log/linear coordinate distances are relevant to the earlier local-sensitivity analysis, whereas this variance analysis weights the tested levels equally.

The primary estimates decompose the seed-averaged surface, matching the dot plots. 1000 seed-cluster bootstrap replicates recompute the mean surface and its complete decomposition. Shared numerical seeds are resampled together across E/I levels, metrics and normalization modes with the same seed set; 95% percentile intervals are retained. No configuration is treated as an independent network sample. Per-seed decomposition is also exported as a distinct estimand. There is one observation per seed/configuration, so those per-seed effects can still include interaction with evaluation noise.

Updated sign-only results (200 seeds) and archival normalization controls (100 seeds) remain separate. The requested E/I percentage denotes negative edges, not inhibitory neurons. All cached mean surfaces were checked against the exact plotted means, and their source hashes were checked against the earlier manifests. No new simulations were run.

## Files

- `cel_removed_hyperparameter_variability_table.png`: table for C. elegans with unknown signs removed.
- `matched_er_hyperparameter_variability_table.png`: table for matched ER.

- `cel_matched_hyperparameter_variability_table.png`: standalone table of the main connectome variance shares.

- `variance_allocation_summary.png`: comparisons across architecture families and metrics.
- `cel_matched_sign_variability_by_ei.png`: main versus total involvement for the main connectome at each E/I level; companion PNGs cover other families and normalization modes.
- `tables/by_ei.csv`: main/total/allocated shares and matched-sweep SD/range with bootstrap intervals.
- `tables/across_ei_summary.csv`: variance-weighted summaries and intervals.
- `tables/components.csv`: every main/interaction variance component at each E/I level.
- `tables/per_seed.csv.gz`: robustness calculated within each individual seed before averaging.
- `manifest.json`: data/cache provenance, code hashes and analysis settings.

Reproduce with `python tools/quantify_hyperparameter_variability.py --out-dir analysis/new_variability`. All figures are PNGs.
