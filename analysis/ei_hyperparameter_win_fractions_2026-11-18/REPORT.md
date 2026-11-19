# CE versus ER: fraction of winning hyperparameter configurations

Primary statistic: for each inhibitory-edge fraction and each of the 96 identical hyperparameter settings, average CE−ER performance over paired seeds, then count the settings with a strictly positive mean difference. Higher is treated as better for MC, IPC and KR; greater KR denotes separation, not universally better computation.

All plots use a 0–100% y scale. Counts weight the finite grid equally; they are not probabilities under a natural distribution of hyperparameters. Connecting lines guide the eye between sampled fractions, with no inferred crossover precision.

## Main sign sweep

| CE variant | Inhibitory edges | MC | IPC | KR |
|---|---:|---:|---:|---:|
| CE: unknown signs completed | 0% | 88.5% (85/96) | 61.5% (59/96) | 100.0% (96/96) |
| CE: unknown signs completed | 20% | 76.0% (73/96) | 65.6% (63/96) | 100.0% (96/96) |
| CE: unknown signs completed | 40% | 5.2% (5/96) | 5.2% (5/96) | 0.0% (0/96) |
| CE: unknown signs completed | 60% | 5.2% (5/96) | 5.2% (5/96) | 0.0% (0/96) |
| CE: unknown signs completed | 80% | 76.0% (73/96) | 60.4% (58/96) | 100.0% (96/96) |
| CE: unknown signs completed | 100% | 90.6% (87/96) | 71.9% (69/96) | 100.0% (96/96) |
| CE: unknown-sign edges removed | 0% | 91.7% (88/96) | 80.2% (77/96) | 100.0% (96/96) |
| CE: unknown-sign edges removed | 20% | 68.8% (66/96) | 55.2% (53/96) | 96.9% (93/96) |
| CE: unknown-sign edges removed | 40% | 4.2% (4/96) | 5.2% (5/96) | 0.0% (0/96) |
| CE: unknown-sign edges removed | 60% | 4.2% (4/96) | 5.2% (5/96) | 0.0% (0/96) |
| CE: unknown-sign edges removed | 80% | 52.1% (50/96) | 42.7% (41/96) | 97.9% (94/96) |
| CE: unknown-sign edges removed | 100% | 90.6% (87/96) | 69.8% (67/96) | 100.0% (96/96) |

## Definitions, pairing and uncertainty

- Main sign sweep: 200 paired computational realizations; updated effective-rank KR. Normalization control experiment: 100 paired realizations, analyzed separately. Archival KR normalization results have different/unverified historical rank provenance and have their own figure.
- Match exactly on seed, inhibitory-edge fraction, normalization, spectral-radius target, leak rate, input scale and bias. Each seed carries its full grid throughout bootstrap resampling. These seeds are computational realizations, not independent biological animals.
- Grid: radius [0.6, 0.8, 0.95, 1.05], leak [0.6, 0.8, 1], input [0.1, 0.5, 1, 1.5], bias [0, 0.1]. The input-stratified plot uses 24 settings per curve point.
- Shared fractions are 0, 20, 40, 60, 80 and 100%. CE-only reference fractions near 24.25% are excluded because ER has no observation there; no interpolation or nearest-fraction matching. Exact unmatched counts appear in pairing_audit.csv.
- Strict CE > ER; exact ties count as non-wins. No significance filter, rounding, performance floor threshold, trimming or outlier clipping. A tiny positive difference counts equally to a large one, so win percentages should be read alongside absolute performance/effect sizes.
- Shading is a pointwise percentile interval from 2,000 paired whole-seed bootstrap resamples of the complete mean-surface calculation. Configurations are fixed, not independently resampled. The hard greater-than-zero decision makes estimates discrete and sensitive to near ties; bootstrap intervals are exploratory, not simultaneous discovery tests.
- Figure 05 instead computes a win fraction within each seed before averaging. This measures how often individual realizations win, and generally differs from the percentage of settings whose mean performance favors CE.
- Original-radius controls fix each family's reference multiplier across the sign sweep for a given target radius. This is not the newer shared-CE multiplier protocol. Sign-sweep ER is the archived binary-weight ER control, not a weight-distribution-matched control; topology and weights both differ.
- Inhibitory percentage refers to assigned edge signs, not an inferred biological inhibitory-neuron fraction. No new simulations were run and no existing result or figure was overwritten.

## Files

- 01_ce_winning_hyperparameter_fraction.png: primary requested plot.
- 02_ce_winning_fraction_by_input.png: condition on input scale.
- 03_normalization_controls_mc_ipc.png and 04_normalization_controls_kr_archival.png: separate normalization-control experiment.
- 05_mean_per_realization_win_fraction.png: alternative order of averaging.
- win_fraction_summary.csv, configuration_comparisons.csv, per_realization_win_fractions.csv and pairing_audit.csv contain numerical outputs. manifest.json records source hashes and analysis settings.
