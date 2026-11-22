# MC200 reanalysis

[Interpretation and main findings](FINDINGS.md) · [PNG gallery](PNGS.md)

The MC200 run contains 1,742,400 logical rows, 100 paired realizations, 11 inhibitory-edge fractions, and 24 hyperparameter settings. Architecture comparisons include 12 variants; ER weight comparisons include 10 distributions. NARMA10 is joined from the archived, identically paired runs and was not rerun.

## Mean-performance AUC rankings

AUC integrates the grid-mean curve over the observed 0–100% sweep, divided by its width. It retains performance units. Win AUC instead integrates the percentage of settings won after averaging paired realization scores. They answer different questions.

### Architecture: own-radius scaling

| Model | MC200 AUC [95% CI] | NARMA10 error AUC [95% CI] |
|---|---:|---:|
| WS-like half-normal | 17.9071 [17.8527, 17.9576] | 0.1532 [0.1492, 0.1576] |
| WS-like exact CE signed weights | 17.8617 [17.7574, 17.9640] | 0.1477 [0.1438, 0.1519] |
| WS-like binary | 17.2961 [17.2485, 17.3447] | 0.1614 [0.1575, 0.1659] |
| ER exact CE signed weights | 16.4880 [16.4152, 16.5613] | 0.1514 [0.1475, 0.1559] |
| ER half-normal | 16.1541 [16.0977, 16.2136] | 0.1626 [0.1586, 0.1671] |
| ER binary | 14.9266 [14.8814, 14.9757] | 0.1814 [0.1772, 0.1863] |
| CE shuffled magnitudes | 14.0013 [13.9446, 14.0575] | 0.1615 [0.1575, 0.1663] |
| CE degree rewired | 13.9834 [13.8995, 14.0643] | 0.1655 [0.1613, 0.1702] |
| CE half-normal | 13.5869 [13.5319, 13.6442] | 0.1778 [0.1738, 0.1827] |
| CE binary | 12.7632 [12.7074, 12.8212] | 0.2040 [0.1971, 0.2125] |
| CE empirical | 11.3943 [11.3294, 11.4577] | 0.1962 [0.1922, 0.2014] |
| CE unknown edges removed | 10.0660 [10.0055, 10.1258] | 0.2365 [0.2313, 0.2419] |

[Mean curves and input effects](architecture/performance_by_input_own_radius.png) · [Performance AUC](architecture/performance_auc_MC200.png) · [Win AUC](architecture/win_fraction_auc_MC200.png)

### Architecture: matched-setting wins

| Comparator | Reference MC200 win AUC | Reference NARMA10 win AUC |
|---|---:|---:|
| CE shuffled magnitudes | 0.0% [0.0, 0.0] | 1.5% [0.8, 3.1] |
| CE degree rewired | 0.4% [0.2, 0.6] | 9.4% [6.5, 10.0] |
| CE binary | 42.7% [40.4, 47.1] | 54.6% [51.0, 56.7] |
| CE half-normal | 16.7% [15.0, 17.9] | 28.1% [25.6, 30.2] |
| ER exact CE signed weights | 0.0% [0.0, 0.0] | 9.0% [7.7, 10.6] |
| ER binary | 21.0% [20.0, 22.9] | 36.7% [35.2, 38.8] |
| ER half-normal | 3.1% [2.1, 3.1] | 15.6% [14.6, 18.1] |
| WS-like exact CE signed weights | 0.0% [0.0, 0.0] | 10.2% [8.1, 11.9] |
| CE unknown edges removed | 100.0% [99.6, 100.0] | 93.1% [91.2, 94.0] |
| WS-like binary | 0.0% [0.0, 0.0] | 15.6% [14.8, 18.1] |
| WS-like half-normal | 0.0% [0.0, 0.0] | 11.7% [10.8, 13.5] |

Reference = CE empirical. Every strict positive difference counts, even a very small one. The tables also record the fraction of settings with a positive lower bootstrap interval; this is a descriptive uncertainty screen, not a multiplicity-adjusted significance claim.

### Er: own-radius scaling

| Model | MC200 AUC [95% CI] | NARMA10 error AUC [95% CI] |
|---|---:|---:|
| Moment-matched gamma | 16.8078 [16.7360, 16.8785] | 0.1501 [0.1461, 0.1545] |
| Empirical capped at Q95 | 16.7163 [16.6574, 16.7733] | 0.1546 [0.1505, 0.1595] |
| Empirical capped at Q99 | 16.6129 [16.5461, 16.6807] | 0.1513 [0.1473, 0.1558] |
| Empirical capped at Q90 | 16.5789 [16.5248, 16.6319] | 0.1581 [0.1543, 0.1626] |
| Empirical CE magnitudes | 16.4880 [16.4152, 16.5613] | 0.1514 [0.1475, 0.1559] |
| Half-normal | 16.1541 [16.0977, 16.2136] | 0.1626 [0.1586, 0.1671] |
| Heterogeneity a = 0.75 | 16.0511 [15.9933, 16.1088] | 0.1579 [0.1539, 0.1626] |
| Moment-matched lognormal | 15.9972 [15.9164, 16.0812] | 0.1541 [0.1500, 0.1587] |
| Heterogeneity a = 0.50 | 15.6376 [15.5794, 15.6953] | 0.1663 [0.1621, 0.1710] |
| Binary | 14.9266 [14.8814, 14.9757] | 0.1814 [0.1772, 0.1863] |

[Mean curves and input effects](er/performance_by_input_own_radius.png) · [Performance AUC](er/performance_auc_MC200.png) · [Win AUC](er/win_fraction_auc_MC200.png)

### Er: matched-setting wins

| Comparator | Reference MC200 win AUC | Reference NARMA10 win AUC |
|---|---:|---:|
| Binary | 70.0% [70.0, 70.0] | 67.5% [65.8, 70.0] |
| Half-normal | 54.6% [52.9, 56.2] | 60.4% [58.3, 62.9] |
| Empirical capped at Q99 | 44.2% [40.8, 47.1] | 50.6% [47.7, 54.8] |
| Empirical capped at Q95 | 50.0% [49.2, 50.8] | 55.0% [51.7, 58.3] |
| Empirical capped at Q90 | 50.8% [50.8, 51.7] | 55.8% [55.0, 61.2] |
| Moment-matched gamma | 0.0% [0.0, 1.2] | 17.9% [15.4, 22.5] |
| Moment-matched lognormal | 100.0% [98.3, 100.0] | 84.6% [79.6, 86.7] |
| Heterogeneity a = 0.50 | 62.1% [60.0, 63.3] | 62.9% [61.3, 66.2] |
| Heterogeneity a = 0.75 | 52.1% [50.8, 54.2] | 57.9% [55.4, 62.9] |

Reference = empirical CE magnitudes placed on ER. Every strict positive difference counts, even a very small one. The tables also record the fraction of settings with a positive lower bootstrap interval; this is a descriptive uncertainty screen, not a multiplicity-adjusted significance claim.

## What extending the MC horizon changes

All new horizons (30, 60, 100, 150, 200) use the same 5,000-sample test segment. Comparing them isolates the effect of summing additional delays within this estimator. The archived 30/60-delay scores used only 500 test samples. Their differences from the new 30/60 sums also reflect test length and sampling variation. New readouts keep the original training, validation, projections, matrices, and exact scaling factors.

| Suite / model | Archived MC30 AUC | New MC30 AUC | MC200 AUC | Rank at new MC30 → MC200 |
|---|---:|---:|---:|---:|
| architecture: CE empirical | 11.191 | 11.156 | 11.394 | 11 → 11 |
| architecture: CE shuffled magnitudes | 13.261 | 13.228 | 14.001 | 6 → 7 |
| architecture: CE degree rewired | 13.234 | 13.199 | 13.983 | 7 → 8 |
| architecture: CE binary | 11.966 | 11.935 | 12.763 | 10 → 10 |
| architecture: CE half-normal | 12.719 | 12.689 | 13.587 | 9 → 9 |
| architecture: ER exact CE signed weights | 14.689 | 14.655 | 16.488 | 4 → 4 |
| architecture: ER binary | 13.139 | 13.105 | 14.927 | 8 → 6 |
| architecture: ER half-normal | 14.060 | 14.025 | 16.154 | 5 → 5 |
| architecture: WS-like exact CE signed weights | 15.447 | 15.410 | 17.862 | 1 → 2 |
| architecture: CE unknown edges removed | 9.968 | 9.933 | 10.066 | 12 → 12 |
| architecture: WS-like binary | 14.842 | 14.805 | 17.296 | 3 → 3 |
| architecture: WS-like half-normal | 15.231 | 15.197 | 17.907 | 2 → 1 |
| er: Empirical CE magnitudes | 14.689 | 14.655 | 16.488 | 3 → 5 |
| er: Binary | 13.139 | 13.105 | 14.927 | 10 → 10 |
| er: Half-normal | 14.060 | 14.025 | 16.154 | 8 → 6 |
| er: Empirical capped at Q99 | 14.711 | 14.675 | 16.613 | 2 → 3 |
| er: Empirical capped at Q95 | 14.613 | 14.579 | 16.716 | 4 → 2 |
| er: Empirical capped at Q90 | 14.390 | 14.356 | 16.579 | 6 → 4 |
| er: Moment-matched gamma | 14.837 | 14.803 | 16.808 | 1 → 1 |
| er: Moment-matched lognormal | 14.405 | 14.371 | 15.997 | 5 → 8 |
| er: Heterogeneity a = 0.50 | 13.788 | 13.755 | 15.638 | 9 → 9 |
| er: Heterogeneity a = 0.75 | 14.240 | 14.206 | 16.051 | 7 → 7 |

[Architecture horizon comparison](architecture/horizon_dependence.png) · [ER distribution horizon comparison](er/horizon_dependence.png) · [Selected memory curves](memory_delay_profiles.png)

## Hyperparameter dependence

Exact balanced-grid functional ANOVA decomposes performance variance within each realization, family, scaling branch and E/I fraction. The main plots pool raw component variances across those surfaces, then split interactions equally among their participating parameters. CSV tables additionally retain the equal-surface average of allocated percentage shares; it is a different weighting. These allocations depend on the selected parameter levels and performance scale and are descriptive, not causal effects.

[Architecture contributions](architecture/hyperparameter_variance_own_radius.png) · [ER contributions](er/hyperparameter_variance_own_radius.png)

## Horizon and protocol limitations

- MC200 remains a finite-delay estimate. Late-delay diagnostics are retained across the full grid; the selected profiles emphasize low-input long-memory regimes.
- `horizon_tail_diagnostics.csv` records recall summed over delays 176–200. A seed-mean block score above 0.05 is an exploratory screen, not a validated null threshold or proof that the horizon is inadequate.
- Squared correlations have positive finite-test chance contributions. No null subtraction or clipping was applied to the reported MC sums.
- Intervals resample whole paired realizations jointly across all fractions and configurations: 1000 bootstrap replicates, seed 20261122. Hyperparameter settings are a fixed design, not independent biological samples. Intervals are pointwise and unadjusted for multiple comparisons.
- Own-radius normalization changes each matrix multiplier. Shared-CE uses the same multiplier at a given fraction, and Frobenius matching equalizes a matrix norm. None alone establishes an isolated eigenmode mechanism.
- The ER gamma/lognormal vectors match realized mean and variance and preserve the rank placement of empirical ER weights (with paired tie-breaking); they are fixed mid-quantile vectors, not iid random samples. Mean/variance matching also matches the raw Frobenius norm for an equal edge count.
- Inhibitory-edge fractions are imposed sign quotas, not estimates of biological functional E/I balance. The removed-edge CE control has fewer edges; directed WS construction follows the existing WS-like generator.
- No reservoir was discarded for poor performance. Grid-mean results, conditional input comparisons and win fractions should be interpreted together. No simulations were launched for this analysis.

## Files and reproducibility

Each suite directory contains PNGs and CSVs for mean curves, per-setting advantages, win curves, both AUC definitions, input dependence, horizon dependence and ANOVA components. The top-level manifest records checksums, filters, pairing keys and bootstrap settings. Cached cubes retain all joined scalar scores. Original data, prior analyses and poster figures are preserved.

