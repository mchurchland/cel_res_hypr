# Integrated CE–ER performance across the E/I sweep

AUC integrates each mean-performance curve over inhibitory fraction f from 0 to 1. Thus AUC is also the average performance under a uniform distribution of f, with linear interpolation between the six sampled fractions. Integrating a percentage axis 0–100 would multiply every area by 100 without changing comparisons.

Mean performance averages all 96 hyperparameter settings within each seed. The paired bootstrap resamples whole seeds (5,000 draws), retaining all settings and fractions. CE-only reference fractions near 24.25% remain excluded. Existing analysis files are unchanged.

| CE variant | Metric | CE AUC | ER AUC | CE−ER AUC [95% interval] | Relative difference |
|---|---|---:|---:|---:|---:|
| CE: unknown signs completed | MC | 5.609 | 7.094 | -1.485 [-1.519, -1.452] | -20.9% |
| CE: unknown signs completed | IPC | 9.155 | 11.671 | -2.516 [-2.567, -2.468] | -21.6% |
| CE: unknown signs completed | KR | 7.149 | 10.757 | -3.608 [-3.747, -3.471] | -33.5% |
| CE: unknown-sign edges removed | MC | 5.195 | 7.094 | -1.900 [-1.933, -1.867] | -26.8% |
| CE: unknown-sign edges removed | IPC | 8.723 | 11.671 | -2.948 [-3.003, -2.892] | -25.3% |
| CE: unknown-sign edges removed | KR | 6.014 | 10.757 | -4.743 [-4.840, -4.646] | -44.1% |

Positive and negative areas split the area between the mean curves at interpolated crossings. Their difference equals the signed AUC difference. They describe the mean curve, not the mean of per-seed positive/negative areas.

This is an overall performance summary, not a standalone robustness measure. It weights the interval 0–100% uniformly, which is an explicit modeling choice and not an empirical distribution of biological conditions. Trapezoidal integration gives each endpoint half the weight of an interior sampled fraction. The unmeasured intervals, including 40–60%, are approximated linearly; the true continuous-sweep AUC is not directly observed.

The comparison uses archived binary-weight ER, own-radius normalization, and the specified 96-point hyperparameter grid. Do not generalize it to weight-matched controls or different normalization protocols. Compare AUCs within each metric only; MC, IPC and KR have different scales. Larger KR measures separation rather than generic computational superiority.

These AUCs concern the mean-performance curves, not the earlier percentage-of-winning-configurations curves.
