# AUC of the fraction-of-settings-won curves

This integrates the percentage of hyperparameter configurations whose mean performance favors CE, not the mean-performance curve. The ordinate is percent settings won; the inhibitory fraction axis runs 0–1. A value of 50% means CE wins half the integrated configuration/E/I domain under this specified weighting.

| CE variant | Metric | CE win AUC [95% interval] | ER win AUC |
|---|---|---:|---:|
| CE: unknown signs completed | MC | 50.42% [49.17, 51.46] | 49.58% |
| CE: unknown signs completed | IPC | 40.62% [38.13, 42.81] | 59.38% |
| CE: unknown signs completed | KR | 60.00% [60.00, 60.00] | 40.00% |
| CE: unknown-sign edges removed | MC | 44.06% [42.19, 45.21] | 55.94% |
| CE: unknown-sign edges removed | IPC | 36.67% [34.58, 39.06] | 63.33% |
| CE: unknown-sign edges removed | KR | 58.96% [58.33, 59.38] | 41.04% |

The CSV additionally includes each input scale and the separate own-radius/fixed-reference normalization-control runs. Archival KR controls retain their distinct provenance.

Each bootstrap resamples whole paired seeds across every fraction and configuration, recomputes mean performance, counts strict wins, then integrates. Intervals are exploratory percentile intervals over 2,000 draws. Exact ties are tracked separately and excluded from both win counts. No main-sweep mean-surface ties occurred.

Linear interpolation connects the six sampled points at 0, 20, 40, 60, 80 and 100%. Trapezoidal integration weights endpoints half as much as interior samples. This approximates the displayed curve; it does not observe unmeasured intermediate fractions. The uniform E/I weighting and equally weighted finite hyperparameter grid are analysis choices, not an empirical distribution of biological conditions.

All positive differences count equally, including tiny improvements between poor reservoirs. These areas measure breadth of relative wins, not effect size, absolute computational usefulness or a biological robustness probability. The earlier performance-AUC results answer a different question and are preserved.
