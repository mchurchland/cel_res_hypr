# Mean performance, near-floor frequency and CV

Fraction below one is calculated within each seeded 96-configuration surface, using only finite scores and the strict comparison P < 1. The percentage is then correlated with that surface's mean and CV. All correlations are Spearman rank correlations. Constant percentages produce undefined correlations, not zero correlations.

| Scope | Metric | Mean–CV ρ | % below 1–CV ρ | % below 1–mean ρ | Grids with any score <1 | Scores <1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pooled main | IPC | 0.629 | undefined | undefined | 0/2000 | 0/192000 |
| Pooled main | KR | 0.963 | undefined | undefined | 0/2000 | 0/192000 |
| Pooled main | MC | 0.874 | -0.074 | -0.141 | 89/2000 | 93/192000 |
| C. elegans only | IPC | -0.444 | undefined | undefined | 0/200 | 0/19200 |
| C. elegans only | KR | 0.296 | undefined | undefined | 0/200 | 0/19200 |
| C. elegans only | MC | -0.391 | 0.117 | -0.119 | 6/200 | 6/19200 |

The cutoff is an absolute score of 1 in each metric's own units. It is not an equivalent distance from a performance floor across MC, IPC and KR. In the main experiment it is sparse for MC and absent for IPC/KR. These particular coefficients therefore cannot establish or exclude a general floor mechanism. Correlation with CV is also not independent of its mean denominator.

Intervals use 2,000 seed-cluster bootstrap replicates with RNG seed 20261117. Pooled panels retain every architecture sharing a resampled seed as one cluster. All architecture-specific comparisons are also exported. Repeated observations are computational realizations of one anatomical connectome, not independent worms. Only PNG figures are generated.
