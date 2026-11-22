# What changes when memory is measured through 200 delays?

The longer horizon strengthens the evidence against a general computational
advantage of empirical CE wiring in this reservoir protocol. It also strengthens
the ER gamma result and changes the interpretation of upper-tail capping.
Average performance, matched-setting wins and the highest observed score remain
different outcomes; the plots retain all three.

The analysis contains **100 paired realizations, 11 inhibitory-edge fractions
(0–100% in 10% steps), and 24 radius/leak/input settings**, with bias fixed at zero.
There are 12 architecture variants and 10 ER weight conditions. New MC estimates
sum held-out squared correlations over delays 1–200 using 5,000 test samples.
NARMA10 results are the archived paired scores, using their original 500 test
samples; NARMA10 was not rerun. All intervals below are descriptive 95% percentile
intervals from 1,000 whole-realization paired bootstrap resamples.

## 1. The 30-delay cutoff hid more memory in several controls than in CE

These comparisons use **the same new 5,000-sample test segment** at both horizons.
The AUC is the integral of grid-mean performance across E/I, divided by the sweep
width; its units remain MC. It is not the AUC of a win curve.

| Own-radius model | MC30 mean-performance AUC | MC200 AUC | Additional measured memory |
|---|---:|---:|---:|
| CE empirical | 11.156 | 11.394 | 0.238 |
| CE shuffled magnitudes | 13.228 | 14.001 | 0.774 |
| ER empirical CE weights | 14.655 | 16.488 | 1.833 |
| ER binary | 13.105 | 14.927 | 1.822 |
| WS-like empirical CE weights | 15.410 | 17.862 | 2.452 |
| WS-like binary | 14.805 | 17.296 | 2.491 |
| WS-like half-normal | 15.197 | 17.907 | 2.710 |

At 50% inhibition, radius 0.95, leak 1 and input 0.01, the memory curves show
substantial recall beyond delay 30 in several controls. Extending the target
range therefore matters for interpreting both peak memory and architecture
rankings. The archived MC30 values differ slightly from the new MC30 values
because the test segment also became longer; those changes are tabulated
separately and are not attributed to the horizon.

[Horizon comparison](architecture/horizon_dependence.png) ·
[Recall at each delay](memory_delay_profiles.png) ·
[Does the cutoff change winning regions?](horizon_win_comparison.png)

## 2. CE's advantage depends strongly on which control is used

Under **own-radius scaling**, empirical CE wins none of the **264** tested
E/I–hyperparameter cells against ER with CE's exact signed-weight multiset,
WS-like with that multiset, WS-like binary, or WS-like half-normal. Here a win
compares the 100-realization mean at the same cell: this does **not** assert
that every individual realization loses.

CE still wins some comparisons against the weaker ER binary baseline, particularly
under low input and E/I imbalance. At 20% inhibition, CE wins **6/24** settings
overall against ER binary, including **4/6** radius/leak settings at input 0.01.
At the same fraction, it wins **0/24** against ER empirical CE weights and
ER half-normal. Fractions of settings won are relative to a named control,
not an intrinsic robustness score.

| Own-radius comparison | CE MC200 win AUC | CE NARMA10 win AUC |
|---|---:|---:|
| ER binary | 21.0% | 36.7% |
| ER half-normal | 3.1% | 15.6% |
| ER empirical CE weights | 0.0% | 9.0% |
| WS-like empirical CE weights | 0.0% | 10.2% |
| CE shuffled magnitudes | 0.0% | 1.5% |

These percentages integrate the win-fraction curve over E/I, giving endpoints
half the weight of interior observations. They are not flat percentages of
all rows or percentages of random seeds.

The aggregate MC disadvantage persists with shared-CE scaling, where the
empirical CE, ER and WS-like matrices receive the same multiplier at each
fraction. ER empirical weights exceed CE by **1.294 MC AUC units
[1.212, 1.383]**, and WS-like empirical weights exceed CE by **1.860
[1.772, 1.950]**. The corresponding gaps under own-radius scaling are larger:
**5.094 [4.993, 5.188]** and **6.467 [6.346, 6.588]**. Thus normalization changes
the size and conditional pattern of the comparison, but does not explain away
the aggregate disadvantage. These ensemble interventions do not identify one
specific topological motif or eigenmode mechanism.

[CE versus the three ER controls](key_win_curves.png) ·
[All control comparisons](architecture/winning_fraction_MC200_own_radius.png) ·
[Performance AUC](key_performance_auc.png)

## 3. Moment-matched gamma remains a consistent, modest improvement on ER

Under own-radius scaling, gamma reaches **16.808** MC200 mean-performance AUC,
compared with **16.488** for empirical CE magnitudes: a paired gain of
**0.320 [0.296, 0.345]**, approximately **1.94%**. Gamma beats empirical
magnitudes in **264/264 seed-mean cells**. Its gain is between **0.013 and
2.255 MC units** across those cells; **257/264** pointwise paired intervals
exclude zero in gamma's favor. These are exploratory, unadjusted intervals,
not a guarantee of improvement in every realization or an untested setting.

The archived NARMA10 error AUC is **0.15009** for gamma versus **0.15144** for
empirical weights. The paired NMSE reduction is **0.001352
[0.000851, 0.002110]**, approximately **0.89%**; gamma's NARMA10 win AUC is
**82.1%**, rather than 100%.

The MC200 gain also remains positive under shared scaling:
**0.152 [0.141, 0.163]**. For empirical, gamma and lognormal, the shared and
Frobenius branches coincide by construction and must not be treated as
independent replications.

Gamma and lognormal match the empirical finite-vector mean and variance, yet
lognormal performs worse: empirical minus lognormal MC200 AUC is
**0.491 [0.452, 0.529]** under own-radius scaling. This supports a role for
distributional properties beyond those moments. It does not establish that a
particular tail statistic, minimum magnitude, or spectral feature is the cause.
The synthetic distributions use fixed mid-quantile vectors placed by empirical
rank with paired tie-breaking, not iid draws from the fitted distributions.

[Gamma gains with paired intervals](gamma_gain_own_radius.png) ·
[All ER performance AUCs](er/performance_auc_MC200.png) ·
[All scaling branches](er/gamma_vs_empirical.png)

## 4. The upper-tail conclusion changes with the MC horizon

All three upper-tail caps now improve aggregate **MC200** under own-radius
scaling, even though the stronger caps worsened the archived MC30 aggregate.
Caps restore mean magnitude afterward; they alter more than the maximum alone.

| ER modification | MC200 AUC | Gain over empirical [95% paired interval] |
|---|---:|---:|
| Empirical | 16.488 | Reference |
| Q99 cap | 16.613 | 0.125 [0.093, 0.158] |
| Q95 cap | 16.716 | 0.228 [0.177, 0.283] |
| Q90 cap | 16.579 | 0.091 [0.031, 0.157] |
| Moment-matched gamma | 16.808 | 0.320 [0.296, 0.345] |

Better aggregate MC200 does not imply winning most settings: empirical still
has win AUCs of **44.2%, 50.0% and 50.8%** against Q99, Q95 and Q90 respectively.
The magnitude of an improvement and the number of positive differences are
different summaries.

NARMA10 does not show the same cap pattern. Q95 and Q90 increase error by
**0.00316** and **0.00666** on average across E/I. Q99's small error change is
uncertain. The strongest empirical weights are therefore not required for
the observed MC200 average, and tail effects depend on task and scaling.

## 5. Broad grid performance and the best observed setting favor different models

WS-like half-normal has the largest own-radius **mean** MC200 AUC, **17.907**,
just above WS-like empirical weights at **17.862**. Their paired difference is
**0.045 [-0.058, 0.151]**: this is not evidence of a clearly established
winner between those two.

The largest observed seed-mean MC200 is instead **70.88** for WS-like binary
at 40% inhibition, radius 0.95, leak 1, input 0.01. ER binary reaches **67.62**,
ER empirical weights **52.31**, ER gamma **54.47**, and empirical CE **24.13**
at their respective best observed cells. These maxima are **descriptive,
test-selected grid maxima**, not independently validated tuned performance.

Consequently, the results support improved performance across much of this
fixed grid for some weight distributions. They do not support a universal
claim that those distributions maximize achievable memory.

[Best observed MC200 curves](best_observed_MC200.png) ·
[Mean-performance comparison](key_performance_auc.png)

## 6. Hyperparameter contributions are task- and architecture-dependent

The table allocates exact within-grid ANOVA variance, splitting interactions
equally across their participating parameters. Raw component variances are
summed over realizations and E/I fractions before calculating percentages.
The full CSV also records an equal-surface weighting, which is a different
summary. These allocations depend on the grid and score scale.

| Own-radius model / task | Radius | Leak | Input |
|---|---:|---:|---:|
| CE empirical / MC200 | 9.0% | 11.9% | 79.0% |
| ER empirical / MC200 | 16.2% | 16.1% | 67.7% |
| ER gamma / MC200 | 16.7% | 16.5% | 66.8% |
| WS-like half-normal / MC200 | 20.1% | 17.5% | 62.3% |
| CE empirical / NARMA10 | 16.4% | 28.0% | 55.6% |
| ER empirical / NARMA10 | 26.5% | 41.9% | 31.6% |
| ER gamma / NARMA10 | 23.3% | 49.6% | 27.1% |

[Architecture variance allocation](architecture/hyperparameter_variance_own_radius.png) ·
[ER weight variance allocation](er/hyperparameter_variance_own_radius.png)

## Limits and reproducibility

- MC200 is still a finite-horizon estimate. The largest seed-mean recall summed
  over delays 176–200 is **0.0848**, in WS-like binary. Most such tail blocks
  are much smaller, but this does not prove zero memory after delay 200.
  The exploratory 0.05 late-block screen is not a formal null test.
- All results concern this fixed grid, static tanh dynamics, all-node input,
  imposed sign quotas and two tasks. They do not establish that a biological
  connectome is generally a poor computational system or is biologically
  optimized for the imposed inhibitory fraction.
- Pointwise strict wins include tiny differences. Use effect sizes, absolute
  performance and paired uncertainty alongside the win percentages. No setting
  was removed because it performed poorly.
- The removed-edge CE model has fewer edges; the WS-like generator is the
  existing directed control, not a newly substituted standard WS graph.
- Multi-normalization performance-AUC figures use separate horizontal limits
  in each panel to show within-branch differences clearly. Compare the numeric
  axes when assessing differences between branches.
- No simulations were launched for this analysis. Existing results, analyses
  and poster figures were preserved. Six new analysis tests and six existing
  AUC tests passed; curves/AUCs, aliases, cumulative horizons and win/tie
  partitions were independently checked.

[All PNGs](PNGS.md) · [Detailed methods and tables](REPORT.md) ·
[Validation](validation.json) · [Sources and settings](manifest.json)
