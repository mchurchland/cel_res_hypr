> The authoritative recommendation is in [RECOMMENDATION.md](RECOMMENDATION.md), which includes the longer-test follow-up. This document describes only the initial 500-test-step pass.

# Local MC-horizon pilot

200 CPU conditions: 8 model variants, 5 low-input settings,
5 paired archived realizations; delays 1–400.
Original model matrices, input projections, streams, train/validation/test split,
ridge grid and float64 protocol are retained. No original results were modified.
NARMA10 was not rerun. States and predictions are cached for follow-up checks.

## Horizon sensitivity

The smallest tested horizon whose worst case/model **seed-mean** estimated
remaining memory is at most 0.1 MC, through delay 400, is **350**.
This is an operational pilot tolerance, not proof of an infinite-horizon capacity.
Inspect final-lag profiles and individual-realization variation before adopting it.

| Horizon | Worst case/model mean excess remaining |
|---|---:|
| 30 | 43.829177 |
| 60 | 14.215500 |
| 80 | 5.065606 |
| 100 | 2.753717 |
| 120 | 1.665685 |
| 150 | 0.705143 |
| 200 | 0.417582 |
| 250 | 0.308896 |
| 300 | 0.230480 |
| 350 | 0.095717 |
| 400 | 0.000000 |

## Interpretation and limitations

- `MC` columns are ordinary sums of squared correlations. Null subtraction is
  a separate horizon diagnostic; it does not silently redefine production MC.
- For each fitted readout, 400 lags are compared with
  held-out target-row permutations. The null keeps predictions and training fixed.
  It measures chance association, not between-network uncertainty or a refitted null.
- Finite-test positive correlations can keep the raw cumulative sum increasing
  after recoverable memory has decayed. Extra raw area alone is not extra memory.
- The practical per-realization flag additionally requires the preceding 25-lag
  block to lie below its permutation-null 95th percentile. It is a descriptive
  screening rule, not a multiple-comparison-adjusted equivalence test.
- The recommended screen uses averages over the sampled seeds; individual
  conditions can differ. This is not the full 24-setting, 11-fraction, 100-seed grid.
- Max target-extension prefix error: 1.76e-12.
  Max CPU/archive prefix error: 6.61e-11.
  Above-unit-radius trajectories may diverge numerically between CPU and GPU;
  such discrepancies are recorded rather than hidden.

## Outputs

- [Delay profiles](memory_delay_profiles.png)
- [Remaining-memory diagnostic](remaining_memory_by_horizon.png)
- [Horizon summaries](horizon_summary.csv)
- [Per-realization diagnostics](horizon_by_realization.csv)
- [Late-delay scores](final_25_lags.csv)
- [CPU/archive parity](parity_and_runtime.csv)

Settings: `balanced_peak` and `imbalanced_peak`: radius .95, leak 1, input .01,
E/I .5 and .2. `slow_leak`: E/I .5, radius .95, leak .6, input .01.
`high_radius` and `high_radius_slow_leak`: E/I .5, radius 1.05,
leak 1 and .6 respectively; input .01. These emphasize candidate long-memory regimes.
