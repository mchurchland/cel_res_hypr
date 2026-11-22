# Local test of the MC delay horizon

**Use 300 delays as a conservative common horizon for the next MC evaluation.** A 200-delay cutoff is a reasonable lower-cost option under the pilot's 0.1-MC mean-residual tolerance. **100 delays is insufficient for the slower-leak conditions.** These are practical recommendations from a limited pilot, not a proven universal cutoff.

![Memory decay and remaining-score diagnostics](horizon_recommendation.png)

## What was run

The first pass tested **200 conditions**: eight model variants, five low-input settings and five paired archived realizations. It evaluated delays 1–400 with the original 500-step held-out test set. Model variants were CE empirical, CE magnitude-shuffled, ER binary, ER empirical magnitudes, ER gamma, WS binary, WS empirical magnitudes and WS half-normal.

The follow-up used **85 conditions**, covering 17 selected model/setting pairs and the same five realizations. It evaluated delays **1–500 with 5,000 held-out test steps**. Selection covered the 12 pairs with the largest initial residual after delay 100, plus five balanced-peak comparisons. This selection deliberately stresses truncation; it is not a random sample of the entire experiment.

The low-input cases used input 0.01: E/I .5 and .2 at radius .95/leak 1; E/I .5 at radius .95/leak .6; and E/I .5 at radius 1.05 with leaks 1 and .6. The pilot does not cover the complete 24-setting, 11-fraction, 100-realization grid.

Networks, E/I signs, input projections, training/validation streams, readout regularization selection and float64 dynamics were retained. Extending the test stream leaves its original prefix unchanged. Only MC was evaluated; NARMA10 was not rerun. The production simulator and sklearn ridge readout were reused.

## How much remains after each cutoff?

For each model/setting, average the remaining score over five paired realizations. The table gives the **largest such mean** among the longer-test pairs. The remaining interval is from the cutoff + 1 **through delay 500**, with the test-target permutation-null mean subtracted. These are estimated excess scores, not confidence bounds or established infinite-horizon memory. The maximum across individual realizations is also shown.

| Cutoff | Largest model/setting mean excess remaining | Largest individual excess remaining |
|---|---:|---:|
| 60 | 13.924 | 15.997 |
| 100 | 2.462 | 8.083 |
| 150 | 0.384 | 1.865 |
| 200 | 0.095 | 0.350 |
| 250 | 0.072 | 0.296 |
| 300 | 0.062 | 0.286 |
| 400 | 0.030 | 0.159 |

At 200 delays the worst sampled model/setting mean residual is about 0.095 MC; at 300 it is about 0.062. The 300-delay choice adds a margin beyond the first tested cutoff meeting the 0.1-MC mean-residual screen. It is not the exact minimum necessary for every model or realization. Individual cases retain more estimated excess, and very weak late correlations remain difficult to distinguish from finite-sample chance association. The 0.1 tolerance is an exploratory analysis choice, not a literature standard.

## Why the longer test set matters

Squared sample correlations have a positive chance baseline. Summing more delays therefore keeps adding some score even when meaningful recall has decayed. The first, shorter-test pass suggested 350 delays under the same mean-excess screen; the longer-test follow-up reduces the first passing cutoff to 200. This sensitivity is why a cutoff should not be chosen from raw cumulative score alone.

The null randomly permutes held-out target rows jointly across delays while keeping predictions fixed (199 permutations in the first pass; 99 in the follow-up). It is a conditional association diagnostic. It does not refit the readout, estimate between-network uncertainty, prove no signal beyond delay 500, or supply a formal equivalence test. The null-adjusted scores are kept separate from ordinary MC; production scores were not redefined.

## What to do in the full experiment

1. Use a **common 300-delay primary score** across models if the additional readout cost is acceptable. Retain shorter-horizon summaries (30, 60, 100, 150 and 200) from the same fitted lag curves as sensitivity checks.
2. Retain diagnostic lag curves to 500 on a subset of the full grid, especially the slow-leak, low-input conditions. Extend further only if those profiles show a consequential tail. No finite horizon is guaranteed by this pilot.
3. Use a longer held-out MC test stream when resolving small late-memory differences. The pilot used 5,000 steps; this has not been optimized as a minimum sample length. The original 1,500/500/500 training/validation/test protocol remains documented and was not overwritten.
4. Recompute matched-setting winning fractions at the longer common horizon before treating the current MC30 fractions as total-memory comparisons. This pilot does not update the full-grid win curves or establish their new rankings.

Changing the delay horizon requires refitting additional readouts and, where states were not retained, regenerating MC trajectories. It does not require rerunning NARMA10. Current poster files and original result sets remain unchanged.

## Validation and files

All 200 initial conditions reproduce the archived 1–60 delay scores to within 6.7e-11. In all 85 longer-test checks, the old state and score prefixes are unchanged at the recorded precision. Six synthetic tests cover persistent memory at the measurement boundary, decaying memory, pure-null curves, positive raw noise accumulation, invalid inputs and the seeded permutation null.

- [Full initial lag profiles](memory_delay_profiles.png)
- [Longer-test profiles and horizon diagnostics](long_test_validation/long_test_horizon_check.png)
- [Summary table](recommendation_table.csv)
- [Longer-test condition summaries](long_test_validation/horizon_summary.csv)
- [Longer-test per-realization results](long_test_validation/horizon_by_realization.csv)
- [Validation receipt](VALIDATION.json) and [unit-test log](unit_tests.log)
- [Initial settings](settings.json) and [longer-test settings](long_test_validation/settings.json)

Frozen scoring source copies and per-condition caches retain provenance. The longer-test process completed all 85 cases before its first summary step was interrupted. Summaries were rebuilt from those saved outputs after removing repeated decompression from the analysis loop; the scoring function was verified unchanged.
