# MC delay-horizon audit — 22 November 2026

The primary score sums squared held-out reconstruction correlations over delays 1–30, so its mathematical maximum is 30. Stored `MC_extended` sums the same diagnostic curve over delays 1–60. These are measurement horizons, not an established network memory limit. No simulations or original result files were modified.

## Best observed setting

The WS binary setting at 50% inhibitory edges, radius 0.95, leak 1, and input 0.01 has seed-mean MC30 = 29.999822 and MC60 = 59.500782. Its mean delay-60 reconstruction score is 0.904080. It therefore retains substantial memory beyond both the primary horizon and the end of the extended diagnostic. Differences between near-30 primary scores are unsuitable evidence for ranking total memory capability.

## Integrated mean scores

Values below average the 24 settings within each realization and integrate uniformly over E/I (trapezoidal area divided by E/I span). They are point estimates; this audit does not recompute bootstrap uncertainty or matched-setting win curves.

### architectures

| Model | MC30 AUC | MC60 AUC | Extra score at delays 31–60 |
|---|---:|---:|---:|
| ws_signed_empirical | 15.447131 | 17.569211 | 2.122080 |
| ws_halfnormal | 15.231451 | 17.462175 | 2.230724 |
| ws_binary | 14.841734 | 16.855020 | 2.013286 |
| er_signed_empirical | 14.689000 | 16.302462 | 1.613462 |
| er_halfnormal | 14.060364 | 15.841660 | 1.781296 |
| er_binary | 13.139087 | 14.611681 | 1.472593 |
| ce_shuffled | 13.260532 | 13.982609 | 0.722077 |
| ce_degree_rewired | 13.234242 | 13.968521 | 0.734279 |
| ce_halfnormal | 12.718525 | 13.541218 | 0.822693 |
| ce_binary | 11.965604 | 12.720902 | 0.755299 |
| ce_empirical | 11.191188 | 11.434873 | 0.243685 |
| ce_unknown_removed | 9.967762 | 10.112850 | 0.145088 |

### ER distributions

| Model | MC30 AUC | MC60 AUC | Extra score at delays 31–60 |
|---|---:|---:|---:|
| gamma | 14.836967 | 16.585112 | 1.748145 |
| cap95 | 14.613265 | 16.438353 | 1.825087 |
| cap99 | 14.711286 | 16.404013 | 1.692727 |
| empirical | 14.689000 | 16.302462 | 1.613462 |
| cap90 | 14.389844 | 16.254142 | 1.864299 |
| lognormal | 14.405406 | 15.848003 | 1.442597 |
| halfnormal | 14.060364 | 15.841660 | 1.781296 |
| hetero75 | 14.239572 | 15.838142 | 1.598569 |
| hetero50 | 13.788125 | 15.384796 | 1.596671 |
| binary | 13.139087 | 14.611681 | 1.472593 |

## Implications

- WS with empirical CE magnitudes remains first among the architecture models on integrated mean MC60. Gamma remains first among the ER distribution conditions. Some lower-ranked comparisons change, including the cap95/cap99 ordering.
- The ER gamma-minus-empirical integrated difference rises from 0.147968 at 30 delays to 0.282650 at 60 delays. These are descriptive estimates until paired intervals are recomputed.
- Report these as MC30 and MC60 rather than unqualified total memory capacity. Existing 60-delay results permit a horizon-sensitivity analysis without new reservoir simulations.
- Inspect the complete lag curves and sensitivity of matched-setting win fractions before finalizing the MC claims. For strong conditions with substantial delay-60 reconstruction, a targeted longer-horizon check is needed to estimate a sufficiently long memory horizon; simply changing the cutoff to 60 does not resolve truncation.
- NARMA10 scores are unaffected by the MC summation horizon.
- Sources, setting-level summaries and sampled lag profiles are in [audit_results.json](audit_results.json). The source score definition is `network_stats/audited_temporal_tasks.py:score_states`.

The current poster and previous figures remain unchanged by this audit.
