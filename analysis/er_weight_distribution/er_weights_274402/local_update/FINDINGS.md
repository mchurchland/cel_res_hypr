# Updated ER weight-distribution findings — run 274402

The completed experiment contains **792,000 rows: 100 paired realizations × 10 weight conditions × 11 E/I fractions × 3 scaling protocols × 24 hyperparameter settings**. It measures MC and NARMA10. All conditions use the parent's ER topology masks, sign masks, input projections, and task streams. These are weight-distribution controls on ER, not additional CE or WS topology comparisons.

## Main observations

- **The empirical distribution helps relative to binary and half-normal under own-radius scaling, but does not win everywhere.** Its win-fraction AUCs are 70.0% and 54.6% for MC, and 67.5% and 60.4% for NARMA10, respectively. Its average performance is also better than both baselines in that branch.
- **Moment-matched gamma performs better than empirical weights in most matched comparisons.** Under own-radius scaling, gamma's complementary win AUC is 96.7% for MC and 82.1% for NARMA10. Under shared scaling, these are 100.0% and 98.8%. The own-radius mean-performance AUC difference is modest: MC 14.837 versus 14.689; NARMA10 NMSE 0.15009 versus 0.15144. Frequent wins do not necessarily imply a large practical improvement.
- **Equal mean and variance do not make distributions interchangeable.** Empirical weights outperform the moment-matched lognormal in 97.9% (MC) and 84.6% (NARMA10) of the own-radius win AUC, and 100% of both shared-scaling win AUCs. Gamma and lognormal match empirical finite-vector mean, variance, and raw Frobenius norm. Their different outcomes implicate distributional properties beyond those two moments, without isolating a specific tail or eigenvalue mechanism.
- **The most extreme 1% of weights are not required for the own-radius aggregate performance.** Q99 capping slightly improves mean MC (14.711 versus 14.689) and leaves NARMA10 very close (0.15126 versus 0.15144). Q95/Q90 capping worsens both aggregate scores under own-radius or shared scaling. Caps are followed by mean restoration; they change more than the maximum alone.
- **The effect of tail capping reverses under Frobenius matching.** Each cap improves aggregate MC and NARMA10 relative to empirical weights; empirical MC win AUC falls to 0–1.0%, and NARMA10 win AUC to 0.6–1.5%. Frobenius matching changes the actual spectral radius, so this does not establish that either tail size or norm alone causes the difference.
- **Heterogeneity compression is also normalization dependent.** Compression worsens aggregate performance under own-radius/shared scaling but improves both scores under Frobenius matching. A general claim that more heterogeneous weights are always better is unsupported.
- **Mean performance and winning more settings can disagree.** Under Frobenius matching, empirical weights have better aggregate MC and NARMA10 than binary weights, yet binary wins more of the matched settings (72.5% MC and 77.7% NARMA10 win AUC). Retain both summaries in the poster.

## Empirical-versus-control win-fraction AUC

Each entry is the normalized area under the empirical matched-setting win-fraction curve over E/I 0–100%. Higher percentages favor empirical weights. Within each setting, scores are averaged over the 100 paired realizations before declaring a strict win; NARMA10 uses lower error as better. Trapezoidal integration gives half weight to endpoints, so these percentages are not a flat count of every tested row. Confidence intervals are in [win_fraction_auc.csv](../win_fraction_auc.csv).

| Control | Own MC | Own NARMA | Shared MC | Shared NARMA | Frobenius MC | Frobenius NARMA |
|---|---:|---:|---:|---:|---:|---:|
| Binary | 70.0% | 67.5% | 100.0% | 100.0% | 27.5% | 22.3% |
| Half-normal | 54.6% | 60.4% | 100.0% | 100.0% | 9.8% | 7.9% |
| Cap Q99 | 43.8% | 50.6% | 92.9% | 83.3% | 0.0% | 1.0% |
| Cap Q95 | 51.2% | 55.0% | 100.0% | 93.3% | 0.0% | 0.6% |
| Cap Q90 | 52.9% | 55.8% | 100.0% | 97.9% | 1.0% | 1.5% |
| Moment-matched gamma | 3.3% | 17.9% | 0.0% | 1.2% | 0.0% | 1.2% |
| Moment-matched lognormal | 97.9% | 84.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| Compression a=0.50 | 60.4% | 62.9% | 100.0% | 100.0% | 15.0% | 11.7% |
| Compression a=0.75 | 53.8% | 57.9% | 100.0% | 99.6% | 4.0% | 4.0% |

## Paired effect sizes

These are normalized performance-AUC differences, with positive values favoring empirical weights. For MC this is empirical minus control; for NARMA10 this is control NMSE minus empirical NMSE. Intervals resample the same whole paired realizations, preserving cross-condition dependence. The gamma improvement under own-radius scaling is small but consistent in the paired aggregate. Q99 capping gives a small MC improvement; its NARMA10 difference is uncertain.

| Scaling / control | MC advantage [95% interval] | NARMA10 advantage [95% interval] |
|---|---:|---:|
| Own radius / gamma | -0.1480 [-0.1570, -0.1394] | -0.001352 [-0.001996, -0.000839] |
| Own radius / cap Q99 | -0.0223 [-0.0326, -0.0128] | -0.000186 [-0.000936, 0.000364] |
| Own radius / half-normal | 0.6286 [0.5970, 0.6613] | 0.011115 [0.010143, 0.012123] |
| Frobenius / half-normal | -0.3941 [-0.4406, -0.3434] | -0.009617 [-0.010769, -0.008331] |

## Interpretation and limits

The supported finding is that **weight-distribution shape and normalization jointly affect computational performance**. These controls do not support a unique optimality of empirical CE magnitudes, or an explanation based solely on the largest weights, variance, or Frobenius norm. Gamma/lognormal use fixed mid-quantile vectors, placed by empirical rank with paired random tie-breaking. They are not independent random samples; they also change minimum magnitudes, weighted E/I balance, spectra, and other matrix properties.

Shared scaling uses the unmodified CE multiplier at each E/I fraction. Frobenius matching matches the norm of that shared-scaled CE reference; it is not an additional own-radius constraint. Shared and Frobenius branches coincide for empirical/gamma/lognormal by construction. The three branches must not be treated as three independent replications.

Results apply to this fixed 24-setting grid, ER topology ensemble, all-node input protocol, and these two tasks. E/I AUC uses uniform weighting of the measured E/I axis. Bootstrap intervals resample entire paired realizations (1,000 draws, seed 20261119); comparisons are exploratory and not corrected for multiplicity. Existing architecture/topology conclusions should be assessed from their original experiments rather than inferred from these ER-only controls.

## Figures and validation

- [MC performance AUC](../performance_auc_MC.png) and [NARMA10 performance AUC](../performance_auc_NARMA10_test_nmse.png).
- [MC matched-setting win AUC](../win_fraction_auc_MC.png) and [NARMA10 matched-setting win AUC](../win_fraction_auc_NARMA10_test_nmse.png).
- [Own-radius MC win curves](../win_fraction_MC_own_radius.png), [shared-scaling MC win curves](../win_fraction_MC_shared_ce_scale.png), and [Frobenius-matched MC win curves](../win_fraction_MC_frobenius_match.png).
- [Original methods report](../REPORT.md), [local validation receipt](validation.json), [archive transfer receipt](transfer_receipt.json), and [paired performance-AUC advantages](paired_performance_auc_advantages.csv). Positive values in the paired-advantage CSV favor empirical weights for either task.

The original downloaded analysis and its manifest are retained unchanged. This summary is an additional local artifact. The performance-AUC plots use separate horizontal limits by scaling branch; compare the numerical axes when judging effect sizes.
