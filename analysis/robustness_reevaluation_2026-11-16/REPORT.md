# Reevaluate the performance–robustness tradeoff

**The positive performance–dispersion association survives removing CV. The broader claim that higher-performing reservoirs have less reliable computational capability does not follow:** higher-performing reservoirs also have better lower-tail performance and attain common absolute thresholds more often. Normalization substantially changes the sign-sweep results.

All experiment data came from nine curated merged tables in `final_results/`. No simulations were launched. New outputs are isolated here, and all input SHA256 hashes were verified unchanged.

## Does the association persist without CV?

Main experiment Spearman correlations, with 95% seed-cluster bootstrap intervals:

| metric | cv | sd | iqr | mad | bottom10 | attain_q50 |
| --- | --- | --- | --- | --- | --- | --- |
| MC | 0.874 [0.865, 0.884] | 0.981 [0.979, 0.983] | 0.952 [0.949, 0.956] | 0.915 [0.910, 0.921] | 0.888 [0.880, 0.896] | 0.994 [0.993, 0.994] |
| IPC | 0.629 [0.611, 0.647] | 0.965 [0.962, 0.968] | 0.902 [0.894, 0.909] | 0.874 [0.865, 0.882] | 0.847 [0.836, 0.858] | 0.992 [0.992, 0.993] |
| KR | 0.963 [0.960, 0.966] | 0.996 [0.995, 0.996] | 0.998 [0.998, 0.998] | 0.999 [0.999, 0.999] | 0.986 [0.984, 0.988] | 0.995 [0.994, 0.996] |

The own-radius shuffle panel gives the same qualitative result; its SD correlations with mean are MC 0.970, IPC 0.945, KR 0.993. Thus CV's denominator alone cannot explain the observed positive association. SD restricted to configurations above the exploratory main Q25 threshold also remains positively associated with mean (minimum 20 surviving configurations). This diagnostic changes the configuration subset and does not isolate a floor mechanism.

Within the C. elegans architecture, the mean–SD correlations across seeds are IPC 0.040 [-0.097, 0.173]; KR 0.919 [0.888, 0.939]; MC 0.214 [0.067, 0.346]. The very strong pooled associations therefore also reflect differences between architectures; they are not a universal within-architecture law.

## Are weak reservoirs being called robust?

Yes, a small spread can reward consistently weak performance. In the main experiment, the binary architecture has the lowest mean and SD for all three metrics. Its bottom-decile means, compared with the best mean-performing architecture (`ws_p0.1+randN`), are:

| metric | binary_bottom10 | WS_bottom10 | binary_threshold_fraction | WS_threshold_fraction |
| --- | --- | --- | --- | --- |
| MC | 2.030 | 3.931 | 0.037 | 0.867 |
| IPC | 4.060 | 7.301 | 0.016 | 0.901 |
| KR | 1.782 | 4.613 | 0.000 | 0.832 |

Threshold fractions use an identical main pooled median threshold within each metric. All thresholds are exploratory; no independently justified useful-performance cutoffs were available. Retention at 80% and 90% of each realization's maximum is reported alongside absolute performance and cannot establish useful capability on its own. A floor need not lower CV, since its denominator also shrinks.

## Which C. elegans comparisons remain supported?

C. elegans exceeds the binary architecture in mean, bottom-decile performance and median-threshold attainment for MC, IPC and KR. Gaussian-weight CE, ER and WS models exceed C. elegans on all three of those capability summaries, while showing greater SD/IQR/MAD. The results therefore support neither universal C. elegans superiority nor a general loss of reliable capability in the higher-performing models. C. elegans has better lower-tail MC and IPC than several sign-preserving controls despite a lower mean; the complete comparison is in [COMPARISON.md](COMPARISON.md) and `tables/comparison.csv`.

In the shuffle panel, weight shuffle, connection shuffle and joint weight/connection shuffle all have higher estimated mean and bottom-decile MC, IPC and KR than the intact weighted connectome, alongside greater absolute dispersion. Thus preserving the biological weight/topology arrangement is not uniformly advantageous under this protocol. Paired seed-bootstrap differences for C. elegans versus each alternative, including all these comparisons, are in `tables/architecture_contrasts.csv`.

Mean matching is limited by overlap: 113/270 architecture-pair/metric comparisons have no matches, and 99/270 have at least 20. We restrict to empirical overlapping mean ranges, then match without replacement within a 5%-of-combined-span caliper. Counts and residual mean differences are recorded for every pair. Conditioning on mean is descriptive and does not establish causality.

Mean-matched differences can change sign; consult `tables/mean_matched.csv` and `mean_matched_sign_consistency.csv` before generalizing an ordering. A pooled positive correlation does not imply a universal comparison at equal mean performance.

One comparison is consistent at comparable means: C. elegans has lower estimated SD, IQR and MAD, and a higher bottom-decile mean, than the sign-preserving ±1 architecture (MC: 91 matched pairs; IPC: 60 matched pairs; KR: 47 matched pairs). These are conditional descriptive comparisons, with sparse or absent overlap preventing many other architecture comparisons. C. elegans and the Gaussian CE/ER/WS architectures have no overlapping mean ranges for any metric.

## How much depends on spectral normalization?

Fixed-reference controls already exist. For the sign-matched connectome at negative-edge fraction 0.4, the following are own-radius minus fixed-reference differences from paired seeds within the same archival experiment:

| metric | measure | own_minus_fixed | ci_low | ci_high | n_paired_seeds |
| --- | --- | --- | --- | --- | --- |
| IPC | mean | 1.364 | 1.203 | 1.522 | 100 |
| IPC | sd | 1.017 | 0.921 | 1.118 | 100 |
| IPC | bottom10 | 0.170 | 0.089 | 0.249 | 100 |
| KR | mean | 4.813 | 4.221 | 5.411 | 100 |
| KR | sd | 2.739 | 2.416 | 3.070 | 100 |
| KR | bottom10 | 1.294 | 1.133 | 1.460 | 100 |
| MC | mean | 1.067 | 0.952 | 1.185 | 100 |
| MC | sd | 0.855 | 0.770 | 0.938 | 100 |
| MC | bottom10 | 0.223 | 0.176 | 0.270 | 100 |

Own-radius scaling amplifies changes near balanced signs because it divides by the altered raw radius; fixed-reference scaling changes that result substantially. This establishes dependence on the normalization protocol, not a causal percentage explained by raw radius. Sign structure, effective recurrent gain and other dynamics change together. The sweep assigns a negative-edge fraction after taking absolute weights, rather than selectively flipping known inhibitory neurons. At the empirical fraction it is still a randomized sign assignment.

KR normalization contrasts stay within the archival control tables: their exact historical rank protocol is not embedded in the CSVs and differs in provenance from the updated main/sign-only KR runs. Cross-run shuffle normalization contrasts therefore use MC/IPC only. Unchanged-scaling baselines also exhibit run variation. These limitations prevent a clean attribution of the total association to spectral normalization. No curated unnormalized performance control was identified.

## Seed variability, bounds and limits

The full design is 4 radii × 3 leaks × 4 input scales × 2 bias ranges = 96 configurations. There are 200 seeds per architecture in main, shuffle and sign-only tables, and 100 per sign-normalization condition. Robustness is computed per seed before averaging; robustness of the configuration-averaged surface is a separate table. Exact balanced-design variance partitioning separates configuration effects, seed effects and residual interaction plus noise. One observation per seed/configuration prevents separating interaction from evaluation noise or separating network, input and stimulus randomness. Seeds are computational repeats of one connectome, not independent biological networks.

Configuration main effects account for MC 83.1%–98.2%; IPC 59.7%–95.4%; KR 61.6%–98.6% of finite-design total variance across main architectures. Residual and seed components remain architecture-dependent. These are descriptive partitions, not identified biological variance components.

Bounds follow from the inspected implementation: MC ≤30, restricted IPC ≤150, and KR ≤297. Historical delay/order/stream metadata are absent, so archived upper-bound diagnostics are conditional on the documented protocol. No selected finite observation violates these bounds. Variance is compared with the feasible ceiling (μ−a)(b−μ), not an assumed expected variance. The ceiling-normalized dispersion remains associated with mean, but cannot establish intrinsic sensitivity.

Local sensitivity uses adjacent configurations differing in one dimension, with log radius/input scale and linear leak/bias coordinates scaled to [0,1]. Both mean and p90 sensitivity are reported. Threshold curves, all dispersion definitions, lower tails, retention, range, missing-value counts and exact seed matches are retained in the tables.

## Scientific conclusion

Replace the unqualified performance–robustness tradeoff with a narrower statement: **higher mean performance is associated with larger absolute hyperparameter variation on this grid, while useful lower-tail capability often improves at the same time.** Topology, weights and sign structure affect the simulated computation, and the size and direction of sign-sweep effects depend on normalization. These observations do not establish evolutionary optimization, in vivo computational superiority, or an intrinsic causal cost of performance.

## Reproduce and inspect

- [Full methods, numerical comparisons and limitations](METHODS_AND_RESULTS.md)
- [Compact main comparison](COMPARISON.md)
- [Compact performance–dispersion–lower-tail comparison](figures/main_key_comparison.pdf)
- [All robustness definitions](figures/main_mean_vs_robustness.pdf)
- [Absolute threshold curves](figures/main_threshold_attainment.pdf)
- [Local sensitivity](figures/main_local_sensitivity.pdf)
- [Normalization diagnostics](figures/norm_spectral_balance_diagnostics.pdf)

`manifest.json` records all sources, hashes, settings and package versions. Intervals use 2000 seed-cluster bootstrap draws, RNG seed 20261116, conditional on the chosen architecture panel/grid. All analyses are exploratory. Regenerate with `python tools/reevaluate_robustness.py --out-dir analysis/robustness_new`; synthetic tests: `python -m unittest discover -s tests -p 'test_robustness.py'`.
