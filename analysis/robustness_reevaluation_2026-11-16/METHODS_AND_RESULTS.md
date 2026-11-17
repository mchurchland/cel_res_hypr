# Reevaluate robustness in connectome reservoir computing

This analysis attempts to falsify a general performance–robustness tradeoff. All performance observations come from `final_results/`. No reservoir simulations were run. Existing results and figures were read only; their SHA256 hashes were verified before and after the analysis.

## Findings

### Does the association survive replacing CV? main

Spearman ρ and 95% seed-cluster bootstrap intervals:

- **MC:** CV 0.87 [0.87, 0.88]; SD 0.98 [0.98, 0.98]; IQR 0.95 [0.95, 0.96]; MAD 0.92 [0.91, 0.92].
- **IPC:** CV 0.63 [0.61, 0.65]; SD 0.96 [0.96, 0.97]; IQR 0.90 [0.89, 0.91]; MAD 0.87 [0.87, 0.88].
- **KR:** CV 0.96 [0.96, 0.97]; SD 1.00 [1.00, 1.00]; IQR 1.00 [1.00, 1.00]; MAD 1.00 [1.00, 1.00].

### Does the association survive replacing CV? shuf

Spearman ρ and 95% seed-cluster bootstrap intervals:

- **MC:** CV 0.88 [0.87, 0.89]; SD 0.97 [0.97, 0.97]; IQR 0.95 [0.95, 0.96]; MAD 0.94 [0.94, 0.95].
- **IPC:** CV 0.84 [0.82, 0.85]; SD 0.94 [0.94, 0.95]; IQR 0.93 [0.92, 0.93]; MAD 0.93 [0.92, 0.93].
- **KR:** CV 0.96 [0.95, 0.96]; SD 0.99 [0.99, 0.99]; IQR 1.00 [0.99, 1.00]; MAD 1.00 [1.00, 1.00].

Positive associations with SD, IQR, or MAD describe greater *absolute dispersion*, not lower useful capability. CV divides SD by mean; an algebraic denominator alone does not predict the observed sign. These correlations cannot identify a causal performance cost. The lower-tail and threshold analyses below test a different, more direct meaning of reliable computation.

### Are low performers called robust because they are uniformly poor?

For MC, the lowest-mean architecture is `binary_base`: mean 3.753, SD 1.118, CV 0.298, bottom-decile mean 2.030, and attainment 0.037 at the common exploratory threshold 6.003. The highest-mean architecture is `ws_p0.1+randN`: mean 10.626, SD 5.573, bottom-decile mean 3.931, attainment 0.867. Their 90% retention fractions are 0.026 and 0.041, respectively.

For IPC, the lowest-mean architecture is `binary_base`: mean 6.609, SD 1.528, CV 0.232, bottom-decile mean 4.060, and attainment 0.016 at the common exploratory threshold 10.209. The highest-mean architecture is `ws_p0.1+randN`: mean 16.699, SD 6.151, bottom-decile mean 7.301, attainment 0.901. Their 90% retention fractions are 0.040 and 0.021, respectively.

For KR, the lowest-mean architecture is `binary_base`: mean 2.942, SD 0.809, CV 0.273, bottom-decile mean 1.782, and attainment 0.000 at the common exploratory threshold 6.742. The highest-mean architecture is `ws_p0.1+randN`: mean 20.921, SD 16.227, bottom-decile mean 4.613, attainment 0.832. Their 90% retention fractions are 0.022 and 0.046, respectively.

A small absolute spread can therefore be consistent with consistently weak computation. A high retention fraction likewise requires a useful optimum before it can support a capability claim. No independently specified usefulness threshold was supplied; threshold attainment here is exploratory. A floor does **not** necessarily reduce CV: SD/mean can grow as the mean approaches zero.

### Which architecture comparisons remain consistent?

The following table gives C. elegans minus each main-experiment alternative. Positive mean, lower-tail and threshold differences favor C. elegans; negative dispersion or sensitivity differences favor C. elegans. These are differences of seed means; full metric uncertainty intervals are in `tables/comparison.csv`.

| metric | alternative | mean | sd | iqr | mad | bottom10 | attain_q50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MC | binary_base | 1.722 | 1.117 | 1.484 | 0.636 | 0.587 | 0.310 |
| MC | cel+randN | -3.706 | -2.693 | -2.634 | -1.268 | -0.578 | -0.382 |
| MC | er+randN | -4.925 | -3.263 | -2.487 | -0.944 | -1.143 | -0.506 |
| MC | global_sign_pres | 0.543 | 0.287 | 0.401 | 0.100 | 0.402 | 0.083 |
| MC | global_sign_pres_real_weight | -0.175 | -0.236 | -0.183 | -0.065 | 0.022 | -0.008 |
| MC | local_sign | -1.727 | -1.237 | -1.295 | -0.741 | -0.105 | -0.228 |
| MC | local_sign+binary | -0.535 | -0.487 | -0.707 | -0.436 | 0.260 | -0.097 |
| MC | local_sign+flat | -1.284 | -0.926 | -1.093 | -0.634 | 0.022 | -0.186 |
| MC | ws_p0.1+randN | -5.150 | -3.338 | -2.568 | -0.960 | -1.314 | -0.521 |
| IPC | binary_base | 2.227 | 1.114 | 1.637 | 0.811 | 0.799 | 0.275 |
| IPC | cel+randN | -5.730 | -3.134 | -3.419 | -1.765 | -1.071 | -0.474 |
| IPC | er+randN | -7.597 | -3.491 | -2.943 | -1.337 | -2.101 | -0.595 |
| IPC | global_sign_pres | 0.407 | -0.036 | 0.038 | 0.019 | 0.534 | 0.049 |
| IPC | global_sign_pres_real_weight | -0.413 | -0.439 | -0.429 | -0.208 | 0.007 | -0.041 |
| IPC | local_sign | -2.962 | -1.904 | -2.394 | -1.218 | -0.201 | -0.302 |
| IPC | local_sign+binary | -1.165 | -1.159 | -1.532 | -0.758 | 0.436 | -0.144 |
| IPC | local_sign+flat | -2.328 | -1.636 | -2.169 | -1.101 | 0.028 | -0.250 |
| IPC | ws_p0.1+randN | -7.862 | -3.509 | -3.030 | -1.298 | -2.442 | -0.611 |
| KR | binary_base | 2.610 | 1.403 | 1.866 | 0.926 | 0.871 | 0.266 |
| KR | cel+randN | -9.948 | -8.567 | -10.105 | -4.492 | -1.490 | -0.506 |
| KR | er+randN | -14.270 | -13.037 | -15.349 | -6.241 | -1.848 | -0.555 |
| KR | global_sign_pres | 0.578 | 0.083 | 0.252 | 0.179 | 0.293 | 0.076 |
| KR | global_sign_pres_real_weight | -1.949 | -1.135 | -1.638 | -0.809 | -0.538 | -0.192 |
| KR | local_sign | -5.164 | -4.434 | -5.231 | -2.305 | -0.821 | -0.357 |
| KR | local_sign+binary | -2.197 | -2.015 | -2.360 | -0.969 | -0.323 | -0.201 |
| KR | local_sign+flat | -3.923 | -3.440 | -4.035 | -1.728 | -0.616 | -0.302 |
| KR | ws_p0.1+randN | -15.369 | -14.015 | -16.693 | -6.780 | -1.961 | -0.566 |

### Comparable mean performance

Pairwise comparisons first restrict to the intersection of the two empirical mean ranges, then greedily match the closest means without replacement, with a caliper of 5% of the combined mean span. This is exploratory conditioning, not causal adjustment. Matches can change seed composition. All architecture pairs, counts and matched differences are in `tables/mean_matched.csv`; exact matched seeds are in `tables/mean_matched_pairs.csv`. A small match count prevents a general comparison.

| dataset | metric | architecture_a | architecture_b | n_overlap_a | n_overlap_b | n_matched | matched_difference_a_minus_b |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main | IPC | binary_base | real | 1 | 1 | 1 | 0.055 |
| main | IPC | cel+randN | real | 0 | 0 | 0 | — |
| main | IPC | er+randN | real | 0 | 0 | 0 | — |
| main | IPC | global_sign_pres | real | 200 | 196 | 113 | -0.007 |
| main | IPC | global_sign_pres_real_weight | real | 167 | 198 | 125 | 0.002 |
| main | IPC | local_sign | real | 7 | 31 | 7 | 0.036 |
| main | IPC | local_sign+binary | real | 100 | 195 | 60 | 0.010 |
| main | IPC | local_sign+flat | real | 20 | 179 | 15 | 0.046 |
| main | IPC | real | ws_p0.1+randN | 0 | 0 | 0 | — |
| main | KR | binary_base | real | 0 | 0 | 0 | — |
| main | KR | cel+randN | real | 0 | 0 | 0 | — |
| main | KR | er+randN | real | 0 | 0 | 0 | — |
| main | KR | global_sign_pres | real | 182 | 200 | 115 | -0.002 |
| main | KR | global_sign_pres_real_weight | real | 94 | 200 | 82 | 0.001 |
| main | KR | local_sign | real | 10 | 134 | 10 | 0.054 |
| main | KR | local_sign+binary | real | 82 | 156 | 47 | 0.016 |
| main | KR | local_sign+flat | real | 23 | 78 | 20 | 0.174 |
| main | KR | real | ws_p0.1+randN | 0 | 0 | 0 | — |
| main | MC | binary_base | real | 0 | 0 | 0 | — |
| main | MC | cel+randN | real | 0 | 0 | 0 | — |
| main | MC | er+randN | real | 0 | 0 | 0 | — |
| main | MC | global_sign_pres | real | 182 | 178 | 47 | -0.000 |
| main | MC | global_sign_pres_real_weight | real | 175 | 200 | 128 | 0.001 |
| main | MC | local_sign | real | 26 | 25 | 18 | 0.042 |
| main | MC | local_sign+binary | real | 156 | 193 | 91 | 0.000 |
| main | MC | local_sign+flat | real | 60 | 166 | 30 | 0.007 |
| main | MC | real | ws_p0.1+randN | 0 | 0 | 0 | — |
| shuf | IPC | binary+conshuffle+wshuffle | real | 200 | 186 | 121 | -0.002 |
| shuf | IPC | binary+shuffle | real | 182 | 198 | 150 | 0.001 |
| shuf | IPC | binary_base | real | 1 | 1 | 1 | 0.055 |
| shuf | IPC | binary_base_topology_shuffle | real | 0 | 0 | 0 | — |
| shuf | IPC | celW+connShuf | real | 5 | 6 | 5 | 0.034 |
| shuf | IPC | conn_shuf_only | real | 9 | 30 | 9 | 0.028 |
| shuf | IPC | global_sign_pres | real | 200 | 196 | 113 | -0.007 |
| shuf | IPC | local_sign+binary | real | 100 | 195 | 60 | 0.010 |
| shuf | IPC | real | shuffle | 18 | 8 | 8 | -0.006 |
| shuf | KR | binary+conshuffle+wshuffle | real | 177 | 195 | 96 | -0.002 |
| shuf | KR | binary+shuffle | real | 191 | 199 | 174 | 0.004 |
| shuf | KR | binary_base | real | 0 | 0 | 0 | — |
| shuf | KR | binary_base_topology_shuffle | real | 0 | 0 | 0 | — |
| shuf | KR | celW+connShuf | real | 5 | 12 | 5 | 0.034 |
| shuf | KR | conn_shuf_only | real | 6 | 91 | 6 | 0.041 |
| shuf | KR | global_sign_pres | real | 182 | 200 | 115 | -0.002 |
| shuf | KR | local_sign+binary | real | 82 | 156 | 47 | 0.016 |
| shuf | KR | real | shuffle | 70 | 18 | 15 | -0.057 |
| shuf | MC | binary+conshuffle+wshuffle | real | 183 | 145 | 57 | -0.003 |
| shuf | MC | binary+shuffle | real | 198 | 200 | 156 | -0.001 |
| shuf | MC | binary_base | real | 0 | 0 | 0 | — |
| shuf | MC | binary_base_topology_shuffle | real | 0 | 0 | 0 | — |
| shuf | MC | celW+connShuf | real | 23 | 27 | 17 | 0.050 |
| shuf | MC | conn_shuf_only | real | 17 | 87 | 15 | 0.040 |
| shuf | MC | global_sign_pres | real | 182 | 178 | 47 | -0.000 |
| shuf | MC | local_sign+binary | real | 156 | 193 | 91 | 0.000 |
| shuf | MC | real | shuffle | 77 | 49 | 23 | -0.025 |

For sd, 85/99 architecture pairs with at least 20 matches retain their full-sample difference sign. This count includes both main and shuffle panels and does not establish significance.

For iqr, 65/99 architecture pairs with at least 20 matches retain their full-sample difference sign. This count includes both main and shuffle panels and does not establish significance.

For mad, 60/99 architecture pairs with at least 20 matches retain their full-sample difference sign. This count includes both main and shuffle panels and does not establish significance.

For bottom10, 50/99 architecture pairs with at least 20 matches retain their full-sample difference sign. This count includes both main and shuffle panels and does not establish significance.

### Floors and finite bounds

The code sums 30 clamped squared correlations for MC and 30 delays × five single-order Legendre targets for IPC. Their implementation-based bounds are [0,30] and [0,150]. KR is effective rank of a state matrix with 297 node features, hence [0,297]; nonzero effective rank is at least one but a zero state matrix returns zero. The 297-node adjacency shape was checked during inspection. The current launcher/protocol supports these ceilings, but the archival CSVs do not embed delay/order/stream counts, so historical upper-bound conclusions are explicitly conditional. IPC here is the implemented restricted target set, not a full polynomial-basis IPC measurement.

For a bounded score a ≤ P ≤ b, nonnegativity of E[(P−a)(b−P)] gives Var(P) ≤ (μ−a)(b−μ). We report variance divided by this ceiling, and plots versus distances from both bounds. The ceiling constrains possibilities; it is not expected null variance or a correction that establishes intrinsic sensitivity. Zero ceilings yield an undefined ratio. Main Q25 threshold fractions, SD restricted to configurations above Q25 (minimum 20 retained configurations), and descriptive partial rank correlations controlling the low-performance fraction provide additional floor diagnostics. The truncated subsets vary between surfaces and should not be treated as the original fixed design.

Observed nonfinite values across selected metric rows: 0; negative values: 0; values outside the conditional bounds: 0. No clipping or imputation was applied.

### Hyperparameter variation versus seed variation

The explicit `(architecture, normalization, seed, hyperparameters)` identity is unique. Each seeded surface contains the same 96 configurations. The main, own-radius shuffle, fixed-reference shuffle and sign-only experiments have 200 repeats; normalization ablations have 100. `src` records chunks; it is not an independent biological sample. `shuffle_id` identifies the repeat but seeds provide the checked cross-configuration identity. Building a reservoir resets its RNG at the same seed for each grid point, keeping the underlying randomized network/input-weight realization paired across the grid. Bias uses a separate fixed seed. Updated KR input streams use explicit per-repeat offsets; MC/IPC and network variation remain bundled in the recorded seed.

For each architecture, metric and normalization, the balanced matrix Y(seed,configuration) is decomposed into grand mean, configuration main effect, seed main effect, and residual. Population variances sum exactly. Between-configuration variance is variance of configuration means; within-configuration variance is the mean variance across seeds, equal to seed plus residual components. One observation per cell means the residual combines seed×configuration interaction with evaluation noise. These data cannot identify those two sources separately, or separate biological wiring uncertainty, randomized input weights and stimulus sampling.

| architecture | metric | configuration_variance | seed_variance | interaction_noise_variance |
| --- | --- | --- | --- | --- |
| binary_base | MC | 0.831 | 0.024 | 0.145 |
| binary_base | IPC | 0.597 | 0.041 | 0.361 |
| binary_base | KR | 0.861 | 0.106 | 0.033 |
| cel+randN | MC | 0.969 | 0.009 | 0.022 |
| cel+randN | IPC | 0.935 | 0.012 | 0.054 |
| cel+randN | KR | 0.952 | 0.029 | 0.019 |
| er+randN | MC | 0.981 | 0.004 | 0.016 |
| er+randN | IPC | 0.950 | 0.006 | 0.043 |
| er+randN | KR | 0.986 | 0.008 | 0.006 |
| global_sign_pres | MC | 0.917 | 0.010 | 0.073 |
| global_sign_pres | IPC | 0.815 | 0.018 | 0.166 |
| global_sign_pres | KR | 0.927 | 0.049 | 0.024 |
| global_sign_pres_real_weight | MC | 0.878 | 0.046 | 0.076 |
| global_sign_pres_real_weight | IPC | 0.799 | 0.052 | 0.149 |
| global_sign_pres_real_weight | KR | 0.616 | 0.286 | 0.098 |
| local_sign | MC | 0.889 | 0.047 | 0.064 |
| local_sign | IPC | 0.832 | 0.054 | 0.114 |
| local_sign | KR | 0.800 | 0.119 | 0.081 |
| local_sign+binary | MC | 0.906 | 0.028 | 0.066 |
| local_sign+binary | IPC | 0.825 | 0.037 | 0.138 |
| local_sign+binary | KR | 0.839 | 0.097 | 0.064 |
| local_sign+flat | MC | 0.893 | 0.043 | 0.065 |
| local_sign+flat | IPC | 0.828 | 0.052 | 0.120 |
| local_sign+flat | KR | 0.810 | 0.113 | 0.077 |
| real | MC | 0.933 | 0.015 | 0.052 |
| real | IPC | 0.836 | 0.022 | 0.142 |
| real | KR | 0.922 | 0.058 | 0.019 |
| ws_p0.1+randN | MC | 0.982 | 0.004 | 0.014 |
| ws_p0.1+randN | IPC | 0.954 | 0.007 | 0.040 |
| ws_p0.1+randN | KR | 0.985 | 0.008 | 0.006 |

`tables/per_realization.csv` computes robustness before averaging; `tables/averaged_surface.csv` instead summarizes the performance surface averaged across seeds. These are distinct estimands: averaging can smooth an individually sensitive surface. Bootstrap intervals resample seed blocks, keeping all architectures/sign conditions that share a seed together for pooled correlations; no hyperparameter is treated as a biological replicate. Architecture mean intervals use independent seed draws within each marginal architecture summary. Correlation intervals condition on the selected architecture panel and grid; architecture-centroid and partial-rank correlations are descriptive without sampling intervals.

### How much can spectral normalization explain?

The stored sign-normalization experiments provide own-radius and fixed-reference scaling on the full grid. Own-radius scaling uses target/ρ(W); reference scaling uses target/ρ(W_reference), allowing actual post-scaling radius to vary. The parameter called sign-flip fraction is the requested **negative-edge fraction**: the implementation takes absolute weights and assigns random edge signs. It does not selectively perturb identified inhibitory neurons. At the empirical fraction it randomizes sign locations, so that point is not the unperturbed connectome.

We compare modes **within each archival normalization experiment**, retaining its KR vintage. We do not substitute updated sign-only KR values into these contrasts. The updated main/shuffle KR and archival control KR are not proven protocol-compatible; cross-run shuffle normalization contrasts therefore use MC/IPC only. Even MC/IPC can differ across runs at unchanged scaling, so these contrasts include evaluation/run effects. Correlations with raw radius, scale factor or E/I fraction do not isolate any one mechanism.

| dataset | architecture | metric | measure | own_minus_fixed | ci_low | ci_high | n_paired_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| norm_cel_matched | sign_test_og_cel0.0 | IPC | mean | 0.196 | 0.135 | 0.254 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | IPC | sd | -0.154 | -0.182 | -0.126 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | IPC | bottom10 | 0.508 | 0.415 | 0.600 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | KR | mean | -2.688 | -2.803 | -2.572 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | KR | sd | -1.806 | -1.873 | -1.738 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | KR | bottom10 | -0.870 | -0.921 | -0.820 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | MC | mean | 0.172 | 0.144 | 0.201 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | MC | sd | 0.063 | 0.048 | 0.078 | 100 |
| norm_cel_matched | sign_test_og_cel0.0 | MC | bottom10 | 0.188 | 0.140 | 0.239 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | IPC | mean | 1.364 | 1.203 | 1.522 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | IPC | sd | 1.017 | 0.921 | 1.118 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | IPC | bottom10 | 0.170 | 0.089 | 0.249 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | KR | mean | 4.813 | 4.221 | 5.411 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | KR | sd | 2.739 | 2.416 | 3.070 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | KR | bottom10 | 1.294 | 1.133 | 1.460 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | MC | mean | 1.067 | 0.952 | 1.185 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | MC | sd | 0.855 | 0.770 | 0.938 | 100 |
| norm_cel_matched | sign_test_og_cel0.4 | MC | bottom10 | 0.223 | 0.176 | 0.270 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | IPC | mean | -0.587 | -0.624 | -0.549 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | IPC | sd | -0.274 | -0.304 | -0.245 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | IPC | bottom10 | -0.250 | -0.331 | -0.166 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | KR | mean | -2.131 | -2.223 | -2.042 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | KR | sd | -1.678 | -1.744 | -1.614 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | KR | bottom10 | -0.497 | -0.514 | -0.481 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | MC | mean | -0.550 | -0.571 | -0.528 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | MC | sd | -0.297 | -0.311 | -0.283 | 100 |
| norm_cel_matched | sign_test_og_cel1.0 | MC | bottom10 | -0.238 | -0.273 | -0.204 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | IPC | mean | -0.089 | -0.111 | -0.067 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | IPC | sd | -0.065 | -0.089 | -0.041 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | IPC | bottom10 | 0.027 | -0.043 | 0.096 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | KR | mean | -0.381 | -0.391 | -0.370 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | KR | sd | -0.219 | -0.224 | -0.214 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | KR | bottom10 | -0.155 | -0.162 | -0.148 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | MC | mean | -0.021 | -0.032 | -0.010 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | MC | sd | 0.011 | 0.002 | 0.021 | 100 |
| norm_cel_removed | sign_test_og_cel0.0 | MC | bottom10 | -0.015 | -0.052 | 0.019 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | IPC | mean | 1.414 | 1.302 | 1.529 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | IPC | sd | 0.823 | 0.742 | 0.906 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | IPC | bottom10 | 0.421 | 0.344 | 0.502 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | KR | mean | 3.453 | 3.189 | 3.728 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | KR | sd | 1.551 | 1.431 | 1.676 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | KR | bottom10 | 1.259 | 1.161 | 1.365 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | MC | mean | 0.996 | 0.915 | 1.071 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | MC | sd | 0.704 | 0.641 | 0.765 | 100 |
| norm_cel_removed | sign_test_og_cel0.4 | MC | bottom10 | 0.321 | 0.275 | 0.366 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | IPC | mean | -0.117 | -0.143 | -0.092 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | IPC | sd | -0.003 | -0.026 | 0.021 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | IPC | bottom10 | -0.083 | -0.159 | -0.011 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | KR | mean | -0.377 | -0.385 | -0.368 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | KR | sd | -0.205 | -0.211 | -0.199 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | KR | bottom10 | -0.143 | -0.148 | -0.137 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | MC | mean | -0.070 | -0.081 | -0.058 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | MC | sd | 0.008 | -0.001 | 0.018 | 100 |
| norm_cel_removed | sign_test_og_cel1.0 | MC | bottom10 | -0.051 | -0.084 | -0.018 | 100 |
| norm_matched_er | sign_test_er0.0 | IPC | mean | 2.946 | 2.888 | 3.003 | 100 |
| norm_matched_er | sign_test_er0.0 | IPC | sd | -0.375 | -0.415 | -0.331 | 100 |
| norm_matched_er | sign_test_er0.0 | IPC | bottom10 | 2.784 | 2.703 | 2.862 | 100 |
| norm_matched_er | sign_test_er0.0 | KR | mean | -3.524 | -3.580 | -3.467 | 100 |
| norm_matched_er | sign_test_er0.0 | KR | sd | -3.722 | -3.787 | -3.659 | 100 |
| norm_matched_er | sign_test_er0.0 | KR | bottom10 | 0.035 | 0.013 | 0.055 | 100 |
| norm_matched_er | sign_test_er0.0 | MC | mean | 1.958 | 1.926 | 1.990 | 100 |
| norm_matched_er | sign_test_er0.0 | MC | sd | 0.108 | 0.091 | 0.125 | 100 |
| norm_matched_er | sign_test_er0.0 | MC | bottom10 | 1.539 | 1.496 | 1.580 | 100 |
| norm_matched_er | sign_test_er0.4 | IPC | mean | 6.068 | 5.963 | 6.171 | 100 |
| norm_matched_er | sign_test_er0.4 | IPC | sd | 1.413 | 1.335 | 1.494 | 100 |
| norm_matched_er | sign_test_er0.4 | IPC | bottom10 | 3.119 | 2.980 | 3.261 | 100 |
| norm_matched_er | sign_test_er0.4 | KR | mean | 20.549 | 20.190 | 20.905 | 100 |
| norm_matched_er | sign_test_er0.4 | KR | sd | 17.788 | 17.518 | 18.061 | 100 |
| norm_matched_er | sign_test_er0.4 | KR | bottom10 | 2.354 | 2.313 | 2.394 | 100 |
| norm_matched_er | sign_test_er0.4 | MC | mean | 4.607 | 4.555 | 4.658 | 100 |
| norm_matched_er | sign_test_er0.4 | MC | sd | 2.669 | 2.637 | 2.701 | 100 |
| norm_matched_er | sign_test_er0.4 | MC | bottom10 | 1.798 | 1.727 | 1.873 | 100 |
| norm_matched_er | sign_test_er1.0 | IPC | mean | 0.692 | 0.633 | 0.745 | 100 |
| norm_matched_er | sign_test_er1.0 | IPC | sd | -1.668 | -1.735 | -1.604 | 100 |
| norm_matched_er | sign_test_er1.0 | IPC | bottom10 | 2.519 | 2.449 | 2.593 | 100 |
| norm_matched_er | sign_test_er1.0 | KR | mean | -2.435 | -2.477 | -2.395 | 100 |
| norm_matched_er | sign_test_er1.0 | KR | sd | -2.783 | -2.825 | -2.743 | 100 |
| norm_matched_er | sign_test_er1.0 | KR | bottom10 | 0.537 | 0.528 | 0.546 | 100 |
| norm_matched_er | sign_test_er1.0 | MC | mean | 0.177 | 0.152 | 0.203 | 100 |
| norm_matched_er | sign_test_er1.0 | MC | sd | -0.817 | -0.845 | -0.787 | 100 |
| norm_matched_er | sign_test_er1.0 | MC | bottom10 | 0.888 | 0.832 | 0.944 | 100 |

Maximum stored normalization-factor identity error is 1.67e-16. Full raw radius, factor, post-radius and robustness diagnostics appear in the spectral figures and tables. No truly unnormalized performance control was identified among the curated merged tables. Fixed-reference controls exist and suffice to test whether the original conclusion depends on changing rescaling. The data support reporting the magnitude and direction of this dependence, not a causal percentage of the total performance–robustness association explained.

### What remains supported about C. elegans?

The stored experiments support architecture-specific differences in simulated MC, the implemented IPC and KR under the chosen dynamics, input/readout protocol and grid. The comparison table identifies whether C. elegans improves absolute mean, lower-tail capability or threshold attainment against each null model. A claim of a universal biological performance–robustness tradeoff requires more than a positive CV correlation, particularly if higher-performing models retain better lower-tail performance. These are randomized computational realizations of one anatomical connectome, not independent worms or direct measurements of in vivo computation. Effective-rank differences are not by themselves task accuracy or biological adaptation.

## Definitions and reproducibility

All configurations receive equal weight. SD uses ddof=0; IQR is Q75−Q25; MAD is the unscaled median absolute deviation. Quantiles use linear interpolation. Worst-decile mean uses the lowest ceil(0.1n) values: 10 of 96, an explicit finite-grid convention. Minimum, p10, maximum and range are also reported. CV is undefined for a nonpositive mean; 80%/90% retention is undefined for a nonpositive optimum. Constant positive surfaces have zero spread and full retention; all-zero surfaces have zero absolute spread and undefined relative metrics. Nonfinite scores are excluded and counted, negative scores are retained in absolute metrics and flagged. Duplicate identities fail rather than being silently averaged. Variance partitioning is omitted if cells are missing.

Local sensitivity uses only adjacent design levels with all other coordinates unchanged. Radius and input scale use log coordinates; leak and bias use linear coordinates, including zero bias. Each dimension is normalized to its full grid span [0,1]. We report mean and p90 slopes per dimension, and an edge-weighted mean across dimensions for the overview scatterplots. Values are performance units per normalized coordinate unit; they are not comparable across MC/IPC/KR scales. A missing intermediate level is not bridged within a realization. Performance range and minima are descriptive and sensitive to extreme observations. Main and shuffle panels share some architectures but are analyzed separately, never pooled as independent repeats.

Thresholds are identical across all architectures/conditions within a metric. Anchor thresholds are main pooled Q25/Q50/Q75, with the main grid equally represented for each architecture. Curves cover zero through the largest stored finite score on 81 equally spaced levels plus anchors and an upper endpoint. All are exploratory; none is claimed to be independently useful. Raw thresholds are never pooled across metrics.

| metric | attain_q25 | attain_q50 | attain_q75 |
| --- | --- | --- | --- |
| MC | 4.023 | 6.003 | 8.659 |
| IPC | 7.269 | 10.209 | 14.322 |
| KR | 4.170 | 6.742 | 12.282 |

The analysis uses 2000 percentile-bootstrap replicates at 95% confidence and RNG seed 20261116. Numerous exploratory associations are reported without multiplicity-adjusted hypothesis tests. Do not read their intervals as confirmatory evidence selected independently of the data.

## Files

- `manifest.json`: exact source paths, SHA256 hashes, code hashes, environment, filtering, coordinates, thresholds and invocation.
- `tables/comparison.csv`: every robustness metric with seed-bootstrap mean intervals.
- `tables/per_realization.csv`, `averaged_surface.csv`, `variance_components.csv`: separate estimands and finite-design decomposition.
- `tables/associations.csv`: pooled, within-architecture, architecture-centroid and floor-controlled associations.
- `tables/mean_matched.csv` and `mean_matched_pairs.csv`: overlap, calipers, observation counts and selected seeds.
- `tables/threshold_curves.csv`: absolute threshold curves with intervals.
- `tables/spectral_per_seed.csv`, `normalization_differences.csv`, `spectral_associations.csv`: normalization diagnostics.
- `tables/data_audit.csv`: completeness, bounds, missing/negative values by architecture and metric.
- `figures/`: publication-resolution PNG and vector PDF figures, preserving thesis sign-balance groups/colors.
- `cache/`: fingerprinted extraction tables and compressed seed×configuration matrices.

Regenerate in a new directory with `python tools/reevaluate_robustness.py --out-dir analysis/robustness_new`; add `--resume` only for this analysis's own directory. Run synthetic checks with `python -m unittest discover -s tests -p 'test_robustness.py'`.

## Methodological references

The implemented IPC should be distinguished from the broader capacity framework in [Dambre et al. (2012)](https://www.nature.com/articles/srep00514). The analysis uses Spearman rank association and seed-block percentile bootstrap intervals; see the official [SciPy Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) and [bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for the underlying statistical definitions. The cluster resampling and finite-design variance partition are implemented explicitly here. Bounds follow directly from the inspected repository score definitions and the inequality derived above.
