# Key figures: connectome reservoirs and weight distributions

Updated 22 November 2026. The first three figures cover the main results. The last two provide the full weight-control comparisons.

All figures below use the **own-radius protocol**, with 100 paired realizations, 24 hyperparameter settings and 11 inhibitory-edge fractions from 0–100%. MC is better when higher; NARMA10 normalized mean squared error (NMSE) is better when lower.

## 1. Where does CE outperform ER?

![CE versus ER with binary, half-normal and empirical CE magnitudes across the inhibitory-edge sweep](figures/01_ce_vs_er.png)

CE's wins against binary ER concentrate near the ends of the inhibitory-edge sweep. Giving ER the CE magnitude distribution removes the MC advantage and substantially reduces the NARMA10 advantage. The comparator therefore changes the conclusion.

The vertical axis is the percentage of **24 matched settings** where CE has better mean performance. Scores are averaged over paired realizations before counting wins; these are not percentages of individual trials won. Blue = binary ER; amber = half-normal ER; purple = ER with CE magnitudes.

**Use this figure to introduce the comparison and explain why weight-matched controls are needed.**

## 2. Absolute performance of the architecture controls

![Mean-performance AUCs for original CE, shuffled CE, degree-rewired CE, ER and directed WS](figures/02_architecture_performance.png)

Original CE has lower aggregate MC and higher NARMA10 error than the shuffled, rewired, ER and directed WS controls shown. These comparisons retain the raw magnitude multiset while changing placement or connectivity. They do not establish that biological wiring is universally a poor computational substrate.

Points summarize performance averaged over settings and integrated over the E/I sweep. Bars show 95% paired-realization bootstrap intervals. This figure shows the **magnitude of performance**, complementing the fraction of settings won in Figure 1. WS denotes the directed Watts–Strogatz variant used in the experiment.

**Use this figure to support the statement that CE's original arrangement underperforms these controls under the tested protocol.**

## 3. Gamma magnitudes improve the ER reservoir

![Paired MC improvement and NARMA10 error reduction when empirical ER magnitudes are replaced by moment-matched gamma magnitudes](figures/03_gamma_improvement.png)

The experiment follows this sequence:

1. Measure the mean and variance of CE connection magnitudes.
2. Construct a positive gamma magnitude vector matching those realized moments.
3. Replace magnitudes on the same ER network, retaining topology, signs, inputs and rank-based placement.

Both panels show **positive values when gamma is better**. The mean MC gain is positive across the sweep. NARMA10 improvements are less uniform, with uncertain or negative differences at some intermediate fractions. Bands retain that uncertainty.

| Result across the E/I sweep | Gamma improvement, with 95% paired interval | Relative improvement | Gamma win-fraction AUC |
|---|---:|---:|---:|
| MC | +0.1480 [0.1394, 0.1570] | 1.0% higher | 96.7% |
| NARMA10 error reduction | +0.001352 [0.000839, 0.001996] | 0.9% lower error | 82.1% |

The effect sizes integrate the mean-performance differences over E/I, divided by the sweep span. Win-fraction AUC instead integrates the fraction of matched settings won after averaging scores over realizations. It is not ROC AUC or a flat count of all individual observations.

**Use this figure for the constructive finding: a gamma approximation of CE's weights improves ER performance without reproducing CE's wiring.** The improvement is modest, and the experiment does not establish that CE's particular moments are uniquely useful. The fit uses no task-performance data; the gamma vector is constructed from fixed distribution quantiles rather than iid draws.

## 4. Full weight-control comparisons: MC

![Fraction of matched settings in which empirical ER magnitudes outperform each alternative distribution for MC](figures/04_mc_weight_controls.png)

Here the comparison direction changes: **higher curves favor empirical magnitudes**, while lower curves favor the named alternative. In particular, the low gamma curve means gamma usually outperforms empirical weights.

Gamma and lognormal match the empirical mean and variance, yet perform differently. This supports a role for distribution shape beyond those moments. Tail caps and heterogeneity compression provide additional context; the result is not simply that every heterogeneous distribution works equally well.

**Keep this as a supporting figure for questions about which weight properties matter.**

## 5. Full weight-control comparisons: NARMA10

![Fraction of matched settings in which empirical ER magnitudes outperform each alternative distribution for NARMA10](figures/05_narma10_weight_controls.png)

This asks the same question on nonlinear prediction, using lower test NMSE to determine the better model. Again, higher curves favor empirical magnitudes and lower curves favor the alternative. Compare this with the MC panel to distinguish findings shared by both tasks from task-dependent differences.

**Use Figures 4–5 together when discussing the complete distribution-control experiment.**

## Reading and sharing the figures

- Confidence intervals resample entire paired realizations: 1,000 draws, seed 20261119. Hyperparameter settings are fixed design points, not independent biological samples. Intervals are exploratory and are not adjusted for multiple comparisons.
- E/I AUC uses uniform trapezoidal integration over the sampled fractions, giving endpoints half the weight of equally spaced interior points. A small advantage counts as a win, which is why the absolute performance figures are also needed.
- These are static-weight tanh reservoir comparisons on MC and NARMA10. They do not establish biological optimality or a specific eigenmode mechanism.
- Keep the adjacent `figures/` folder with this Markdown file when sharing it. All five previews are copied here; the originals are preserved. Exact source paths and PNG hashes are in [figure_sources.json](figure_sources.json). The gamma estimates are also in [gamma_effect_sizes.csv](gamma_effect_sizes.csv).

[Current poster PDF](../../thesis_tex/neurreps_2026/poster/poster.pdf) · [Poster preview](../../thesis_tex/neurreps_2026/poster/poster_preview.png) · [Detailed methods and figure provenance](../../thesis_tex/neurreps_2026/poster/SOURCES.md)
