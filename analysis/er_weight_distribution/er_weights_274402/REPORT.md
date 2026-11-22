# Paired ER weight-distribution experiment

792,000 rows; 100 paired realizations; 24 hyperparameter settings;
11 inhibitory-edge fractions. All topology masks,
sign masks, input projections and task streams come from parent 271871.
MC and NARMA10 use its audited float64 scoring protocol. Higher MC is better;
lower NARMA10 NMSE is better. IPC is not part of this experiment.

Ten conditions: empirical, binary, half-normal; empirical caps at Q99, Q95 and
Q90 with mean restoration; gamma and lognormal matched to realized mean and
population variance; heterogeneity a=0.5 and 0.75. Synthetic controls use fixed
mid-quantile vectors from SciPy distributions and preserve empirical ranks,
with random tie-breaking paired across conditions and fractions. They are not
iid sampled distributions. Moment matching also matches raw squared-weight
energy; it does not match the entire shape or minimum weight.

Own-radius scaling fixes each current radius. Shared-CE scaling freezes the
unmodified parent's multiplier at each fraction across all weight conditions.
Frobenius matching fixes the norm to that of the shared-scaled, unmodified CE
reference; actual spectral radii can differ. A matched norm is a sum-of-squares
constraint, not a claim of matched dynamical gain. Equal-norm cases use an exact
shared multiplier within the documented 1e-10 relative tolerance.

Baseline scores are reused only for identical matrices, inputs and scaling;
identical-matrix scaling copies are checked and recorded separately. Raw
metadata record both reuse flags and source shard hashes. Original experiments
and outputs are preserved. Refitting synthetic distribution parameters does not
use task scores, validation scores or test targets.

Win curves compare empirical ER against each condition at identical settings,
averaging scores across realizations before counting strict wins. Performance
AUCs and win-fraction AUCs are distinct, normalized by the observed E/I span.
Trapezoidal interpolation connects observed E/I points. No hyperparameter
configuration is discarded. The chosen grid and uniform E/I weighting are
analysis choices, not biological sampling distributions.

Intervals resample whole paired realizations jointly across all conditions,
fractions, settings and tasks: 1000 draws, seed 20261119. They are
pointwise descriptive percentile intervals with no multiplicity adjustment.
Inspect diagnostics for actual radii, applied factors, raw/scaled Frobenius
norms, weight variance, fitted shapes and changes in magnitude-weighted E/I.
