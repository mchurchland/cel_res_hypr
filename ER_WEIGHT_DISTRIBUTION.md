# Paired ER weight-distribution controls

From the updated Thesis repository on Orca:

```bash
sbatch r_er_weight_distribution.sbatch
```

This submits GPU preflight, a GPU array and dependent merge/analysis. It does not
submit production jobs until you run this command. Each simulation task requests
one GPU, four CPUs, 24 GB and a 12-hour scheduling limit. Defaults: 100 paired
realizations, 20 array tasks, MC and conventional NARMA10, zero bias, the existing
24 radius/leak/input settings, and **all eleven inhibitory fractions 0–100% in
10% increments**. The two heterogeneity levels are a=0.5 and a=0.75; they do not
restrict the E/I sweep to two fractions. IPC is not included in this mechanism
experiment.

Ten weight conditions use the exact ER masks, signs and input projection cached
by completed run 271871: empirical CE magnitudes; binary; archived half-normal;
empirical caps at Q99, Q95 and Q90; moment-matched gamma and lognormal;
heterogeneity compression `mu + a*(w-mu)` at a=0.5 and a=0.75.

Caps are computed using NumPy's linear quantile convention, before restoring the
empirical mean. Record both the pre-restoration cutoff and restoration factor:
mean restoration can raise the final maximum above the original cutoff. The
compression preserves mean and scales population variance by a squared.

Gamma/lognormal controls match the **actual finite-vector mean and population
variance** (ddof=0), not just theoretical moments. SciPy `gamma.ppf` /
`lognorm.ppf` supplies a fixed vector at mid-quantiles `(j+0.5)/M`; `brentq`
calibrates the shape, then positive scale restores the mean. These are fixed
quantile representations of the distributions, not iid weight samples. Parameter
fitting never uses task performance. Constant empirical magnitudes give a
constant matched vector. Invalid or unrepresentable distributions fail loudly;
there is no clipping to zero, negative magnitude or silent sign change.

Synthetic values are placed by empirical magnitude rank, with deterministic
seeded tie-breaking. The same vector is used across fractions; signs are taken
from each parent's mask. This holds rank placement fixed where empirical ranks
are distinct. Minimum magnitude and magnitude-weighted E/I are not separately
matched; both are reported as diagnostics. Gamma/lognormal shape matching tests
the distribution beyond mean/variance, not the upper tail alone.

Three scaling branches are included:

- `own_radius`: factor `target / current_raw_radius`.
- `shared_ce_scale`: factor `target / parent_CE_radius_at_current_fraction`,
  identical across all weight conditions and frozen to the unmodified parent.
- `frobenius_match`: the shared factor times `parent_CE_raw_F / control_raw_F`.
  Thus every condition matches the Frobenius norm of the shared-scaled empirical
  CE reference at that fraction. Actual spectral radii are free to differ.

The grid's radius coordinate is a reference scaling parameter in the latter two
branches, not necessarily the ER reservoir's actual radius. Mean/variance-matched
controls already match raw F; their Frobenius and shared branches therefore
coincide within a documented relative tolerance of 1e-10. Equal-norm factors are
canonicalized to the exact shared factor, allowing bitwise-identical matrices
to reuse scores. A matched Frobenius norm does not establish matched dynamical
gain or isolate a spectral mechanism.

The raw grid has **792,000 rows**. Up to 158,400 baseline rows are inherited
from the parent without rerunning MC/NARMA; 79,200 further rows copy identical
shared/Frobenius conditions. Default ordering gives **554,400 newly evaluated
conditions, or 1,108,800 trajectories**. Reuse requires verified raw matrices,
input projections and scaling; parent task streams are read from checked caches.
Fresh CPU/GPU scores are checked against inherited baseline scores in preflight.
No reference data are modified. `baseline_scores_reused` and
`scaling_scores_reused` record these operations; the old `MC_NARMA_reused` field
is reserved for the previous IPC-backfill mechanism and remains zero here.

Useful overrides:

```bash
TOTAL_REPEATS=20 ARRAY_TASKS=10 sbatch r_er_weight_distribution.sbatch
```

`PARENT_RUN_DIR` defaults to
`results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871` and must
contain the original manifest, merged CSV/checksum, task/network caches and
baseline shards. `CONDA_ENV` defaults to `cel`; `PYTHON_BIN`, `BATCH_SIZE`,
`BOOTSTRAP`, `PARTITION`, `OUTROOT` and `ANALYSIS_DIR` follow the existing jobs.
Extra arguments to the sbatch file go to `prepare`, for example
`--fractions 0 .2 .5 1` or `--families empirical gamma lognormal`.

Results:
`results/er_weight_distribution/er_weights_<submission-job-ID>/`

PNG figures, CSV summaries and report:
`analysis/er_weight_distribution/er_weights_<submission-job-ID>/`

Analysis includes mean-performance curves, empirical-versus-control matched
setting win fractions, configuration advantages, both kinds of AUC, actual
spectral radii, raw/scaled F, fitted moments and score-reuse counts. All uncertainty
resamples whole paired realizations jointly; the fixed grid is not an independent
biological sample. Existing results, snapshots and figures are preserved. Resume
an unchanged run using the same RUN_ID and settings; changed sources/settings
require a new run ID. There is no runtime prediction beyond the scheduling limits.

Library methods: [SciPy gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html),
[lognormal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html),
[Brent root finder](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html).
