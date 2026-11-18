"""Paired curve summaries and descriptive, non-extrapolated crossover estimates."""
import numpy as np
from scipy.stats import bootstrap


def bootstrap_panel(values, n_boot=2000, seed=271601):
    """Resample complete realization rows, preserving every paired condition.

    Return mean and shared bootstrap draws with bootstrap index first. SciPy
    handles resampling; flattening retains all within-realization dependence.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim < 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Require at least two finite complete realization rows.")
    result = bootstrap(
        (a.reshape(len(a), -1),), np.mean, axis=0, vectorized=True,
        n_resamples=n_boot, batch=16, method="percentile",
        rng=np.random.default_rng(seed),
    )
    draws = np.moveaxis(result.bootstrap_distribution, -1, 0)
    return a.mean(axis=0), draws.reshape((n_boot, *a.shape[1:]))


def curve_crossings(x, y):
    """Find all sign reversals, using linear interpolation inside sampled edges.

    Positive y means CE is favored. Isolated zero observations are classified
    by the nearest nonzero values; plateaus/tangencies are not sign reversals.
    No root is inferred outside the tested domain.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Require equal 1D curves of length >=2.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Require finite curves and strictly increasing coordinates.")
    roots = []
    nonzero = np.flatnonzero(y != 0)
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        if y[left] * y[right] >= 0:
            continue
        if right - left > 1:
            # A multi-point equality plateau does not identify a unique root.
            if right - left != 2:
                continue
            root = x[left + 1]
        else:
            root = x[left] - y[left] * (x[right] - x[left]) / (y[right] - y[left])
        roots.append(dict(root=float(root), bracket_low=float(x[left]), bracket_high=float(x[right]),
                          direction="CE_to_ER" if y[left] > 0 else "ER_to_CE"))
    return roots


def summarize_crossing(x, mean_curve, bootstrap_curves):
    """Bootstrap the unique interpolated root; retain failures and multiplicity.

    Intervals are conditional on bootstrap curves having exactly one crossing.
    They quantify seed uncertainty of the interpolant, not uncertainty about
    its between-point shape or evidence of biological E/I optimization.
    """
    roots = curve_crossings(x, mean_curve)
    draws = [curve_crossings(x, y) for y in bootstrap_curves]
    unique = np.asarray([r[0]["root"] for r in draws if len(r) == 1])
    result = dict(n_mean_crossings=len(roots), mean_crossings=roots,
                  bootstrap_unique_fraction=float(np.mean([len(r) == 1 for r in draws])),
                  bootstrap_no_crossing_fraction=float(np.mean([len(r) == 0 for r in draws])),
                  bootstrap_multiple_fraction=float(np.mean([len(r) > 1 for r in draws])),
                  root=np.nan, bracket_low=np.nan, bracket_high=np.nan,
                  ci_low=np.nan, ci_high=np.nan, direction="none")
    if len(roots) == 1:
        result.update(roots[0])
        if len(unique):
            result["ci_low"], result["ci_high"] = np.quantile(unique, [.025, .975])
    return result
