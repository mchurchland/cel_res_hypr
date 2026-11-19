"""Integrals of piecewise-linear performance curves on an explicit E/I axis."""
import numpy as np


def curve_areas(values, coordinates):
    """Return signed, positive and negative-magnitude areas along the last axis.

    Positive/negative areas split segments at interpolated zero crossings, rather
    than clipping samples before trapezoidal integration.
    """
    y = np.asarray(values, dtype=float)
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 1 or len(x) < 2 or y.ndim < 1 or y.shape[-1] != len(x):
        raise ValueError("Curve and coordinate dimensions do not match")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not (np.diff(x) > 0).all():
        raise ValueError("Finite curves and strictly increasing coordinates required")
    dx = np.diff(x)
    left, right = y[..., :-1], y[..., 1:]
    signed = ((left + right) * dx / 2).sum(axis=-1)

    def positive(a, b):
        segments = np.where((a >= 0) & (b >= 0), (a + b) * dx / 2, 0.)
        crossing = a * b < 0
        denominator = 2 * (np.abs(a) + np.abs(b))
        triangle = np.divide(np.maximum(a, b) ** 2 * dx, denominator,
                             out=np.zeros_like(segments), where=crossing)
        return (segments + triangle).sum(axis=-1)

    return signed, positive(left, right), positive(-left, -right)


def win_fraction_auc(differences, counts, coordinates):
    """Integrate mean-surface win percentages, resampling whole paired seeds.

    Differences have shape (seed, fraction, configuration). This integrates
    indicators AFTER averaging performance over seeds, as in the primary plot.
    """
    y = np.asarray(differences, dtype=float)
    counts = np.asarray(counts)
    if y.ndim != 3 or not y.size or not np.isfinite(y).all():
        raise ValueError("Complete seed-by-fraction-by-configuration differences required")
    if (counts.ndim != 2 or counts.shape[1] != y.shape[0] or len(counts) == 0
            or (counts < 0).any() or not np.all(counts.sum(axis=1) == y.shape[0])):
        raise ValueError("Invalid whole-seed bootstrap counts")
    mean = y.mean(axis=0)
    boot = (counts @ y.reshape(y.shape[0], -1) / y.shape[0]).reshape(len(counts), *y.shape[1:])
    result = {}
    for name, predicate in [("ce", lambda a: a > 0), ("er", lambda a: a < 0), ("tie", lambda a: a == 0)]:
        percentage = 100 * predicate(mean).mean(axis=-1)
        bootstrap_percentage = 100 * predicate(boot).mean(axis=-1)
        estimate = curve_areas(percentage, coordinates)[0]
        low, high = np.quantile(curve_areas(bootstrap_percentage, coordinates)[0], [.025, .975])
        result[f"{name}_win_auc_percent"] = float(estimate)
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = float(low), float(high)
    return result
