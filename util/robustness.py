"""Robustness of a finite hyperparameter grid; no reservoir simulations.

Dispersion uses population SD (ddof=0): configurations are the specified finite
design, not independent draws from a biological population. Missing values are
excluded explicitly and counted. CV and relative retention are undefined for
nonpositive means/maxima, respectively. Negative values remain in absolute metrics.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PARAMETERS = ("rho_target", "leak", "input_scale", "neuron_bias")
COORDINATES = {"rho_target": "log", "leak": "linear", "input_scale": "log", "neuron_bias": "linear"}


def performance_metrics(values, thresholds=()):
    """Summarize one finite surface. Bottom decile includes ceil(.1*n) points.

    Quantiles use NumPy's linear interpolation. MAD is unscaled. Thresholds
    operate in each performance metric's original units. Retention is inclusive.
    """
    raw = np.asarray(values, dtype=float).ravel()
    x = raw[np.isfinite(raw)]
    names = ("mean", "cv", "sd", "iqr", "mad", "minimum", "p10", "bottom10", "maximum", "retention80", "retention90", "range", "variance", "floor_fraction")
    result = dict.fromkeys(names, np.nan)
    result.update(n_total=len(raw), n_valid=len(x), n_nonfinite=len(raw)-len(x), n_negative=int((x < 0).sum()))
    if len(x):
        mean, sd, maximum = x.mean(), x.std(ddof=0), x.max()
        q10, q25, median, q75 = np.quantile(x, [.1, .25, .5, .75], method="linear")
        result.update(mean=mean, sd=sd, cv=sd/mean if mean > 0 else np.nan,
                      iqr=q75-q25, mad=np.median(np.abs(x-median)), minimum=x.min(),
                      p10=q10, bottom10=np.sort(x)[:int(np.ceil(.1*len(x)))].mean(),
                      maximum=maximum, range=maximum-x.min(), variance=sd**2,
                      retention80=np.mean(x >= .8*maximum) if maximum > 0 else np.nan,
                      retention90=np.mean(x >= .9*maximum) if maximum > 0 else np.nan,
                      floor_fraction=np.mean(x <= 0))
    for name, threshold in thresholds:
        if not np.isfinite(threshold):
            raise ValueError("Thresholds must be finite")
        result[name] = np.mean(x >= threshold) if len(x) else np.nan
    return result


def threshold_attainment(values, thresholds):
    x = np.asarray(values, float).ravel()
    x = x[np.isfinite(x)]
    t = np.asarray(thresholds, float)
    if not np.isfinite(t).all():
        raise ValueError("Thresholds must be finite")
    return np.mean(x[:, None] >= t, axis=0) if len(x) else np.full(t.shape, np.nan)


def grid_edges(grid: pd.DataFrame, coordinates=None):
    """Adjacent *global design levels*, holding every other coordinate fixed.

    Coordinates span [0,1] after log/linear transformation. Missing intermediate
    grid points never create a spurious long edge. Returns index positions, so
    reset/reorder the grid before calling. Duplicate coordinates are an error.
    """
    coordinates = COORDINATES if coordinates is None else coordinates
    if grid[list(PARAMETERS)].duplicated().any():
        raise ValueError("Duplicate grid coordinates")
    if not np.isfinite(grid[list(PARAMETERS)].to_numpy(float)).all():
        raise ValueError("Nonfinite grid coordinates")
    lookup = {tuple(row): i for i, row in enumerate(grid[list(PARAMETERS)].to_numpy(float))}
    result = {}
    for dim, name in enumerate(PARAMETERS):
        levels = np.sort(grid[name].unique().astype(float))
        transformed = levels.copy()
        if coordinates[name] == "log":
            if np.any(levels <= 0):
                raise ValueError(f"Log coordinates require positive {name}")
            transformed = np.log(levels)
        elif coordinates[name] != "linear":
            raise ValueError(f"Unknown coordinate transformation: {coordinates[name]}")
        start, end, distance = [], [], []
        span = np.ptp(transformed)
        if len(levels) > 1:
            for row, idx in lookup.items():
                level_idx = np.searchsorted(levels, row[dim])
                if level_idx == len(levels)-1:
                    continue
                other = list(row)
                other[dim] = levels[level_idx+1]
                if tuple(other) in lookup:
                    start.append(idx)
                    end.append(lookup[tuple(other)])
                    distance.append((transformed[level_idx+1]-transformed[level_idx])/span)
        result[name] = (np.asarray(start, int), np.asarray(end, int), np.asarray(distance, float))
    return result


def local_sensitivity(values, edges):
    """Mean and 90th percentile absolute finite difference per dimension."""
    x = np.asarray(values, float)
    result = {}
    all_slopes = []
    for name, (left, right, distance) in edges.items():
        valid = np.isfinite(x[left]) & np.isfinite(x[right])
        slopes = np.abs(x[right[valid]]-x[left[valid]])/distance[valid]
        result[f"sensitivity_{name}_mean"] = slopes.mean() if len(slopes) else np.nan
        result[f"sensitivity_{name}_p90"] = np.quantile(slopes, .9) if len(slopes) else np.nan
        result[f"edges_{name}"] = len(slopes)
        if len(slopes):
            all_slopes.append(slopes)
    result["sensitivity_mean"] = np.concatenate(all_slopes).mean() if all_slopes else np.nan
    return result


def variance_decomposition(y):
    """Exact finite balanced-design ANOVA using population variances.

    Y[r,c] = grand + seed[r] + configuration[c] + residual[r,c]. With one
    observation per cell, residual includes interaction AND evaluation noise.
    This is descriptive variance partitioning, not random-effect estimation.
    """
    y = np.asarray(y, float)
    if y.ndim != 2 or min(y.shape) < 2 or not np.isfinite(y).all():
        raise ValueError("Requires a finite, balanced matrix with >=2 rows and columns")
    grand = y.mean()
    seed = y.mean(axis=1)-grand
    config = y.mean(axis=0)-grand
    residual = y-grand-seed[:, None]-config[None, :]
    return dict(total_variance=y.var(), configuration_variance=config.var(),
                seed_variance=seed.var(), interaction_noise_variance=np.mean(residual**2),
                within_configuration_variance=y.var(axis=0).mean(),
                mean_within_seed_variance=y.var(axis=1).mean(),
                n_seeds=y.shape[0], n_configurations=y.shape[1])


def feasible_variance(mean, lower, upper):
    """For bounded P, Var(P) <= (mean-lower)*(upper-mean).

    Follows directly from E[(P-lower)*(upper-P)] >= 0. This is a ceiling,
    not a model of the variance expected under a null hypothesis.
    """
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Require finite lower < upper")
    mean = np.asarray(mean, float)
    return np.where((mean >= lower) & (mean <= upper), (mean-lower)*(upper-mean), np.nan)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    a, b = rankdata(x[valid]), rankdata(y[valid])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def cluster_spearman(frame, x, y, cluster="seed", n_boot=2000, rng=None):
    """Resample whole seed blocks; architecture panel stays fixed.

    All observations sharing a seed are resampled together. Percentile intervals
    quantify seed sampling uncertainty conditional on these architectures/grid.
    No configuration is resampled as if it were an independent network.
    """
    rng = np.random.default_rng(20261116) if rng is None else rng
    d = frame[[cluster, x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = [g[[x, y]].to_numpy() for _, g in d.groupby(cluster)]
    estimate = spearman(d[x], d[y])
    if len(groups) < 3 or not np.isfinite(estimate):
        return dict(rho=estimate, ci_low=np.nan, ci_high=np.nan, n=len(d), n_clusters=len(groups), n_boot_valid=0)
    draws = []
    # Vectorize the common balanced-panel case, ranking again in each draw.
    if len({len(g) for g in groups}) == 1:
        array = np.stack(groups)
        for offset in range(0, n_boot, 64):
            take = rng.integers(len(groups), size=(min(64, n_boot-offset), len(groups)))
            v = array[take].reshape(len(take), -1, 2)
            ranks = rankdata(v, axis=1)
            ranks -= ranks.mean(axis=1, keepdims=True)
            numerator = (ranks[:, :, 0]*ranks[:, :, 1]).sum(axis=1)
            denominator = np.sqrt((ranks[:, :, 0]**2).sum(axis=1)*(ranks[:, :, 1]**2).sum(axis=1))
            draws.extend(np.divide(numerator, denominator, out=np.full(len(take), np.nan), where=denominator > 0))
    else:
        for _ in range(n_boot):
            v = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            draws.append(spearman(v[:, 0], v[:, 1]))
    finite = np.asarray(draws)[np.isfinite(draws)]
    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return dict(rho=estimate, ci_low=lo, ci_high=hi, n=len(d), n_clusters=len(groups), n_boot_valid=len(finite))


def bootstrap_mean(values, n_boot=2000, rng=None):
    """One observation per seed; all columns resampled jointly."""
    rng = np.random.default_rng(20261116) if rng is None else rng
    a = np.asarray(values, float)
    if a.ndim == 1:
        a = a[:, None]
    draws = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for offset in range(0, n_boot, 64):
            take = rng.integers(len(a), size=(min(64, n_boot-offset), len(a)))
            draws.append(np.nanmean(a[take], axis=1))
        draws = np.concatenate(draws)
        return np.nanmean(a, axis=0), np.nanquantile(draws, [.025, .975], axis=0)
