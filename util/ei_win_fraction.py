"""Matched finite-grid win fractions; configurations are not replicates."""
from __future__ import annotations

import numpy as np
import pandas as pd

PARAMETERS = ["rho_target", "leak", "input_scale", "neuron_bias"]
KEYS = ["normalization", "negative_fraction", "seed", *PARAMETERS]


def pair_surfaces(ce, er, metrics):
    """Match exact seed/grid identities and explicitly audit unmatched rows."""
    for name, frame in [("CE", ce), ("ER", er)]:
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{name}: missing pairing identity")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name}: duplicate pairing identity")
    joined = ce[KEYS + list(metrics)].merge(
        er[KEYS + list(metrics)], on=KEYS, how="outer", suffixes=("_ce", "_er"),
        indicator=True, validate="one_to_one")
    audit = (joined.groupby(["normalization", "negative_fraction", "_merge"], observed=True)
             .size().rename("n_rows").reset_index())
    paired = joined[joined._merge == "both"].drop(columns="_merge").copy()
    for metric in metrics:
        if not np.isfinite(paired[[f"{metric}_ce", f"{metric}_er"]].to_numpy(float)).all():
            raise ValueError(f"{metric}: nonfinite paired performance")
        paired[metric] = paired[f"{metric}_ce"] - paired[f"{metric}_er"]
    return paired, audit


def bootstrap_counts(n_seeds, n_boot, rng):
    return rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=n_boot)


def summarize_surface(differences, counts):
    """Rows are paired seeds, columns are fixed configurations.

    Primary: proportion of configurations with a positive mean paired difference.
    Secondary: mean per-seed fraction of positive configuration differences.
    Bootstrap whole seed rows, retaining every configuration together.
    """
    y = np.asarray(differences, dtype=float)
    counts = np.asarray(counts)
    if y.ndim != 2 or not y.size or not np.isfinite(y).all():
        raise ValueError("A complete finite seed-by-configuration matrix is required")
    if counts.ndim != 2 or counts.shape[1] != y.shape[0] or len(counts) == 0:
        raise ValueError("Bootstrap counts do not match seeds")
    if (counts < 0).any() or not np.all(counts.sum(axis=1) == y.shape[0]):
        raise ValueError("Each bootstrap draw must resample exactly n_seeds rows")
    means = y.mean(axis=0)
    boot_means = counts @ y / len(y)
    boot_wins = 100 * (boot_means > 0).mean(axis=1)
    per_seed = 100 * (y > 0).mean(axis=1)
    boot_seed_wins = counts @ per_seed / len(y)
    lo, hi = np.quantile(boot_wins, [.025, .975])
    seed_lo, seed_hi = np.quantile(boot_seed_wins, [.025, .975])
    return dict(n_seeds=len(y), n_configurations=y.shape[1],
                n_ce_better=int((means > 0).sum()), n_er_better=int((means < 0).sum()),
                n_ties=int((means == 0).sum()), percent_ce_better=100 * (means > 0).mean(),
                ci_low=lo, ci_high=hi,
                mean_seed_win_percent=per_seed.mean(), seed_ci_low=seed_lo, seed_ci_high=seed_hi), means, (boot_means > 0).mean(axis=0), per_seed


def surface_matrix(group, metric, seeds):
    grid = group[PARAMETERS].drop_duplicates().sort_values(PARAMETERS)
    columns = pd.MultiIndex.from_frame(grid)
    matrix = group.pivot(index="seed", columns=PARAMETERS, values=metric).reindex(index=seeds, columns=columns)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Incomplete paired grid: refusing to silently change denominators")
    return matrix.to_numpy(float), grid.reset_index(drop=True)
