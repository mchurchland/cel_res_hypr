"""Weighted nulls and configurable continuous STP for the cluster follow-up.

Matrices use the repository convention W[source, target]. This intentionally
retains the exploratory abs(tanh activity) rate proxy, not a calibrated worm
voltage/release model. All STP types have BOTH utilization and depletion.
"""
from dataclasses import asdict

import numpy as np
from numba import njit

from network_stats.short_term_plasticity import STPParameters, _rk4_step


FAMILIES = ("ce_empirical", "ce_shuffled", "ce_binary", "er_empirical", "er_signed_empirical",
            "er_halfnormal", "er_binary")
NORMALIZATIONS = ("own_radius", "shared_ce_scale")
PARAMETER_SETS = {
    "fast": {"facilitating": STPParameters(.10, 30., 3.),
             "depressing": STPParameters(.35, 3., 30.)},
    "reference": {"facilitating": STPParameters(.15, 75., 5.),
                  "depressing": STPParameters(.45, 5., 75.)},
    "slow": {"facilitating": STPParameters(.20, 150., 10.),
             "depressing": STPParameters(.55, 10., 150.)},
}


def parameter_spec():
    return {name: {kind: asdict(p) for kind, p in preset.items()}
            for name, preset in PARAMETER_SETS.items()}


def condition_grid(parameter_sets, mixed_fractions):
    conditions = [("static", "none", 0.)]
    for preset in parameter_sets:
        if preset not in PARAMETER_SETS:
            raise ValueError(f"Unknown parameter set: {preset}")
        conditions.extend([(f"{preset}_facilitating", preset, 1.),
                           (f"{preset}_depressing", preset, 0.)])
        for fraction in mixed_fractions:
            if not np.isfinite(fraction) or not 0 < fraction < 1:
                raise ValueError("Mixed facilitating fractions must be strictly between 0 and 1.")
            conditions.append((f"{preset}_mixed_{fraction:g}", preset, fraction))
    if len({c[0] for c in conditions}) != len(conditions):
        raise ValueError("Duplicate STP conditions.")
    return conditions


def unsigned_controls(completed, er_mask, seed):
    """Exact CE magnitude multiset in CE-shuffled and ER-empirical, without replacement.

    Raw mean nonzero magnitude is one in every family. Own-radius scaling can
    change their absolute magnitude distributions differently; shared_ce_scale
    applies one CE-derived multiplier to every family at the same E/I fraction.
    """
    ce = np.abs(np.asarray(completed, dtype=np.float32))
    er = np.asarray(er_mask, dtype=bool)
    if ce.ndim != 2 or ce.shape[0] != ce.shape[1] or er.shape != ce.shape:
        raise ValueError("Require equal square CE and ER matrices.")
    if not np.isfinite(ce).all() or np.any(np.diag(ce)) or np.any(np.diag(er)):
        raise ValueError("Require finite loop-free matrices.")
    mask = ce != 0
    values = ce[mask].copy()
    if not len(values) or int(er.sum()) != len(values):
        raise ValueError("CE and ER must have the same positive number of edges.")
    values /= np.mean(values, dtype=np.float64)
    result = {}
    for family, target in (("ce_empirical", mask), ("ce_shuffled", mask),
                           ("ce_binary", mask), ("er_empirical", er),
                           ("er_halfnormal", er), ("er_binary", er)):
        w = np.zeros_like(ce)
        if family.endswith("binary"):
            v = np.ones(len(values), dtype=np.float32)
        elif family == "ce_empirical":
            v = values
        elif family == "er_halfnormal":
            v = np.abs(np.random.default_rng(seed + 43_000_003).normal(size=len(values)))
            v /= v.mean()
        else:
            offset = 41_000_003 if family == "ce_shuffled" else 42_000_003
            v = np.random.default_rng(seed + offset).permutation(values)
        w[target] = v
        if np.count_nonzero(w) != len(values):
            raise ValueError("A control lost edges.")
        result[family] = w
    return result


def spectral_radius(weights):
    radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(weights, dtype=float)))))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Cannot radius-normalize a zero/nilpotent or nonfinite matrix.")
    return radius


def match_signed_weights(ce_signed, er_signs, seed):
    """Match the entire signed CE weight multiset on an ER edge set.

    This controls total positive/negative weight as well as edge-sign counts.
    Magnitude placements can change across fractions to maintain that match.
    """
    ce = np.asarray(ce_signed, dtype=np.float32)
    signs = np.asarray(er_signs)
    if ce.shape != signs.shape or not np.isfinite(ce).all() or not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError("Invalid CE weights/ER signs.")
    w = np.zeros_like(ce)
    rng = np.random.default_rng(seed + 45_000_003)
    for sign in (-1, 1):
        values = ce[ce * sign > 0]
        mask = signs == sign
        if len(values) != int(mask.sum()):
            raise ValueError("CE/ER positive and negative edge counts must match.")
        w[mask] = rng.permutation(values)
    return w


def scale_matrix(raw, mode, target, ce_raw_radius):
    if mode not in NORMALIZATIONS or not np.isfinite(target) or target <= 0:
        raise ValueError("Invalid normalization or fixed initial radius target.")
    radius = spectral_radius(raw)
    reference = radius if mode == "own_radius" else ce_raw_radius
    if not np.isfinite(reference) or reference <= 1e-12:
        raise ValueError("Invalid CE reference radius.")
    factor = target / reference
    return (raw * factor).astype(np.float32), {
        "raw_rho": radius, "initial_scale": factor,
        "initial_post_rho": radius * factor, "ce_reference_raw_rho": ce_raw_radius,
    }


def stp_vectors(n, preset, fraction, assignment_order):
    order = np.asarray(assignment_order)
    if order.shape != (n,) or set(order.tolist()) != set(range(n)):
        raise ValueError("STP assignment must be a node permutation.")
    if preset not in PARAMETER_SETS or not 0 <= fraction <= 1:
        raise ValueError("Invalid STP preset or mixture.")
    facilitating = np.zeros(n, dtype=bool)
    facilitating[order[:int(round(n * fraction))]] = True
    fac, dep = (PARAMETER_SETS[preset][k] for k in ("facilitating", "depressing"))
    return tuple(np.where(facilitating, getattr(fac, k), getattr(dep, k)).astype(float)
                 for k in ("U", "tau_f", "tau_d")) + (facilitating,)


@njit(cache=True)
def _simulate(a, b, signal, U, tf, td, dynamic, fixed_gain, washout, train_end, substeps):
    n = len(a)
    h = np.zeros(n, np.float32)
    u = U.copy()
    x = np.ones(n, np.float64)
    transmitted = np.empty(n, np.float32)
    states = np.empty((len(signal), n), np.float32)
    training_gain = np.zeros(n, np.float64)
    gain_sum = 0.
    gain_sq = 0.
    gain_min = 1e30
    gain_max = -1e30
    for t in range(len(signal)):
        for j in range(n):
            gain = fixed_gain[j]
            if dynamic:
                for _ in range(substeps):
                    u[j], x[j] = _rk4_step(u[j], x[j], abs(h[j]), U[j], tf[j], td[j], 1. / substeps)
                gain = u[j] * x[j] / U[j]
            transmitted[j] = gain * h[j]
            if washout <= t < train_end:
                training_gain[j] += gain
            if t >= washout:
                gain_sum += gain
                gain_sq += gain * gain
                gain_min = min(gain_min, gain)
                gain_max = max(gain_max, gain)
        h = np.tanh(a @ transmitted + b * signal[t])
        states[t] = h
    samples = (len(signal) - washout) * n
    mean = gain_sum / samples
    stats = np.array([mean, np.sqrt(max(0., gain_sq / samples - mean * mean)), gain_min, gain_max])
    return states, stats, training_gain / (train_end - washout), u, x


def run_states(weights, input_weights, signal, vectors=None, washout=500,
               train_end=2000, fixed_gain=None, substeps=4):
    """Optional frozen gain is fitted on dynamic TRAINING samples only, per neuron.

    The frozen control matches coefficients derived from training, not the
    subsequently realized activity or gain in a separate static trajectory.
    """
    w, b, signal = (np.asarray(v, dtype=np.float32) for v in (weights, input_weights, signal))
    n = len(w)
    if w.shape != (n, n) or b.shape != (n, 1) or signal.ndim != 2 or signal.shape[1] != 1:
        raise ValueError("Require square W, N×1 input weights and T×1 signal.")
    if not all(np.isfinite(v).all() for v in (w, b, signal)):
        raise ValueError("Nonfinite simulation input.")
    if not 0 <= washout < train_end <= len(signal) or substeps < 1 or int(substeps) != substeps:
        raise ValueError("Invalid segments/integration substeps.")
    dynamic = vectors is not None
    if dynamic and fixed_gain is not None:
        raise ValueError("Dynamic STP and frozen gain are separate conditions.")
    if dynamic:
        U, tf, td = (np.asarray(v, dtype=float) for v in vectors[:3])
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in (U, tf, td)):
            raise ValueError("Invalid per-neuron STP vectors.")
        if np.any(U <= 0) or np.any(U > 1) or np.any(tf <= 0) or np.any(td <= 0):
            raise ValueError("STP requires 0<U<=1 and positive time constants.")
    else:
        U, tf, td = np.full(n, .15), np.ones(n), np.ones(n)
    gain = np.ones(n) if fixed_gain is None else np.asarray(fixed_gain, dtype=float)
    if gain.shape != (n,) or not np.isfinite(gain).all() or np.any(gain < 0):
        raise ValueError("Invalid frozen gain.")
    states, stats, training_gain, u, x = _simulate(
        np.ascontiguousarray(w.T), np.ascontiguousarray(b[:, 0]), np.ascontiguousarray(signal[:, 0]),
        U, tf, td, dynamic, gain, washout, train_end, substeps)
    if not np.isfinite(states).all() or np.any(u < -1e-8) or np.any(u > 1 + 1e-8) or np.any(x < -1e-8) or np.any(x > 1 + 1e-8):
        raise ValueError("Nonfinite states or STP integration left physical bounds; no clipping.")
    details = dict(zip(("gain_mean", "gain_sd", "gain_min", "gain_max"), stats.tolist()))
    details.update(saturation_fraction=float(np.mean(abs(states[washout:]) > .9)),
                   mean_state_variance=float(np.var(states[washout:].astype(float), axis=0).mean()))
    return states, details, training_gain
