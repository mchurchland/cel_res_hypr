"""Audited MC and conventional NARMA10 for the new unified static sweep.

This is a new protocol; archived task functions/results remain unchanged.
Target generation reuses unmodified EchoTorch. Readouts use sklearn Ridge SVD,
training-only global RMS scaling, and held-out validation to select regularizers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import torch
from third_party.echotorch.NARMADataset import NARMADataset

GENERATOR_SHA256 = '07a9a4edf2ae6b7b70a2e97590172159b6fac516eab94462293747dbdf71256f'


class NarmaDivergenceError(ValueError):
    """An unusable generated target stream, independent of reservoir scores."""


@dataclass(frozen=True)
class TaskSettings:
    washout: int = 500
    train: int = 1500
    validation: int = 500
    test: int = 500
    narma_burnin: int = 1000
    mc_max_delay: int = 30
    mc_diagnostic_delay: int = 60
    ridge_alphas: tuple = (1e-8, 1e-6, 1e-4, 1e-2, 1.)

    def __post_init__(self):
        lengths = [self.washout, self.train, self.validation, self.test,
                   self.narma_burnin, self.mc_max_delay, self.mc_diagnostic_delay]
        if any(int(v) != v or v < 0 for v in lengths):
            raise ValueError('Task lengths/delays must be nonnegative integers')
        if min(self.train, self.validation, self.test) < 3:
            raise ValueError('Train/validation/test segments must each have at least 3 samples')
        if self.narma_burnin < 10:
            raise ValueError('NARMA burn-in must be at least 10')
        if not 1 <= self.mc_max_delay <= self.mc_diagnostic_delay <= self.washout:
            raise ValueError('Require 1 <= primary delay <= diagnostic delay <= washout')
        if not self.ridge_alphas or len(set(self.ridge_alphas)) != len(self.ridge_alphas) or any(not np.isfinite(a) or a <= 0 for a in self.ridge_alphas):
            raise ValueError('Ridge penalties must be unique, positive and finite')

    @property
    def length(self):
        return self.washout+self.train+self.validation+self.test


def conventional_narma(length, burnin, seed):
    """Call the pinned upstream class; align consumed u[t] with target y[t+1]."""
    import inspect
    if length < 1 or burnin < 10 or int(length) != length or int(burnin) != burnin:
        raise ValueError('Require a positive integer length and burn-in >= 10')
    source = Path(inspect.getfile(NARMADataset))
    if hashlib.sha256(source.read_bytes()).hexdigest() != GENERATOR_SHA256:
        raise ValueError('Upstream NARMA source changed; rerun the protocol audit')
    previous_dtype = torch.get_default_dtype()
    try:
        # Scope RNG/default precision changes; no GPU/global-generator dependence.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            torch.set_default_dtype(torch.float64)
            dataset = NARMADataset(length+burnin+1, n_samples=1, system_order=10)
            u, y = (a.detach().cpu().numpy() for a in dataset[0])
    finally:
        torch.set_default_dtype(previous_dtype)
    if not np.isfinite(u).all() or not np.isfinite(y).all():
        raise NarmaDivergenceError(f'Conventional NARMA generator diverged for seed {seed}')
    # Verify the defining recurrence, independently of upstream slice syntax.
    t = np.arange(9, len(u)-1)
    history = np.column_stack([y[t-i, 0] for i in range(10)]).sum(axis=1)
    expected = .3*y[t, 0]+.05*y[t, 0]*history+1.5*u[t-9, 0]*u[t, 0]+.1
    residual = float(np.max(np.abs(y[t+1, 0]-expected)))
    if residual > 1e-12:
        raise ValueError(f'Upstream generator violates the conventional recurrence: residual {residual}')
    return u[burnin:-1].copy(), y[burnin+1:].copy(), residual


def make_task_data(settings, seed):
    mc = np.random.default_rng(int(seed)+31_000_003).uniform(-1, 1, (settings.length, 1))
    rejected = []
    for attempt in range(100):
        generator_seed = int(seed)+37_000_003+1009*attempt
        try:
            u, y, residual = conventional_narma(settings.length, settings.narma_burnin, generator_seed)
            break
        except NarmaDivergenceError:
            rejected.append(generator_seed)
    else:
        raise ValueError('100 nonfinite NARMA streams; stop rather than changing the task')
    return dict(mc_input=mc.astype(np.float64), narma_input=u.astype(np.float64),
                narma_target=y.astype(np.float64), generator_residual=np.array(residual),
                narma_generator_seed=np.array(generator_seed),
                narma_rejected_seeds=np.asarray(rejected,dtype=np.int64))


def delay_targets(signal, max_delay):
    """Y[t,k-1]=u[t-k]. Warm-up history is available across split boundaries."""
    u = np.asarray(signal, dtype=float)
    if u.ndim != 2 or u.shape[1] != 1 or not np.isfinite(u).all() or not 1 <= max_delay < len(u):
        raise ValueError('Invalid scalar input or delay')
    result = np.full((len(u), max_delay), np.nan)
    for k in range(1, max_delay+1):
        result[k:, k-1] = u[:-k, 0]
    return result


def validated_ridge(x_train, y_train, x_validation, y_validation, x_test, alphas):
    """Only training and validation targets can influence fit or selection.

    A single training-derived RMS rescales ALL features together. This removes
    arbitrary global state-unit dependence of the ridge penalty without changing
    the relative weighting of neurons. No per-feature standardization or PCA.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (x_train, y_train, x_validation, y_validation, x_test)]
    xtr, ytr, xva, yva, xte = arrays
    if any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Readout data must be finite 2D arrays')
    if any(0 in a.shape for a in arrays) or len(xtr) != len(ytr) or len(xva) != len(yva) or ytr.shape[1] != yva.shape[1] or not xtr.shape[1] == xva.shape[1] == xte.shape[1]:
        raise ValueError('Readout shapes do not match')
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError('Invalid ridge penalties')
    center = xtr.mean(axis=0)
    scale = float(np.sqrt(np.mean((xtr-center)**2)))
    if scale == 0:
        scale = 1.
    xtr, xva, xte = ((a-center)/scale for a in (xtr, xva, xte))
    # sklearn supports one penalty per output. Tile targets across penalties so
    # its SVD solver factorizes the common design matrix just once.
    ordered = sorted(alphas, reverse=True)
    count = ytr.shape[1]
    model = Ridge(alpha=np.repeat(ordered, count), fit_intercept=True, solver='svd')
    model.fit(xtr, np.tile(ytr, (1, len(ordered))))
    validation_prediction = model.predict(xva).reshape(len(xva), len(ordered), count)
    losses = np.mean((validation_prediction-yva[:, None, :])**2, axis=0)
    # np.argmin retains the larger penalty on exact ties.
    chosen = np.argmin(losses, axis=0)
    best_loss = losses[chosen, np.arange(count)]
    all_prediction = model.predict(xte).reshape(len(xte), len(ordered), count)
    prediction = all_prediction[:, chosen, np.arange(count)]
    if not np.isfinite(prediction).all() or not np.isfinite(best_loss).all():
        raise ValueError('Nonfinite readout prediction or validation loss')
    return prediction, np.asarray(ordered)[chosen], best_loss, scale


def squared_correlations(target, prediction):
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Correlation arrays must have identical finite 2D shape')
    if (np.ptp(y, axis=0) == 0).any():
        raise ValueError('Constant target has no memory-correlation definition')
    scores = np.zeros(y.shape[1])
    varying = np.ptp(p, axis=0) > 0
    if varying.any():
        scores[varying] = pearsonr(y[:, varying], p[:, varying], axis=0).statistic**2
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1+1e-12).any():
        raise ValueError('Invalid squared correlations')
    return scores


def score_states(states, data, settings, task):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or len(states) != settings.length or not np.isfinite(states).all():
        raise ValueError('Require a complete finite state trajectory')
    start, train_end = settings.washout, settings.washout+settings.train
    val_end = train_end+settings.validation
    y = delay_targets(data['mc_input'], settings.mc_diagnostic_delay) if task == 'MC' else data['narma_target']
    if task not in ('MC', 'NARMA10'):
        raise ValueError('Unknown temporal task')
    prediction, alphas, val_mse, scale = validated_ridge(
        states[start:train_end], y[start:train_end], states[train_end:val_end], y[train_end:val_end],
        states[val_end:], settings.ridge_alphas)
    result = {f'{task}_readout_train_state_rms': scale}
    for name, a, b in [('train', start, train_end), ('validation', train_end, val_end), ('test', val_end, len(states))]:
        x = states[a:b]
        result[f'{task}_state_rms_{name}'] = float(np.sqrt(np.mean(x*x)))
        result[f'{task}_state_variance_{name}'] = float(np.var(x, axis=0).mean())
        result[f'{task}_saturation_{name}'] = float(np.mean(abs(x) > .9))
    if task == 'MC':
        curve = squared_correlations(y[val_end:], prediction)
        result.update(MC=float(curve[:settings.mc_max_delay].sum()), MC_extended=float(curve.sum()),
                      MC_tail=float(curve[settings.mc_max_delay:].sum()),
                      MC_validation_mse=float(val_mse.mean()), MC_ridge_alphas=json.dumps(alphas.tolist()))
        result.update({f'MC_delay_{k+1:02d}': float(v) for k, v in enumerate(curve)})
    else:
        variance = float(np.var(y[val_end:]))
        val_variance = float(np.var(y[train_end:val_end]))
        if min(variance, val_variance) <= 0:
            raise ValueError('NARMA targets have zero variance')
        mse = float(mean_squared_error(y[val_end:], prediction))
        baseline = np.full_like(y[val_end:], y[start:train_end].mean())
        result.update(NARMA10_test_nmse=mse/variance, NARMA10_test_nrmse=float(np.sqrt(mse/variance)),
                      NARMA10_test_rmse=float(np.sqrt(mse)), NARMA10_val_nmse=float(val_mse[0]/val_variance),
                      NARMA10_ridge_alpha=float(alphas[0]),
                      NARMA10_test_mean_baseline_nmse=float(mean_squared_error(y[val_end:], baseline)/variance))
    return result


@torch.no_grad()
def simulate_batch(weights, inputs, leaks, signal, washout, device='cpu', dtype=torch.float32, echo_initial=None):
    """Library batched matrix multiplication; no hand-written GPU recurrence.

    The secondary initial state is evolved during washout only, to report a
    finite-time initialization sensitivity diagnostic, not a proof of the ESP.
    """
    w = torch.as_tensor(np.asarray(weights), device=device, dtype=dtype)
    b = torch.as_tensor(np.asarray(inputs), device=device, dtype=dtype)
    leak = torch.as_tensor(np.asarray(leaks), device=device, dtype=dtype).reshape(-1, 1)
    u = torch.as_tensor(np.asarray(signal), device=device, dtype=dtype)
    if w.ndim != 3 or w.shape[1] != w.shape[2] or b.shape != w.shape[:2] or leak.shape != (len(w), 1) or u.ndim != 2 or u.shape[1] != 1:
        raise ValueError('Invalid recurrence batch dimensions')
    if not 0 < washout < len(u) or not all(torch.isfinite(a).all() for a in (w, b, leak, u)) or torch.any(leak <= 0) or torch.any(leak > 1):
        raise ValueError('Invalid recurrence values/washout')
    z = torch.zeros_like(b)
    second = torch.full_like(b, .5) if echo_initial is None else torch.as_tensor(echo_initial, device=device, dtype=dtype).expand_as(b).clone()
    if not torch.isfinite(second).all():
        raise ValueError('Nonfinite secondary initial condition')
    states = torch.empty((len(w), len(u), w.shape[1]), device=device, dtype=dtype)
    gap = None
    for t in range(len(u)):
        drive = b*u[t, 0]
        recurrent = torch.bmm(z.unsqueeze(1), w).squeeze(1)
        z = (1-leak)*z+leak*torch.tanh(recurrent+drive)
        states[:, t] = z
        if t < washout:
            recurrent_second = torch.bmm(second.unsqueeze(1), w).squeeze(1)
            second = (1-leak)*second+leak*torch.tanh(recurrent_second+drive)
            if t == washout-1:
                gap = torch.sqrt(torch.mean((z-second)**2, dim=1))
    if not torch.isfinite(states).all():
        raise ValueError('Nonfinite reservoir trajectory')
    return states.cpu().numpy(), gap.cpu().numpy()
