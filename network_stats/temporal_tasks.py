"""MC and NARMA10 evaluation for the focused E/I crossover experiment.

The reservoir consumes u[t] before predicting NARMA y[t+1]. Only reservoir
states enter the linear readout; target outputs are never fed back.
"""
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from network_stats.run_one import run_reservoir_states
from network_stats.stats import compute_MC


@dataclass(frozen=True)
class TemporalTaskSettings:
    washout: int = 500
    train: int = 1500
    validation: int = 500
    test: int = 500
    mc_max_delay: int = 30
    mc_ridge_alpha: float = 1e-4
    narma_burnin: int = 1000
    narma_ridge_alphas: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)

    def __post_init__(self):
        if self.washout < 0 or self.narma_burnin < 0:
            raise ValueError("Washout and target burn-in must be non-negative.")
        if min(self.train, self.validation, self.test, self.mc_max_delay) <= 0:
            raise ValueError("Segment lengths and MC delay must be positive.")
        if min(self.train, self.test) <= self.mc_max_delay + 1:
            raise ValueError("MC train/test segments must exceed max_delay + 1.")
        alphas = (self.mc_ridge_alpha, *self.narma_ridge_alphas)
        if not self.narma_ridge_alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError("Ridge penalties must be finite and positive.")


def make_task_data(settings: TemporalTaskSettings, seed: int) -> dict[str, torch.Tensor]:
    """CPU-generated signals are shared across models, fractions and grids."""
    # Use the installed package, not a local reimplementation or copied recurrence.
    from reservoirpy.datasets import narma

    mc_rng = np.random.default_rng(seed + 31_000_003)
    mc_length = settings.washout + settings.train + settings.test
    mc_u = mc_rng.uniform(-1, 1, mc_length).astype(np.float32)
    length = settings.washout + settings.train + settings.validation + settings.test
    u_full, y_full = narma(
        n_timesteps=length + settings.narma_burnin, order=10,
        a1=.3, a2=.05, b=1.5, c=.1, seed=seed + 37_000_003,
    )
    # ReservoirPy returns inputs of length n+10 and targets y[10:] of length n.
    # Input at index t-1 is consumed before predicting target y[t].
    aligned_input = u_full[9:-1]
    if aligned_input.shape != y_full.shape or not np.isfinite(y_full).all():
        raise ValueError("Invalid/divergent ReservoirPy NARMA10 data; no resampling performed.")
    start = settings.narma_burnin
    return {
        "mc_input": torch.from_numpy(mc_u[:, None]),
        "narma_input": torch.from_numpy(aligned_input[start:].astype(np.float32)),
        "narma_target": torch.from_numpy(y_full[start:]),
    }


def _nmse(target: np.ndarray, prediction: np.ndarray) -> float:
    variance = np.var(target)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("NMSE requires finite targets with positive variance.")
    return float(mean_squared_error(target, prediction) / variance)


@torch.no_grad()
def fit_narma_readout(
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_validation: torch.Tensor, y_validation: torch.Tensor,
    x_test: torch.Tensor, y_test: torch.Tensor,
    alphas: tuple[float, ...],
) -> dict[str, float]:
    """Select ridge using validation only; score the selected model on test.

    Uses scikit-learn Ridge with a fitted intercept and float64 arrays.
    The chosen readout is not refit on validation.
    """
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge penalties must be finite and positive.")
    tensors = [x_train, y_train, x_validation, y_validation, x_test, y_test]
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError("Readout inputs and targets must be finite.")
    xtr, ytr, xva, yva, xte, yte = [t.detach().cpu().double().numpy() for t in tensors]
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky").fit(xtr, ytr)
        loss = _nmse(yva, model.predict(xva))
        if best is None or loss < best[0]:
            best = (loss, float(alpha), model)
    val_nmse, alpha, model = best
    test_prediction = model.predict(xte)
    test_nmse = _nmse(yte, test_prediction)
    mse = mean_squared_error(yte, test_prediction)
    result = {
        "NARMA10_val_nmse": val_nmse,
        "NARMA10_test_nmse": test_nmse,
        "NARMA10_test_nrmse": test_nmse ** .5,
        "NARMA10_test_rmse": mse ** .5,
        "NARMA10_ridge_alpha": alpha,
        "NARMA10_test_mean_baseline_nmse": _nmse(yte, np.full_like(yte, ytr.mean())),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite NARMA10 score.")
    return result


@torch.no_grad()
def evaluate_temporal_tasks(
    weights: torch.Tensor, input_weights: torch.Tensor, leak: float,
    data: dict[str, torch.Tensor], settings: TemporalTaskSettings,
) -> dict[str, float]:
    """Run MC and NARMA10 trajectories; never evaluate IPC/KR/GR."""
    device = weights.device
    mc_u = data["mc_input"].to(device=device, dtype=weights.dtype)
    mc_states = run_reservoir_states(weights, input_weights, mc_u, leak)
    train_start, train_end = settings.washout, settings.washout + settings.train
    # Keep the existing per-delay alignment, ridge epsilon and corr^2 scoring.
    # Float64 readout avoids ill-conditioned float32 normal-equation solves.
    mc, _ = compute_MC(
        mc_states[train_start:train_end].double(), mc_states[train_end:].double(),
        mc_u[train_start:train_end].double(), mc_u[train_end:].double(),
        settings.mc_max_delay, settings.mc_ridge_alpha, device,
    )
    narma_u = data["narma_input"].to(device=device, dtype=weights.dtype)
    target = data["narma_target"].to(device=device, dtype=torch.float64)
    states = run_reservoir_states(weights, input_weights, narma_u, leak)
    val_end = train_end + settings.validation
    result = fit_narma_readout(
        states[train_start:train_end], target[train_start:train_end],
        states[train_end:val_end], target[train_end:val_end],
        states[val_end:], target[val_end:], settings.narma_ridge_alphas,
    )
    result["MC"] = float(mc)
    if not np.isfinite(mc):
        raise ValueError("Nonfinite MC score.")
    return result
