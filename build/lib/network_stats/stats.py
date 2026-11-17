from network_stats.stats_util import (
    corr2_score,
    legendre_P,
    effective_rank_from_states,
    ridge_fit_predict,
)
from torch import Tensor
import torch
import numpy as np


def compute_IPC(
    Xtr: Tensor,
    Xte: Tensor,
    utr: Tensor,
    ute: Tensor,
    max_delay: int,
    alpha: float,
    device: torch.device,
    orders: list[int] = [1, 3, 5],
    capacity_threshold: float = 0.0,
) -> float:
    """Sum squared correlations for single-delay Legendre targets.

    For each d=1..max_delay and k in orders, the target is P_k(u[t-d]).
    Xtr/Xte are [time, features]; utr/ute are scalar streams [time] or
    [time,1]. Alignment uses states[d:] and inputs[:-d] independently in
    the training and test segments. This restricted IPC omits products
    spanning multiple delays. One torch.linalg.solve handles all polynomial
    orders at a given delay, using ridge regularization alpha + 1e-6.
    Terms below capacity_threshold are excluded; scores are correlation
    squared rather than normalized mean-squared-error capacities.
    """
    total = 0.0

    for d in range(1, max_delay + 1):
        # shared slices for this delay
        base_tr = utr[:-d]  # [T_train - d, 1] or [T_train - d]
        base_te = ute[:-d]  # [T_test  - d, 1] or [T_test  - d]
        Xtr_d = Xtr[d:]  # [T_train - d, N]
        Xte_d = Xte[d:]  # [T_test  - d, N]

        # build all Legendre targets for orders k=1..max_order
        ytr_cols = []
        yte_cols = []
        for k in orders:
            ytr_k = legendre_P(base_tr, k)
            yte_k = legendre_P(base_te, k)

            # ensure 2D [T, 1]
            if ytr_k.dim() == 1:
                ytr_k = ytr_k.unsqueeze(1)
            if yte_k.dim() == 1:
                yte_k = yte_k.unsqueeze(1)

            ytr_cols.append(ytr_k)
            yte_cols.append(yte_k)

        # [T_train - d, max_order], [T_test - d, max_order]
        Ytr = torch.cat(ytr_cols, dim=1).to(device)
        Yte = torch.cat(yte_cols, dim=1).to(device)

        # ONE ridge solve for all orders k at this delay
        Yhat = ridge_fit_predict(Xtr_d, Ytr, Xte_d, alpha, DEVICE=device)
        # Yhat: [T_test - d, max_order]

        for j in range(Ytr.shape[1]):
            score = corr2_score(Yte[:, j], Yhat[:, j])
            if score >= capacity_threshold:
                total += score

    return float(total)


def compute_KR(X: Tensor) -> float:
    """Return centered Shannon effective rank of independent-input states."""
    return effective_rank_from_states(X)


def compute_GR(X: Tensor) -> float:
    """Return centered Shannon effective rank of related-input states."""
    return effective_rank_from_states(X)


def compute_MC(
    Xtr: Tensor,
    Xte: Tensor,
    utr: Tensor,
    ute: Tensor,
    max_delay: int,
    alpha: float,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Return total linear memory capacity and its float32 per-delay scores.

    Xtr/Xte are [time, features]; utr/ute are scalar streams [time] or
    [time,1]. For tau=1..max_delay, regress utr[:-tau] on Xtr[tau:] and
    score ute[:-tau] against predictions from Xte[tau:]. Segments are aligned
    separately, so no target crosses the train/test boundary. The readout has
    no fitted intercept; Pearson scoring centers the target and prediction.
    """
    capacities = []
    for tau in range(1, max_delay + 1):
        ytr = utr[:-tau]
        yte = ute[:-tau]
        Xtr_d = Xtr[tau:]  ##so that they are the same dimensiobn
        Xte_d = Xte[tau:]  ##so that they are the same dimesions
        yhat = ridge_fit_predict(Xtr_d, ytr, Xte_d, alpha, DEVICE=device)
        capacities.append(corr2_score(yte, yhat))
    capacities = np.array(capacities, dtype=np.float32)
    return float(np.sum(capacities)), capacities


def compute_GR_pilati(X_similar: Tensor, threshold: float = 1e-3) -> int:
    """Return thresholded RCbench rank of related-input final states."""
    return _relative_svd_rank_pilati(X_similar, threshold, "X_similar")


def compute_KR_pilati(X_states: Tensor, threshold: float = 1e-3) -> int:
    """Return thresholded RCbench rank of independent-input final states."""
    return _relative_svd_rank_pilati(X_states, threshold, "X_states")


def _relative_svd_rank_pilati(X: Tensor, threshold: float, name: str) -> int:
    """Count singular values above a fraction of the largest singular value."""
    if X.dim() != 2:
        raise ValueError(f"{name} must be a 2D state matrix.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    singular_values = torch.linalg.svdvals(X)
    if singular_values.numel() == 0:
        return 0
    s_max = singular_values.max()
    if not torch.isfinite(s_max) or s_max <= 0:
        return 0
    return int(torch.sum(singular_values > threshold * s_max).item())
