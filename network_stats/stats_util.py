from torch import Tensor
import torch


def legendre_P(x: Tensor, order: int) -> Tensor:
    """
    Legendre polynomials P_n on [-1, 1] via torch.special.legendre_polynomial_p.
    """
    if order < 0:
        raise ValueError("Supported orders: n >= 0")
    return torch.special.legendre_polynomial_p(x, order)


def effective_rank_from_states(X: Tensor) -> float:
    """Return centered Shannon effective rank (Roy & Vetterli, 2007).

    X is a finite [observations, neurons] matrix. Center each neuron over
    observations, compute singular values s, retain s > 1e-12, and return
    exp(-sum(p * log(p))) for p = s / sum(s). The cutoff is absolute.
    Singular values are square roots of Gram-matrix eigenvalues; the
    probabilities use s, not s**2. Exactly zero centered states return zero;
    float32 centering residuals can survive the cutoff even for constant input.
    KR/GR observations are final states from separate input streams.
    """
    if X.dim() != 2:
        raise ValueError("X must be a 2D state matrix.")
    if not torch.all(torch.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    # Excluding tiny singular values preserves the historical absolute cutoff.
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    erank = torch.exp(H)
    return float(erank.item())


def ridge_fit_predict(
    Xtr: Tensor,
    ytr: Tensor,
    Xte: Tensor,
    alpha: float,
    DEVICE: torch.device,
    eps: float = 1e-6,
) -> Tensor:
    """Predict test targets with a ridge readout and no intercept.

    Xtr is [T_train,N], Xte is [T_test,N], ytr is [T_train] or [T_train,K].
    Solve (Xtr.T @ Xtr + (alpha + eps) I) W = Xtr.T @ ytr using
    torch.linalg.solve with all K right-hand sides together. No explicit
    inverse or Cholesky factorization is used. Output is [T_test] for a
    one-dimensional target, otherwise [T_test,K].
    """
    Xtr = Xtr.to(DEVICE)
    Xte = Xte.to(DEVICE)
    ytr = ytr.to(DEVICE)

    single_target = False
    if ytr.dim() == 1:
        ytr = ytr.unsqueeze(1)  # [T, 1]
        single_target = True

    T_train, n_feat = Xtr.shape
    Xt = Xtr.transpose(0, 1)  # [N, T]

    # Gram matrix
    G = Xt @ Xtr  # [N, N]
    G = G + (alpha + eps) * torch.eye(n_feat, device=DEVICE, dtype=Xtr.dtype)

    # All right-hand sides at once
    B = Xt @ ytr  # [N, K]
    # Single linear solve for all targets
    W = torch.linalg.solve(G, B)  # [N, K]

    # Predictions
    yhat = Xte @ W  # [T_test, K]

    if single_target:
        return yhat.squeeze(1)

    return yhat


def corr2_score(y_true: Tensor, y_pred: Tensor, eps: float = 1e-12) -> float:
    """
    Squared Pearson correlation used in Jaeger-style memory capacity.
    """
    y_true_c = y_true - y_true.mean()  ## remove the means
    y_pred_c = y_pred - y_pred.mean()  ## remove the means
    num = torch.sum(y_true_c * y_pred_c)  ## this is top on pearson correlation
    den = torch.sqrt(torch.sum(y_true_c**2) * torch.sum(y_pred_c**2)) + eps
    corr2 = (num / den) ** 2
    return float(torch.clamp(corr2, 0.0, 1.0))
