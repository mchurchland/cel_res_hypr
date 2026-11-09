"""Spectral-radius normalization and its recorded diagnostics."""

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def spectral_radius_power(W: Tensor) -> float:
    """Return max(abs(eigvals(W))); the legacy name does not imply iteration."""
    eigs = torch.linalg.eigvals(W)
    return float(torch.max(torch.abs(eigs)).item())


@torch.no_grad()
def scale_to_sr(W: torch.Tensor, target_sr: float | None):
    """
    Scale W so that spectral radius rho(W) = target_sr (Jaeger, 2002, GMD Report 152); if None, return unchanged.
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py#L59 (spectral-radius scaling in ESNs).
    """
    if target_sr is None:
        return W
    sr = spectral_radius_power(W)
    if sr < 1e-9:
        return W
    return (target_sr / sr) * W


@torch.no_grad()
def normalize_reservoir_weights(
    W: torch.Tensor,
    target_sr: float | None,
    mode: str,
    reference_W: np.ndarray | torch.Tensor | None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """
    Normalize raw reservoir weights for sign-balance ablations.

    Modes:
      - spectral_radius: W * target_sr / rho(W), the existing RC comparison.
      - original_radius: W * target_sr / rho(W_orig), so sign perturbations do not
        get amplified just because they changed their own raw spectral radius.
    """
    mode = str(mode)
    if mode not in {"spectral_radius", "original_radius"}:
        raise ValueError(
            "normalization_mode must be one of: spectral_radius, original_radius."
        )

    raw_rho = spectral_radius_power(W)
    ref_rho = float("nan")
    scale = 1.0
    ref_t = None

    if reference_W is not None:
        if isinstance(reference_W, torch.Tensor):
            ref_t = reference_W.to(device=W.device, dtype=W.dtype)
        else:
            ref_t = torch.from_numpy(reference_W.astype(np.float32, copy=False)).to(
                device if device is not None else W.device,
                dtype=W.dtype,
            )
        ref_rho = spectral_radius_power(ref_t)

    if mode == "spectral_radius":
        if target_sr is not None and raw_rho >= 1e-9:
            scale = float(target_sr) / raw_rho
    else:
        if ref_t is None:
            raise ValueError(
                f"normalization_mode={mode!r} requires reference_W/ce_W_bio."
            )
        target = 1.0 if target_sr is None else float(target_sr)
        if ref_rho >= 1e-9:
            scale = target / ref_rho

    W_scaled = W * scale
    info = {
        "normalization": mode,
        "raw_rho": raw_rho,
        "ref_rho": ref_rho,
        "post_rho": spectral_radius_power(W_scaled),
        "scale_factor": float(scale),
    }
    return W_scaled, info


def spectral_norm(W: Tensor) -> float:
    """Spectral norm via leading singular value (Golub & Van Loan, 2013, Matrix Computations 4th ed.). See: https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py"""
    return float(torch.linalg.svdvals(W)[0])
