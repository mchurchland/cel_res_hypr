"""Build and evaluate one architecture over its hyperparameter grid."""

from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
from scipy.stats import entropy, halfnorm
from sklearn.metrics.pairwise import cosine_similarity

from network_stats.run_one import run_one
from cel_res_hypr.networks.builder import build_reservoir
from cel_res_hypr.networks.weights import _cel_to_bin, _signed_binary_from_cel
from cel_res_hypr.networks.normalization import scale_to_sr
from .config import SimulationParams, DEFAULT_SIM_PARAMS, VariantContext


def _kl_empirical_to_fitted_gaussian(
    values: np.ndarray,
    bins: int = 64,
    eps: float = 1e-12,
) -> float:
    """
    Magnitude-only KL on edge weights.
    Computes KL(|W| || fitted shifted-half-normal) on nonzero finite weights.
    """
    x = values.reshape(-1)
    x = x[np.isfinite(x)]
    x = x[x != 0]
    if x.size < 2:
        return float("nan")

    x_abs = np.abs(x)
    x_max = float(np.max(x_abs))
    if x_max <= 0.0:
        return 0.0

    loc = float(np.min(x_abs))
    y = x_abs - loc
    sigma = float(np.sqrt(np.mean(y * y)))
    if sigma <= eps:
        return 0.0

    counts, edges = np.histogram(x_abs, bins=bins, range=(loc, x_max))
    p = counts.astype(np.float64)
    p = np.clip(p, eps, None)
    p = p / p.sum()

    # Shifted half-normal CDF over the same bin edges.
    q_cdf = halfnorm.cdf(edges, loc=loc, scale=sigma)
    q = np.diff(q_cdf)
    q = np.clip(q, eps, None)
    q = q / q.sum()
    return float(entropy(p, q))


def evaluate_reservoir(
    Wt: torch.Tensor,
    Win: torch.Tensor,
    leak: float,
    device: torch.device,
    sim_params: SimulationParams = DEFAULT_SIM_PARAMS,
    output_idx: np.ndarray | torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    kr_seed: int = 0,
    gr_seed: int = 0,
    rank_only: bool = False,
    kr_only: bool = False,
) -> dict[str, float]:
    """Wrapper around run_one with shared defaults."""
    return run_one(
        Wt,
        Win,
        leak,
        device,
        sim_params.washout,
        sim_params.t_train,
        sim_params.t_test,
        sim_params.mc_max_delay,
        sim_params.ipc_max_delay,
        sim_params.ipc_orders,
        sim_params.ridge_alpha,
        output_idx,
        bias=bias,
        kr_num_streams=sim_params.kr_num_streams,
        kr_stream_length=sim_params.kr_stream_length,
        kr_seed=kr_seed,
        gr_num_streams=sim_params.gr_num_streams,
        gr_stream_length=sim_params.gr_stream_length,
        gr_common_tail_length=sim_params.gr_common_tail_length,
        gr_seed=gr_seed,
        rank_only=rank_only,
        kr_only=kr_only,
    )


def _apply_input_subset(
    Win: torch.Tensor, input_idx: np.ndarray | None
) -> torch.Tensor:
    """Keep the usual random input weights, but drive only selected nodes."""
    if input_idx is None:
        return Win
    idx = torch.as_tensor(input_idx, device=Win.device, dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("input_idx must contain at least one node when provided.")
    mask = torch.zeros_like(Win)
    mask.index_fill_(0, idx, 1.0)
    return Win * mask


def _unpack_col_param(param: Sequence[float]) -> tuple[float, float, float, float]:
    if len(param) == 3:
        target_sr, leak, in_scale = param
        return float(target_sr), float(leak), float(in_scale), 0.0
    if len(param) == 4:
        target_sr, leak, in_scale, neuron_bias = param
        return float(target_sr), float(leak), float(in_scale), float(neuron_bias)
    raise ValueError(f"Expected 3 or 4 sweep parameters, got {len(param)}: {param}")


def _make_neuron_bias(
    n_nodes: int,
    bias_range: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor | None:
    if bias_range == 0.0:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.empty(n_nodes, device=device, dtype=dtype).uniform_(
        -float(bias_range),
        float(bias_range),
        generator=gen,
    )


def _family_baseline_ref(ctx: VariantContext, mode_label: str) -> np.ndarray:
    """Reference matrix for baseline-radius normalization within a shuffle family."""
    if mode_label in {"binary_base", "binary_base_topology_shuffle"}:
        return _cel_to_bin(ctx.ce_W_bio)
    if mode_label in {
        "local_sign+binary",
        "global_sign_pres",
        "binary+shuffle",
        "binary+conshuffle+wshuffle",
    }:
        return _signed_binary_from_cel(ctx.ce_W_bio)
    return ctx.ce_W_bio


def _run_variant_row(
    ctx: VariantContext,
    *,
    feature_conn: str,
    mode_label: str,
    ce_override: np.ndarray | None,
    nnz_target: int | None,
    seed_base: int,
) -> list[tuple]:
    rows_local = []

    # choose CE matrix when needed
    ce_for_conn = ce_override if ce_override is not None else ctx.ce_W_bio
    if feature_conn == "cel" and ce_for_conn is None:
        raise ValueError("feature_conn='cel' requires a CE adjacency matrix.")
    if ctx.ce_W_bio is None and feature_conn != "cel":
        raise ValueError("Non-CEL variants need ce_W_bio to set N/nnz.")

    Nloc = ce_for_conn.shape[0] if ce_for_conn is not None else ctx.ce_W_bio.shape[0]
    for ci, param in enumerate(ctx.col_params):
        target_sr, leak, in_scale, neuron_bias = _unpack_col_param(param)
        assert ctx.ce_ei == None
        Wt, Win, norm_info = build_reservoir(
            feature_conn=feature_conn,
            target_sr=target_sr,
            N=Nloc,
            ce_W_bio=ce_for_conn if feature_conn == "cel" else ctx.ce_W_bio,
            input_scale=in_scale,
            seed=seed_base,
            drive_idx=None,
            nnz_target=nnz_target,
            DEVICE=ctx.device,
            per_neg=ctx.per_neg,
            alpha=ctx.alpha,
            normalization_mode=ctx.normalization_mode,
            normalization_ref=_family_baseline_ref(ctx, mode_label),
            return_info=True,
        )
        Win = _apply_input_subset(Win, ctx.input_idx)
        bias_vec = _make_neuron_bias(
            Wt.shape[0],
            neuron_bias,
            ctx.device,
            Wt.dtype,
            seed_base + 17_000_003,
        )
        w_np = Wt.detach().cpu().numpy().reshape(-1)
        kl_to_gaussian = _kl_empirical_to_fitted_gaussian(w_np)
        scores = evaluate_reservoir(
            Wt,
            Win,
            leak,
            ctx.device,
            ctx.sim_params,
            ctx.output_idx,
            bias=bias_vec,
            kr_seed=ctx.seed + ctx.sim_params.kr_seed_offset,
            gr_seed=ctx.seed + ctx.sim_params.gr_seed_offset,
            rank_only=ctx.rank_only,
            kr_only=ctx.kr_only,
        )
        Wt_ce = torch.from_numpy(_cel_to_bin(ctx.ce_W_bio)).to(ctx.device)
        sigma_ce = scale_to_sr(Wt_ce, target_sr)
        row_mode_label = (
            f"{mode_label}__norm_{ctx.normalization_mode}"
            if ctx.label_normalization
            else mode_label
        )
        rows_local.append(
            (
                row_mode_label,
                ctx.normalization_mode,
                ctx.sid,
                target_sr,
                leak,
                in_scale,
                neuron_bias,
                float(scores["MC"]),
                float(scores["IPC"]),
                float(scores["KR"]),
                float(scores["GR"]),
                float(Wt.mean().item()),
                float(
                    cosine_similarity(
                        sigma_ce.reshape(1, -1).detach().cpu().numpy(),
                        Wt.reshape(1, -1).detach().cpu().numpy(),
                    )[0, 0]
                ),
                kl_to_gaussian,
                seed_base,
                ctx.src_tag,
                float(norm_info["raw_rho"]),
                float(norm_info["ref_rho"]),
                float(norm_info["post_rho"]),
                float(norm_info["scale_factor"]),
            )
        )
    return rows_local
