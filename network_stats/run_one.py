"""Evaluate MC/IPC and independent-input/shared-tail rank ensembles."""

import torch
from torch import Tensor

from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
from network_stats.dynamics import (
    run_reservoir_with_pre,
    run_reservoir_states,
    run_reservoir_stream_batch,
)
from network_stats.streams import (
    make_gr_input_streams,
    make_gr_input_stream_sweep,
    make_kr_input_streams,
)


def run_one(
    W: Tensor,
    Win: Tensor,
    leak: float,
    device: torch.device,
    WASHOUT: int,
    T_TRAIN: int,
    T_TEST: int,
    MC_MAX_DELAY: int,
    IPC_MAX_DELAY: int,
    IPC_ORDERS: list[int],
    RIDGE_ALPHA: float,
    output_idx: Tensor | None = None,
    bias: Tensor | None = None,
    kr_num_streams: int = 300,
    kr_stream_length: int = 10,
    kr_seed: int = 0,
    gr_num_streams: int = 300,
    gr_stream_length: int = 10,
    gr_common_tail_length: int = 3,
    gr_seed: int = 0,
    rank_only: bool = False,
    kr_only: bool = False,
) -> dict:
    """Evaluate a reservoir with MC/IPC trajectories and KR/GR ensembles.

    W is [neurons,neurons] with rows as sources, Win [neurons,1]. MC/IPC
    use one global-RNG uniform input trajectory after washout. KR and GR use
    dedicated generators and zero-reset final-state ensembles; GR shares
    its recent input tail. Optional output_idx selects observed neurons
    only. Returns scalar MC, IPC, KR, GR entries; skipped entries are NaN.
    """
    if rank_only and kr_only:
        raise ValueError("rank_only and kr_only are mutually exclusive.")

    metrics_only = rank_only or kr_only
    if not metrics_only:
        T_total = WASHOUT + T_TRAIN + T_TEST
        u = torch.rand(T_total, 1, device=device) * 2.0 - 1.0  ## rescale to [-1, 1]
        X, _ = run_reservoir_with_pre(W, Win, u, leak, bias=bias)

    # Both rank metrics use one final state from each sequence, with every
    # sequence initialized from the same zero state. KR sequences are fully
    # independent; GR sequences share the configured recent input tail.
    rank_initial_state = torch.zeros(W.shape[0], device=device, dtype=W.dtype)
    kr_generator = torch.Generator(device=device)
    kr_generator.manual_seed(kr_seed)
    kr_streams = make_kr_input_streams(
        n_streams=kr_num_streams,
        stream_length=kr_stream_length,
        n_inputs=Win.shape[1],
        device=device,
        dtype=W.dtype,
        generator=kr_generator,
    )
    X_kr = run_reservoir_stream_batch(
        W,
        Win,
        kr_streams,
        leak,
        initial_state=rank_initial_state,
        bias=bias,
    )

    X_gr = None
    if not kr_only:
        gr_generator = torch.Generator(device=device)
        gr_generator.manual_seed(gr_seed)
        gr_streams = make_gr_input_streams(
            n_streams=gr_num_streams,
            stream_length=gr_stream_length,
            common_tail_length=gr_common_tail_length,
            n_inputs=Win.shape[1],
            device=device,
            dtype=W.dtype,
            generator=gr_generator,
        )
        X_gr = run_reservoir_stream_batch(
            W,
            Win,
            gr_streams,
            leak,
            initial_state=rank_initial_state,
            bias=bias,
        )
    if output_idx is not None:
        idx = torch.as_tensor(output_idx, device=device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("output_idx must contain at least one node when provided.")
        if not metrics_only:
            X = X.index_select(1, idx)
        X_kr = X_kr.index_select(1, idx)
        if X_gr is not None:
            X_gr = X_gr.index_select(1, idx)

    if metrics_only:
        MC_total = float("nan")
        IPC_total = float("nan")
    else:
        Xtr = X[WASHOUT : WASHOUT + T_TRAIN]  ## t_train
        Xte = X[WASHOUT + T_TRAIN :]  ## t_test
        utr = u[WASHOUT : WASHOUT + T_TRAIN]  ## u_train
        ute = u[WASHOUT + T_TRAIN :]  ## u_test
        MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA, device)
        IPC_total = compute_IPC(
            Xtr,
            Xte,
            utr,
            ute,
            IPC_MAX_DELAY,
            RIDGE_ALPHA,
            device,
            IPC_ORDERS,
        )
    KR_val = compute_KR(X_kr)
    GR_val = float("nan") if X_gr is None else compute_GR(X_gr)
    return dict(
        MC=MC_total,
        IPC=IPC_total,
        KR=KR_val,
        GR=GR_val,
    )
