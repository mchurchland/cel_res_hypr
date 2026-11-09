"""Build recurrent [N,N] and input [N,1] weights for architecture controls."""

import numpy as np
import torch
from torch import Tensor

from .normalization import normalize_reservoir_weights
from .topology import (
    er_adjacency,
    ws_directed_adjacency_exact,
    degree_matched_shuffle_directed,
    _match_edge_count,
)
from .weights import (
    _count_edges,
    negative_edge_fraction,
    _shuffle_ce_weights,
    _shuffle_ce_weights_except_1,
    _conn_and_w_shuffle_ce,
    _conn_shuffle_ce,
    _sample_from_cel,
    _sample_from_cel_sign,
    _cel_to_bin,
    _signed_binary_from_cel,
    _binary_to_cel_interpolation,
    _cel_to_shuffled_cel_interpolation,
    apply_percent_negative,
    scale_weights,
)


def build_reservoir(
    target_sr: float
    | None,  # None keeps own-radius weights; baseline mode targets radius 1.
    input_scale: float,
    seed: int,
    feature_conn: str,  # 'cel', 'deg_shuffle', 'ws_p=1.0', 'ws_p=0.1', 'ws_p=0.0', 'er_p=...'
    N: int | None = None,
    ce_W_bio: np.ndarray | None = None,
    drive_idx: np.ndarray | None = None,  # targeted drive for CEL rows if desired
    nnz_target: int | None = None,  # <--- desired number of edges (from CE)
    DEVICE: torch.device | None = None,
    per_neg: float | None = None,
    alpha: float | None = None,
    normalization_mode: str = "spectral_radius",
    normalization_ref: np.ndarray | None = None,
    return_info: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, float | str]]:
    """Construct one seeded recurrent matrix and its input weights.

    ``ce_W_bio`` is [N,N], with rows as sources. ``feature_conn`` selects
    topology and weight/sign controls; CEL controls require the adjacency.
    ``nnz_target`` sets the edge count for synthetic graphs. Returns ``(W, Win)``
    or ``(W, Win, diagnostics)`` when ``return_info=True``. ``Win`` is [N,1].
    Both recurrent and input construction retain the original float32/RNG
    conventions. Sign controls operate on edges and do not enforce Dale's law.
    Normalization diagnostics contain raw, reference, and achieved radii plus
    the multiplicative scale. See docs/ARCHITECTURE.md for control definitions.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_normalization_ref = normalization_ref

    # ---------- base adjacency/weights ----------
    if feature_conn == "cel":
        if ce_W_bio is None:
            raise RuntimeError(
                "you must pass in bio weights (ce_W_bio) to use the celegan reservoirs"
            )
        else:
            W = ce_W_bio.copy().astype(np.float32)
            # Keep the CE edge set as-is; if a different nnz_target was provided, ignore for CEL row.
    elif feature_conn in ("weight_test", "weight_test_unsigned", "weight_test_signed"):
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        assert alpha != None
        W = scale_weights(
            W,
            alpha=alpha,
            rng=rng,
            preserve_sign=(feature_conn == "weight_test_signed"),
        )
    elif feature_conn == "weight_test_binary_to_cel":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        assert alpha != None
        W = _binary_to_cel_interpolation(
            ce_W_bio,
            alpha=alpha,
            rng=rng,
            shuffle_magnitudes=False,
        )
    elif feature_conn == "weight_test_binary_to_shuffled_cel":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        assert alpha != None
        W = _binary_to_cel_interpolation(
            ce_W_bio,
            alpha=alpha,
            rng=rng,
            shuffle_magnitudes=True,
        )
    elif feature_conn == "weight_test_cel_to_shuffled_cel":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        assert alpha != None
        W = _cel_to_shuffled_cel_interpolation(
            ce_W_bio,
            alpha=alpha,
            rng=rng,
        )

    elif feature_conn == "sign_test_cel":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        if per_neg == None:
            raise ValueError("Per neg required for sign_test")
        W = ce_W_bio.copy().astype(np.float32)
        W = _cel_to_bin(W)
        W = apply_percent_negative(W=W, per_neg=per_neg, rng=rng)
    elif feature_conn == "sign_test_og_cel":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        if per_neg == None:
            raise ValueError("Per neg required for sign_test")
        W = ce_W_bio.copy().astype(np.float32)
        W = np.abs(W)
        W = apply_percent_negative(W=W, per_neg=per_neg, rng=rng)
    elif feature_conn == "sign_test_er":
        if per_neg == None:
            raise ValueError("Per neg required for sign_test")
        assert N != None
        # Historical sign-sweep protocol fixes p=0.1; see docs/AUDIT.md.
        p = 0.1
        A = er_adjacency(N, p, rng).astype(np.float32)
        mask = (A != 0).astype(np.float32)
        if nnz_target is not None:
            mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(
                np.float32
            )
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
        W = _cel_to_bin(W)
        # Use this ER realization signed at the reference CE negative-edge
        # fraction. Clone the RNG state so the reference and sweep point use
        # the same sign permutation; at the CE fraction they are identical.
        reference_source = (
            normalization_ref if normalization_ref is not None else ce_W_bio
        )
        if reference_source is None or np.count_nonzero(reference_source) == 0:
            raise ValueError(
                "sign_test_er requires a nonempty CE matrix for its reference sign fraction."
            )
        reference_negative_fraction = float(
            np.count_nonzero(reference_source < 0)
        ) / float(np.count_nonzero(reference_source))
        reference_rng = np.random.default_rng()
        reference_rng.bit_generator.state = rng.bit_generator.state
        model_normalization_ref = apply_percent_negative(
            W.copy(),
            per_neg=reference_negative_fraction,
            rng=reference_rng,
        )
        W = apply_percent_negative(W=W, per_neg=per_neg, rng=rng)

    elif feature_conn == "local_sign":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        sel_p = W > 0
        sel_n = W < 0

        # match cel+randN: N(0, 1) on existing edges
        num_pos = int(sel_p.sum())
        num_neg = int(sel_n.sum())
        if num_pos:
            W[sel_p] = np.abs(
                rng.normal(loc=0.0, scale=1.0, size=num_pos).astype(np.float32)
            )
        if num_neg:
            W[sel_n] = -np.abs(
                rng.normal(loc=0.0, scale=1.0, size=num_neg).astype(np.float32)
            )

    elif feature_conn == "local_sign+flat":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        sel_p = W > 0
        sel_n = W < 0

        # W = _shuffle_ce_weights_except_1(Wbio = W,rng=rng)
        mags = rng.uniform(0.0, 1.0, size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * mags
        W[sel_p] = np.abs(W[sel_p])
        W[sel_n] = -np.abs(W[sel_n])

    elif feature_conn == "local_sign+sample":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        W = _sample_from_cel_sign(Wbio=W, rng=rng)

    elif feature_conn == "local_sign+binary":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = _signed_binary_from_cel(ce_W_bio)

    elif feature_conn == "binary+shuffle":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = _signed_binary_from_cel(ce_W_bio)
        W = degree_matched_shuffle_directed(W, tries=20_000, rng=rng).astype(np.float32)

    elif feature_conn == "binary+conshuffle+wshuffle":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = _signed_binary_from_cel(ce_W_bio)
        W = _conn_and_w_shuffle_ce(Wbio=W, rng=rng)

    elif feature_conn == "binary_base":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)
        W = _cel_to_bin(Wbio=W)
    elif feature_conn == "binary_base_topology_shuffle":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)
        W = _cel_to_bin(Wbio=W)
        W = degree_matched_shuffle_directed(W, tries=20_000, rng=rng).astype(np.float32)

    elif feature_conn == "global_sign_pres":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)
        nz = np.nonzero(W)
        count_n = np.count_nonzero(W < 0)

        sel_n = rng.choice(len(nz[0]), count_n, replace=False)
        W = _cel_to_bin(Wbio=W)
        W[nz[0][sel_n], nz[1][sel_n]] = -W[nz[0][sel_n], nz[1][sel_n]]

    elif feature_conn == "global_sign_pres_real_weight":
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)
        nz = np.nonzero(W)
        count_n = np.count_nonzero(W < 0)

        sel_n = rng.choice(len(nz[0]), count_n, replace=False)
        W = np.abs(W)
        W[nz[0][sel_n], nz[1][sel_n]] = -W[nz[0][sel_n], nz[1][sel_n]]

    elif feature_conn.startswith("ws_p="):
        assert N != None
        assert nnz_target != None
        p = float(feature_conn.split("=")[1])
        A = ws_directed_adjacency_exact(N, nnz_target, p, rng).astype(np.float32)
        mask = (A != 0).astype(np.float32)
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)

    elif feature_conn.startswith("er_p="):
        assert N != None
        p = float(feature_conn.split("=")[1])
        A = er_adjacency(N, p, rng).astype(np.float32)
        mask = (A != 0).astype(np.float32)
        if nnz_target is not None:
            mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(
                np.float32
            )
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
    elif feature_conn == "cel_randN":
        W = ce_W_bio.copy().astype(np.float32)
        mask = (W != 0).astype(np.float32)
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
    else:
        raise RuntimeError("u gotta pass in feature_conn that actually exists")

    Wt = torch.from_numpy(W).to(DEVICE)
    ref = model_normalization_ref if model_normalization_ref is not None else ce_W_bio
    Wt, norm_info = normalize_reservoir_weights(
        Wt,
        target_sr=target_sr,
        mode=normalization_mode,
        reference_W=ref,
        device=DEVICE,
    )

    if drive_idx is not None and len(drive_idx) > 0:
        Win = torch.zeros(Wt.shape[0], 1, device=DEVICE)
        # Index tensors must be integer type.
        Win[torch.as_tensor(drive_idx, device=DEVICE, dtype=torch.long), 0] = 1.0
        Win = Win * (input_scale / (Win.norm() + 1e-12))
    else:
        Win = torch.randn(Wt.shape[0], 1, device=DEVICE, dtype=Wt.dtype) * input_scale

    if return_info:
        return Wt, Win, norm_info
    return Wt, Win
