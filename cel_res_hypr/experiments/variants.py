"""Architecture names, historical labels, and seeded control dispatch."""

from __future__ import annotations

import numpy as np

from cel_res_hypr.networks.weights import (
    _count_edges,
    _shuffle_ce_weights,
    _conn_and_w_shuffle_ce,
    _conn_shuffle_ce,
    _sample_from_cel,
    _cel_to_bin,
    _signed_binary_from_cel,
)
from .config import VariantContext
from .evaluation import _run_variant_row

VARIANT_LABELS = {
    "real": "real",
    "shuffle_weights": "shuffle",
    "cel_randN": "cel+randN",
    "er_randN": "er+randN",
    "ws_p01_randN": "ws_p0.1+randN",
    "conn_shuf": "celW+connShuf",
    "local_sign": "local_sign",
    "conn_shuf_only": "conn_shuf_only",
    "cel_sample": "cel_sample",
    "local_sign+flat": "local_sign+flat",
    "local_sign+sample": "local_sign+sample",
    "local_sign+binary": "local_sign+binary",
    "global_sign_pres": "global_sign_pres",
    "global_sign_pres_real_weight": "global_sign_pres_real_weight",
    "binary_base": "binary_base",
    "binary_base_topology_shuffle": "binary_base_topology_shuffle",
    "binary+conshuffle+wshuffle": "binary+conshuffle+wshuffle",
    "sign_test_cel": "sign_test_cel",
    "sign_test": "sign_test",
    "sign_test_er": "sign_test_er",
    "weight_test": "weight_test",
    "weight_test_unsigned": "weight_test_unsigned",
    "weight_test_signed": "weight_test_signed",
    "weight_test_cel_to_shuffled_cel": "weight_test_cel_to_shuffled_cel",
    "weight_test_binary_to_cel": "weight_test_binary_to_cel",
    "weight_test_binary_to_shuffled_cel": "weight_test_binary_to_shuffled_cel",
    "binary+shuffle": "binary+shuffle",
    "sign_test_og_cel": "sign_test_og_cel",
}


VARIANT_DESCRIPTIONS = {
    "real": "C. elegans adjacency with empirical signed weights.",
    "shuffle_weights": "Permute the signed empirical weight multiset over the existing C. elegans edge set.",
    "cel_randN": "C. elegans edge set with independent Gaussian N(0,1) weights.",
    "er_randN": "Directed ER topology with Gaussian weights and the C. elegans edge count.",
    "ws_p01_randN": "Directed WS-like topology with Gaussian weights and the C. elegans edge count.",
    "conn_shuf": "Degree-preserving directed topology shuffle followed by signed-weight permutation.",
    "local_sign": "C. elegans edge signs with independent absolute-Gaussian magnitudes.",
    "conn_shuf_only": "Degree-preserving directed topology shuffle transporting each edge weight.",
    "cel_sample": "C. elegans edge set with signed weights resampled from the empirical nonzero weights.",
    "local_sign+flat": "C. elegans edge signs with independent uniform [0,1) magnitudes.",
    "local_sign+sample": "C. elegans edge signs with weights resampled separately from positive and negative empirical weights.",
    "local_sign+binary": "C. elegans edge set and signs with unit nonzero magnitudes.",
    "sign_test_cel": "C. elegans edge set with unit magnitudes and a specified negative-edge fraction.",
    "sign_test_og_cel": "C. elegans edge set and empirical magnitudes with a specified negative-edge fraction.",
    "sign_test_er": "Edge-count-matched ER graph (historical p=0.1) with unit magnitudes and a specified negative-edge fraction.",
    "weight_test": "Backward-compatible alias for weight_test_unsigned.",
    "weight_test_unsigned": "C. elegans weights plus signed Gaussian noise; edge signs may change.",
    "weight_test_signed": "C. elegans weights plus Gaussian noise with original edge signs restored.",
    "weight_test_cel_to_shuffled_cel": "Interpolate empirical and shuffled empirical magnitude placement while preserving edge signs.",
    "weight_test_binary_to_cel": "Interpolate signed-unit and empirical C. elegans magnitudes while preserving edge signs.",
    "weight_test_binary_to_shuffled_cel": "Interpolate signed-unit and shuffled empirical magnitudes while preserving edge signs.",
    "global_sign_pres": "C. elegans edge set with unit magnitudes and randomized signs preserving the global negative-edge count.",
    "global_sign_pres_real_weight": "C. elegans edge set and empirical magnitudes with randomized signs preserving the global negative-edge count.",
    "binary_base": "C. elegans edge set with all nonzero weights equal to +1.",
    "binary_base_topology_shuffle": "Degree-preserving topology shuffle of the all-positive unit-weight C. elegans graph.",
    "binary+conshuffle+wshuffle": "Degree-preserving topology shuffle of signed-unit weights, followed by sign permutation over existing edges.",
    "binary+shuffle": "Degree-preserving topology shuffle transporting signed-unit edge weights.",
    "sign_test": "CLI job expanding to C. elegans and ER negative-edge-fraction sweeps.",
}


VARIANT_ALIASES = {}


VARIANT_KEYS: tuple[str, ...] = tuple(VARIANT_LABELS.keys())


WEIGHT_TEST_FEATURES = {
    "weight_test_binary_to_shuffled_cel": "weight_test_binary_to_shuffled_cel",
    "weight_test_cel_to_shuffled_cel": "weight_test_cel_to_shuffled_cel",
    "weight_test_binary_to_cel": "weight_test_binary_to_cel",
    "weight_test_unsigned": "weight_test_unsigned",
    "weight_test_signed": "weight_test_signed",
    "weight_test": "weight_test_unsigned",
}


def _seed(ctx: VariantContext, *, offset: int, sid_stride: int = 0) -> int:
    """Shared seed rule used across variants."""
    return ctx.seed + offset + ctx.sid * sid_stride


def _nnz_match_ce(ctx: VariantContext) -> int:
    if ctx.ce_W_bio is None:
        raise ValueError("nnz_match_ce requires ce_W_bio.")
    return _count_edges(ctx.ce_W_bio)


def _resolve_key(key: str) -> str:
    return VARIANT_ALIASES.get(key, key)


def _weight_test_feature_conn(key: str) -> str:
    for prefix, feature_conn in WEIGHT_TEST_FEATURES.items():
        if key.startswith(prefix):
            return feature_conn
    raise ValueError(f"Unknown weight_test variant key: {key}")


def _require_ce(ctx: VariantContext):
    if ctx.ce_W_bio is None:
        raise ValueError("This variant requires ce_W_bio (CE adjacency).")


# Direct controls use the same construction seed and no adjacency override.
_DIRECT_FEATURES = {
    "real": "cel",
    "cel_randN": "cel_randN",
    "local_sign": "local_sign",
    "local_sign+flat": "local_sign+flat",
    "local_sign+sample": "local_sign+sample",
    "local_sign+binary": "local_sign+binary",
    "global_sign_pres": "global_sign_pres",
    "global_sign_pres_real_weight": "global_sign_pres_real_weight",
    "binary_base": "binary_base",
}


def run_variant(key: str, ctx: VariantContext) -> list[tuple]:
    """Evaluate one named control while preserving labels and seed assignments.

    Returns rows in CSV_HEADER order, one per context hyperparameter tuple.
    Sign-fraction and weight-test keys may carry a numeric label suffix;
    their numeric parameters are supplied by ctx.per_neg and ctx.alpha.
    """
    key = _resolve_key(key)
    if key not in VARIANT_KEYS and not key.startswith(
        ("sign_test_cel", "weight_test", "sign_test_er", "sign_test_og_cel")
    ):
        raise ValueError(f"Unknown variant key: {key}")
    _require_ce(ctx)
    seed_base = _seed(ctx, offset=1)
    ce_override = None
    nnz_target = None
    mode_label = VARIANT_LABELS.get(key, key)

    if key in _DIRECT_FEATURES:
        feature_conn = _DIRECT_FEATURES[key]
    elif key in {"er_randN", "ws_p01_randN"}:
        feature_conn = f"er_p={ctx.er_p}" if key == "er_randN" else f"ws_p={ctx.ws_p}"
        nnz_target = _nnz_match_ce(ctx)
    elif key in {"shuffle_weights", "conn_shuf", "conn_shuf_only", "cel_sample"}:
        transform = {
            "shuffle_weights": _shuffle_ce_weights,
            "conn_shuf": _conn_and_w_shuffle_ce,
            "conn_shuf_only": _conn_shuffle_ce,
            "cel_sample": _sample_from_cel,
        }[key]
        ce_override = transform(ctx.ce_W_bio, np.random.default_rng(ctx.seed))
        feature_conn = "cel"
    elif key in {
        "binary_base_topology_shuffle",
        "binary+conshuffle+wshuffle",
        "binary+shuffle",
    }:
        binary = (
            _cel_to_bin(ctx.ce_W_bio)
            if key == "binary_base_topology_shuffle"
            else _signed_binary_from_cel(ctx.ce_W_bio)
        )
        transform = (
            _conn_and_w_shuffle_ce
            if key == "binary+conshuffle+wshuffle"
            else _conn_shuffle_ce
        )
        ce_override = transform(binary, np.random.default_rng(seed_base))
        feature_conn = "cel"
    elif key.startswith(("sign_test_cel", "sign_test_er", "sign_test_og_cel")):
        feature_conn = next(
            prefix
            for prefix in ("sign_test_cel", "sign_test_er", "sign_test_og_cel")
            if key.startswith(prefix)
        )
        mode_label = key
        if feature_conn == "sign_test_er":
            nnz_target = _nnz_match_ce(ctx)
    elif key.startswith("weight_test"):
        feature_conn = _weight_test_feature_conn(key)
        mode_label = key
    else:
        raise ValueError(f"Variant key not implemented: {key}")

    return _run_variant_row(
        ctx,
        feature_conn=feature_conn,
        mode_label=mode_label,
        ce_override=ce_override,
        nnz_target=nnz_target,
        seed_base=seed_base,
    )


def list_variants() -> list[tuple[str, str]]:
    """Return (key, description) pairs sorted by key."""
    return sorted(
        ((k, VARIANT_DESCRIPTIONS[k]) for k in VARIANT_KEYS), key=lambda kv: kv[0]
    )
