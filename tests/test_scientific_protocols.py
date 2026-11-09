"""Controlled scientific examples and explicit historical behavior guards."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from network_stats.dynamics import (
    run_reservoir_states,
    run_reservoir_with_pre,
    run_reservoir_stream_batch,
)
from network_stats.stats import (
    compute_MC,
    compute_IPC,
    compute_KR,
    compute_GR,
    compute_KR_pilati,
)
from network_stats.stats_util import corr2_score, legendre_P, ridge_fit_predict
from cel_res_hypr.experiments.sampling import (
    _split_indices,
    _draw_node_subsets,
    _build_col_params,
)
from cel_res_hypr.networks.weights import _shuffle_ce_weights_except_1


def test_source_row_orientation_matches_manual_recurrence():
    W = torch.tensor([[0.0, 0.7], [-0.2, 0.1]])
    Win = torch.tensor([[0.4], [-0.5]])
    inputs = torch.tensor([[1.0], [-0.2], [0.3]])
    bias = torch.tensor([0.1, -0.1])
    expected, pre_values, z = [], [], np.zeros(2)
    for u in inputs.numpy():
        pre = W.numpy().T @ z + Win.numpy() @ u + bias.numpy()
        z = 0.4 * z + 0.6 * np.tanh(pre)
        expected.append(z.copy())
        pre_values.append(pre)
    states, pre = run_reservoir_with_pre(W, Win, inputs, 0.6, bias)
    np.testing.assert_allclose(states.numpy(), expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(pre.numpy(), pre_values, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(run_reservoir_states(W, Win, inputs, 0.6, bias), states)
    final = run_reservoir_stream_batch(W, Win, inputs[None], 0.6, torch.zeros(2), bias)
    torch.testing.assert_close(final[0], states[-1])


def test_legacy_preactivation_float64_limitation_is_explicit():
    with pytest.raises(RuntimeError):
        run_reservoir_with_pre(
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
            torch.zeros(3, 1, dtype=torch.float64),
            0.6,
        )
    assert (
        run_reservoir_states(
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, 1, dtype=torch.float64),
            torch.zeros(3, 1, dtype=torch.float64),
            0.6,
        ).dtype
        == torch.float64
    )


def delayed_polynomial_states(seed):
    u = torch.rand(100, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    states = torch.zeros(100, 3)
    for col, order in enumerate((1, 2, 3)):
        states[1:, col] = legendre_P(u[:-1], order)
    return u, states


def test_mc_and_restricted_ipc_recover_known_single_delay_targets():
    utr, Xtr = delayed_polynomial_states(11)
    ute, Xte = delayed_polynomial_states(12)
    mc, capacities = compute_MC(
        Xtr[:, :1], Xte[:, :1], utr, ute, 1, 0.0, torch.device("cpu")
    )
    ipc = compute_IPC(Xtr, Xte, utr, ute, 1, 0.0, torch.device("cpu"), [1, 2, 3])
    assert mc == pytest.approx(1.0, abs=2e-6)
    assert capacities.dtype == np.float32 and capacities.shape == (1,)
    assert ipc == pytest.approx(3.0, abs=6e-6)
    assert (
        compute_IPC(Xtr, Xte, utr, ute, 1, 0.0, torch.device("cpu"), [1, 2, 3], 1.1)
        == 0.0
    )
    assert corr2_score(torch.ones(5), torch.arange(5.0)) == 0.0


def test_ridge_regularization_includes_historical_epsilon():
    X = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    y = torch.tensor([2.0, 4.0])
    expected = torch.tensor([2.0 / (1.0 + 1e-6), 16.0 / (4.0 + 1e-6)])
    torch.testing.assert_close(
        ridge_fit_predict(X, y, X, 0.0, torch.device("cpu")), expected
    )


def test_rank_uses_centered_singular_values_not_variance_probabilities():
    X = torch.tensor(
        [[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float64
    )
    p = np.array([2 / 3, 1 / 3])
    expected = np.exp(-np.sum(p * np.log(p)))
    assert compute_KR(X) == pytest.approx(expected, rel=1e-12)
    assert compute_GR(X + torch.tensor([8.0, -3.0])) == pytest.approx(
        expected, rel=1e-12
    )
    assert compute_KR(torch.ones(4, 2)) == 0.0
    assert (
        compute_KR_pilati(torch.ones(4, 2)) == 1
    )  # Alternative historical definition is uncentered.
    assert compute_KR(torch.zeros(4, 2)) == 0.0


def test_historical_absolute_rank_cutoff_retains_centering_roundoff():
    # A decimal constant need not center to exact zero in float32.
    constant = torch.full((7, 4), 0.1)
    centered = constant - constant.mean(dim=0, keepdim=True)
    if torch.count_nonzero(centered):
        assert compute_GR(constant) == pytest.approx(1.0, abs=2e-6)
    else:
        assert compute_GR(constant) == 0.0


def test_grid_order_partitions_and_disjoint_sampling_are_stable():
    assert _build_col_params([0.8, 1.0], [0.6], [0.1, 0.5], [0.0]) == [
        (0.8, 0.6, 0.1, 0.0),
        (0.8, 0.6, 0.5, 0.0),
        (1.0, 0.6, 0.1, 0.0),
        (1.0, 0.6, 0.5, 0.0),
    ]
    parts = [_split_indices(11, 3, rank) for rank in range(3)]
    assert parts == [list(range(4)), list(range(4, 8)), list(range(8, 11))]
    assert _split_indices(11, 3, 3) == parts[-1]
    a, b = _draw_node_subsets(12, 3, 4, 5)
    aa, bb = _draw_node_subsets(12, 3, 4, 5)
    assert len(set(a) & set(b)) == 0
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)
    with pytest.raises(ValueError):
        _draw_node_subsets(4, 3, 2, 5)


def test_historical_except_one_shuffle_indexes_rows():
    W = np.array([[0, 2, 1], [3, 0, 4], [1, 1, 0]], dtype=np.float32)
    result = _shuffle_ce_weights_except_1(W, np.random.default_rng(0))
    np.testing.assert_array_equal(result, [[3, 0, 4], [3, 0, 4], [1, 1, 0]])


def test_cli_historically_ignores_unknown_sign_fraction(tmp_path, monkeypatch):
    import inv_arc_test as cli

    adjacency = tmp_path / "adj.npy"
    unknown = tmp_path / "unknown.npy"
    np.save(adjacency, np.array([[0, 1], [1, 0]], dtype=np.float32))
    np.save(unknown, np.array([[0, 2], [2, 0]], dtype=np.float32))
    captured = []
    monkeypatch.setattr(
        cli,
        "_run_and_save",
        lambda key, ctx, *args, **kwargs: captured.append(ctx.ce_W_bio.copy()),
    )
    for fraction in ("0.0", "1.0"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "inv_arc_test.py",
                "--job",
                "real",
                "--ce-adj",
                str(adjacency),
                "--ce-ei",
                "",
                "--unknown-sign-policy",
                "random_unknown_sign_matched",
                "--ce-unknown-sign-weights",
                str(unknown),
                "--unknown-sign-inhibitory-frac",
                fraction,
                "--out-dir",
                str(tmp_path / "results"),
                "--n-repeats",
                "1",
                "--cuda",
                "-1",
            ],
        )
        cli.main()
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], captured[1])


def test_sign_er_historically_ignores_context_probability():
    from dataclasses import replace
    from cel_res_hypr.experiments.config import VariantContext, SimulationParams
    from cel_res_hypr.experiments.variants import run_variant

    W = np.ones((8, 8), dtype=np.float32) - np.eye(8, dtype=np.float32)
    W[:3] *= -1
    ctx = VariantContext(
        W,
        None,
        10,
        [(0.8, 0.6, 0.5, 0.0)],
        torch.device("cpu"),
        5,
        0,
        per_neg=0.25,
        rank_only=True,
        sim_params=SimulationParams(kr_num_streams=12, gr_num_streams=12),
    )
    a = run_variant("sign_test_er0.25", ctx)[0]
    b = run_variant("sign_test_er0.25", replace(ctx, er_p=0.8))[0]
    for x, y in zip(a, b):
        if isinstance(x, float) and np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y
