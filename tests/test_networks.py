"""Small CPU tests of graph, sign, normalization, and loading invariants."""

import numpy as np
import pytest
import torch

from cel_res_hypr.connectome import (
    assign_random_unknown_signs,
    load_connectome,
    load_unknown_sign_weights,
)
from cel_res_hypr.networks.builder import build_reservoir
from cel_res_hypr.networks.normalization import (
    normalize_reservoir_weights,
    scale_to_sr,
    spectral_radius_power,
)
from cel_res_hypr.networks.topology import (
    degree_matched_shuffle_directed,
    ws_directed_adjacency_exact,
)
from cel_res_hypr.networks.weights import _shuffle_ce_weights


def adjacency():
    W = np.zeros((12, 12), dtype=np.float32)
    for i in range(12):
        for d in (1, 3, 5):
            W[i, (i + d) % 12] = (i + d + 1) * (-1 if d == 3 else 1)
    return W


@pytest.mark.parametrize(
    "feature",
    ["cel", "local_sign", "local_sign+flat", "local_sign+sample", "local_sign+binary"],
)
def test_local_controls_preserve_support_and_signs(feature):
    W = adjacency()
    result, _ = build_reservoir(None, 0.5, 7, feature, ce_W_bio=W)
    np.testing.assert_array_equal(result.numpy() != 0, W != 0)
    np.testing.assert_array_equal(np.sign(result.numpy()), np.sign(W))


def test_weight_shuffle_preserves_support_and_signed_multiset():
    W = adjacency()
    result = _shuffle_ce_weights(W, np.random.default_rng(4))
    np.testing.assert_array_equal(result != 0, W != 0)
    np.testing.assert_array_equal(np.sort(result[result != 0]), np.sort(W[W != 0]))
    np.testing.assert_array_equal(W, adjacency())


def test_directed_shuffle_preserves_both_degrees_and_weights():
    W = adjacency()
    result = degree_matched_shuffle_directed(
        W, tries=2000, rng=np.random.default_rng(5)
    )
    np.testing.assert_array_equal((result != 0).sum(axis=0), (W != 0).sum(axis=0))
    np.testing.assert_array_equal((result != 0).sum(axis=1), (W != 0).sum(axis=1))
    np.testing.assert_array_equal(np.sort(result[result != 0]), np.sort(W[W != 0]))
    assert not np.any(np.diag(result))


@pytest.mark.parametrize("probability", [0.0, 0.1, 1.0])
def test_ws_has_exact_edge_count_and_fixed_row_degrees(probability):
    initial = ws_directed_adjacency_exact(12, 37, 0.0, np.random.default_rng(8))
    result = ws_directed_adjacency_exact(12, 37, probability, np.random.default_rng(8))
    assert np.count_nonzero(result) == 37
    assert not np.any(np.diag(result))
    np.testing.assert_array_equal(result.sum(axis=1), initial.sum(axis=1))


@pytest.mark.parametrize("feature", ["er_p=0.1", "ws_p=0.1", "cel_randN"])
def test_builder_seed_and_edge_count(feature):
    W = adjacency()
    kwargs = dict(N=12, ce_W_bio=W, nnz_target=np.count_nonzero(W))
    first = build_reservoir(0.95, 0.5, 9, feature, **kwargs)
    torch.rand(23)  # Building must reseed, independent of the preceding global stream.
    second = build_reservoir(0.95, 0.5, 9, feature, **kwargs)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    assert torch.count_nonzero(first[0]) == np.count_nonzero(W)
    assert spectral_radius_power(first[0]) == pytest.approx(0.95, rel=2e-5, abs=2e-6)


def test_normalization_uses_selected_reference_and_zero_radius_convention():
    W = torch.tensor([[0.0, 2.0], [0.5, 0.0]])
    reference = W * 2
    own, own_info = normalize_reservoir_weights(W, 0.8, "spectral_radius", reference)
    fixed, fixed_info = normalize_reservoir_weights(
        W, 0.8, "original_radius", reference
    )
    torch.testing.assert_close(own, W * 0.8)
    torch.testing.assert_close(fixed, W * 0.4)
    assert own_info["post_rho"] == pytest.approx(0.8)
    assert fixed_info["post_rho"] == pytest.approx(0.4)
    zero = torch.zeros(2, 2)
    assert torch.equal(scale_to_sr(zero, 0.8), zero)
    assert scale_to_sr(W, None) is W
    with pytest.raises(ValueError, match="requires reference"):
        normalize_reservoir_weights(W, 0.8, "original_radius", None)


def test_connectome_cleaning_names_and_unknown_sign_assignment(tmp_path):
    path = tmp_path / "sample_adj.npy"
    np.save(path, [[4, np.nan, 2], [-3, 9, np.inf], [1, 0, 8]])
    np.save(tmp_path / "sample_nodes.npy", ["A", "B", "C"])
    np.save(
        tmp_path / "sample_unknown_sign_weights.npy", [[0, -2, 0], [0, 0, 3], [0, 0, 0]]
    )
    W, ei, names = load_connectome(str(path), None)
    np.testing.assert_array_equal(W, [[0, 0, 2], [-3, 0, 0], [1, 0, 0]])
    assert W.dtype == np.float32 and ei is None and names == {"A": 0, "B": 1, "C": 2}
    unknown = load_unknown_sign_weights(str(path), n_nodes=3)
    assigned = assign_random_unknown_signs(
        W, unknown, np.random.default_rng(6), inhibitory_fraction=0.5
    )
    assert np.count_nonzero(assigned[unknown != 0] < 0) == 1
    np.testing.assert_array_equal(np.abs(assigned[unknown != 0]), [2, 3])
    np.testing.assert_array_equal(W, [[0, 0, 2], [-3, 0, 0], [1, 0, 0]])


def test_legacy_util_names_are_original_function_objects():
    from util import util

    assert util.build_reservoir is build_reservoir
    assert util.load_connectome is load_connectome
    assert util.degree_matched_shuffle_directed is degree_matched_shuffle_directed
