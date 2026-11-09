"""Production grids, legacy array partitions, and disjoint node sampling."""

import itertools

import numpy as np
import torch


WS_K = 10  # signature compatibility with util.build_reservoir


ALL_JOB_KEYS = (
    "real",
    "cel_randN",
    "er_randN",
    "ws_p01_randN",
    "local_sign",
    "local_sign+flat",
    "local_sign+binary",
    "global_sign_pres_real_weight",
    "binary_base",
)


TOPOLOGY_SHUFFLE_JOB_KEYS = (
    "real",
    "shuffle_weights",
    "conn_shuf_only",
    "conn_shuf",
    "binary_base",
    "binary_base_topology_shuffle",
    "binary+shuffle",
    "binary+conshuffle+wshuffle",
    "global_sign_pres",
    "local_sign+binary",
)


WEIGHT_TEST_JOB_KEYS = (
    "weight_test",
    "weight_test_unsigned",
    "weight_test_signed",
    "weight_test_cel_to_shuffled_cel",
    "weight_test_binary_to_cel",
    "weight_test_binary_to_shuffled_cel",
)


SWEEP_SR = [0.6, 0.8, 0.95, 1.05]


SWEEP_LEAK = [0.6, 0.8, 1.0]


SWEEP_U = [0.1, 0.5, 1.0, 1.5]


SWEEP_NEURON_BIAS = [0.0, 0.1]


NORMALIZATION_MODES = ("spectral_radius", "original_radius")


def _build_col_params(
    sr_grid: list[float],
    leak_grid: list[float],
    u_grid: list[float],
    neuron_bias_grid: list[float],
) -> list[tuple[float, float, float, float]]:
    # Cartesian product of (spectral radius target, leak, input scale, neuron bias range)
    return [
        (sr, leak, u, neuron_bias)
        for sr, leak, u, neuron_bias in itertools.product(
            sr_grid, leak_grid, u_grid, neuron_bias_grid
        )
    ]


def _split_indices(n_total: int, split: int, rank: int) -> list[int]:
    """Return contiguous grid/repeat indices for the legacy array convention.

    Ranks 0..split-1 are zero-based. rank==split aliases the final partition;
    other ranks are invalid. Ambiguous ranks 1..split-1 are never shifted.
    Production launchers use zero-based array IDs. split<=1 returns all rows.
    """
    if n_total == 0:
        return []

    if split <= 1:
        # single job handles entire grid
        return list(range(n_total))

    # Normalize rank
    if not (0 <= rank < split):
        # try to interpret as 1-based
        if 1 <= rank <= split:
            rank = rank - 1
        else:
            raise ValueError(
                f"--rank must be in [0, {split - 1}] or [1, {split}] for --split={split}, got {rank}"
            )

    base = n_total // split
    rem = n_total % split

    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def _pick_device(cuda_index: int | None) -> torch.device:
    if cuda_index is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def _validate_subset_size(name: str, value: int | None, n_nodes: int) -> int:
    if value is None:
        raise ValueError(f"--{name} is required when --random-node-subsets is enabled.")
    if not (1 <= value <= n_nodes):
        raise ValueError(f"--{name} must be between 1 and {n_nodes}, got {value}.")
    return value


def _draw_node_subsets(
    n_nodes: int,
    k_in: int,
    k_out: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k_in + k_out > n_nodes:
        raise ValueError(
            f"Disjoint input/output subsets require k_in + k_out <= {n_nodes}; "
            f"got k_in={k_in}, k_out={k_out}."
        )
    rng = np.random.default_rng(seed)
    nodes = rng.choice(n_nodes, size=k_in + k_out, replace=False)
    input_idx = np.sort(nodes[:k_in]).astype(np.int64)
    output_idx = np.sort(nodes[k_in:]).astype(np.int64)
    return input_idx, output_idx
