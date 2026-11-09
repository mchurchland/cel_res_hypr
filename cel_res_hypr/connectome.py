"""Load processed connectome arrays and assign unknown edge signs.

Adjacency convention: W[i, j] is the weight from source i to target j.
Loading returns float32 weights and removes diagonal/nonfinite entries.
"""
import os
from pathlib import Path

import numpy as np

UNKNOWN_SIGN_INHIBITORY_FRACTION = 0.2425287356321839

def _connectome_node_path_candidates(adj_path: str) -> list[Path]:
    adj = Path(adj_path)
    stem = adj.stem
    stems: list[str] = []

    if "_adj" in stem:
        stems.append(stem.replace("_adj", "_nodes", 1))
    stems.append("ce_nodes")

    unique_stems = list(dict.fromkeys(stems))
    return [
        adj.with_name(f"{node_stem}{suffix}")
        for node_stem in unique_stems
        for suffix in (".npy", ".txt")
    ]


def _connectome_unknown_sign_path_candidates(adj_path: str) -> list[Path]:
    adj = Path(adj_path)
    stem = adj.stem
    stems: list[str] = []

    if "_adj" in stem:
        stems.append(stem.replace("_adj", "_unknown_sign_weights", 1))
    if stem.startswith("ce_adj_"):
        suffix = stem[len("ce_adj_"):]
        stems.append(f"ce_{suffix}_unknown_sign_weights")
    if stem.endswith("_adj"):
        stems.append(f"{stem[:-4]}_unknown_sign_weights")
    stems.append(f"{stem}_unknown_sign_weights")

    unique_stems = list(dict.fromkeys(stems))
    return [adj.with_name(f"{unknown_stem}.npy") for unknown_stem in unique_stems]


def _load_connectome_names(adj_path: str, n_nodes: int) -> np.ndarray | None:
    for path in _connectome_node_path_candidates(adj_path):
        if not path.is_file():
            continue
        if path.suffix == ".npy":
            names = np.load(path)
        else:
            with path.open("r", encoding="utf-8") as f:
                names = np.array([ln.strip() for ln in f if ln.strip()])
        if len(names) == n_nodes:
            return names
    return None


def load_connectome_node_names(adj_path: str, n_nodes: int) -> list[str]:
    names = _load_connectome_names(adj_path, n_nodes)
    if names is None:
        return [str(i) for i in range(n_nodes)]
    out = [str(name) if str(name) else str(i) for i, name in enumerate(names)]
    if len(out) < n_nodes:
        out.extend(str(i) for i in range(len(out), n_nodes))
    return out[:n_nodes]


def load_unknown_sign_weights(
    adj_path: str | None,
    unknown_path: str | None = None,
    *,
    n_nodes: int | None = None,
) -> np.ndarray | None:
    """Load magnitudes for new-connectome complex/no-pred edges, if present."""
    candidates: list[Path] = []
    if unknown_path:
        candidates.append(Path(unknown_path))
    elif adj_path is not None:
        candidates.extend(_connectome_unknown_sign_path_candidates(adj_path))

    for path in candidates:
        if not path.is_file():
            continue
        W = np.load(path).astype(np.float32, copy=False)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("Unknown-sign weight matrix must be a square 2D array.")
        if n_nodes is not None and W.shape != (n_nodes, n_nodes):
            raise ValueError(
                "Unknown-sign weight matrix shape must match adjacency: "
                f"got {W.shape}, expected {(n_nodes, n_nodes)}."
            )
        if not np.isfinite(W).all():
            W = np.where(np.isfinite(W), W, 0.0).astype(np.float32, copy=False)
        W = np.abs(W).astype(np.float32, copy=False)
        np.fill_diagonal(W, 0.0)
        return W
    return None


def assign_random_unknown_signs(
    W_known: np.ndarray,
    W_unknown_magnitudes: np.ndarray | None,
    rng: np.random.Generator,
    *,
    inhibitory_fraction: float = UNKNOWN_SIGN_INHIBITORY_FRACTION,
) -> np.ndarray:
    """
    Add complex/no-pred edges back with random signs once per trial.

    ``inhibitory_fraction`` controls the edge-level negative-sign fraction
    among the formerly unknown-sign edges.
    """
    W = W_known.copy().astype(np.float32)
    if W_unknown_magnitudes is None:
        return W
    if W_unknown_magnitudes.shape != W.shape:
        raise ValueError(
            "Unknown-sign weight matrix shape must match known adjacency: "
            f"{W_unknown_magnitudes.shape} vs {W.shape}."
        )
    if not (0.0 <= inhibitory_fraction <= 1.0):
        raise ValueError("inhibitory_fraction must be between 0 and 1.")

    mags = np.abs(W_unknown_magnitudes).astype(np.float32, copy=False)
    nz = np.nonzero(mags)
    n_unknown = len(nz[0])
    if n_unknown == 0:
        return W

    n_inhibitory = int(round(inhibitory_fraction * n_unknown))
    order = rng.permutation(n_unknown)
    signs = np.ones(n_unknown, dtype=np.float32)
    signs[order[:n_inhibitory]] = -1.0
    W[nz] += signs * mags[nz]
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32, copy=False)


def load_connectome(adj_path: str | None, ei_path: str | None):
    """
    Load C. elegans adjacency and EI labels (Varshney et al., 2011, PLoS Comput. Biol. 7:e1001066).
    See: https://github.com/OpenWorm/CElegansNeuroML/blob/master/CElegansNeuronTables.xlsx

    Returns:
      W_bio: np.ndarray [N,N] or None
      ei_labels: np.ndarray [N] with values in {-1,0,+1} or None
      name2idx: dict[str,int] mapping neuron name -> index, or None

    Behavior:
      - Replaces NaN/inf in W with 0, zeros self-loops, dtype float32.
      - EI kept in {-1,0,+1}; tiny values -> 0; dtype float32.
      - If a matching names file exists alongside the adjacency, builds name2idx.
        For ce_adj_new.npy, looks for ce_nodes_new.npy/txt before ce_nodes.npy/txt.
    """
    W_bio, ei_labels, name2idx = None, None, None

    # ---- adjacency ----
    if adj_path is not None and os.path.isfile(adj_path):
        W_bio = np.load(adj_path)
        if W_bio.ndim != 2 or W_bio.shape[0] != W_bio.shape[1]:
            raise ValueError("CE adjacency must be a square 2D array.")
        W_bio = W_bio.astype(np.float32, copy=False)

        # clean numerics & remove self-loops
        if not np.isfinite(W_bio).all():
            W_bio = np.where(np.isfinite(W_bio), W_bio, 0.0).astype(np.float32, copy=False)
        np.fill_diagonal(W_bio, 0.0)

        names = _load_connectome_names(adj_path, W_bio.shape[0])

        if names is not None:
            # build mapping
            name2idx = {str(n): i for i, n in enumerate(names)}

    # ---- EI labels ----
    if ei_path is not None and os.path.isfile(ei_path):
        ei_labels = np.load(ei_path).astype(np.float32, copy=False)
        if ei_labels.ndim != 1:
            raise ValueError("EI labels must be a 1D array.")
        if W_bio is not None and ei_labels.shape[0] != W_bio.shape[0]:
            raise ValueError("EI labels length must match adjacency size.")
        # sanitize: tiny -> 0, then sign to {-1,0,+1}
        ei_clean = ei_labels.copy()
        ei_clean[np.abs(ei_clean) < 1e-6] = 0.0
        ei_clean = np.sign(ei_clean).astype(np.float32, copy=False)
        ei_labels = ei_clean

    return W_bio, ei_labels, name2idx
