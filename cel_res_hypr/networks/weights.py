"""Edge-weight and sign controls that preserve historical RNG call order."""

import numpy as np

from .topology import degree_matched_shuffle_directed


def _count_edges(A: np.ndarray) -> int:
    M = np.abs(A) > 0

    return int(M.sum())


def negative_edge_fraction(W: np.ndarray) -> float:
    nz = int(np.count_nonzero(W))
    if nz == 0:
        return 0.0
    return float(np.count_nonzero(W < 0)) / float(nz)


def _shuffle_ce_weights(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = Wbio.copy().astype(np.float32)
    nz = np.nonzero(W)
    vals = W[nz].copy()
    rng.shuffle(vals)
    W[nz] = vals
    return W


def _shuffle_ce_weights_except_1(
    Wbio: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Retain the unused historical row-indexing behavior; see docs/AUDIT.md.

    This helper does not reliably preserve graph support or the weight multiset.
    Use _shuffle_ce_weights for the production nonzero-edge shuffle.
    """
    W = Wbio.copy().astype(np.float32)
    nz = np.where(np.abs(W) > 1)[0]
    vals = W[nz].copy()
    rng.shuffle(vals)
    W[nz] = vals
    return W


def _conn_and_w_shuffle_ce(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degree-matched shuffle of CE adjacency with CE weight multiset."""
    W_s = degree_matched_shuffle_directed(
        Wbio.astype(np.float32), tries=20_000, rng=rng
    ).astype(np.float32)
    W = _shuffle_ce_weights(W_s, rng)
    return W


def _conn_shuffle_ce(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degree-matched shuffle of CE adjacency with CE weight multiset."""
    W_s = degree_matched_shuffle_directed(
        Wbio.astype(np.float32), tries=20_000, rng=rng
    ).astype(np.float32)
    np.fill_diagonal(W_s, 0.0)
    return W_s


def _sample_from_cel(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = np.zeros_like(Wbio, dtype=np.float32)
    nz = np.nonzero(Wbio)
    sample_weights = Wbio[nz].astype(np.float32)
    W[nz] = rng.choice(sample_weights, sample_weights.size, replace=True)
    return W


def _sample_from_cel_sign(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = np.zeros_like(Wbio, dtype=np.float32)
    pos_idx = Wbio > 0
    neg_idx = Wbio < 0
    pos_weights = Wbio[pos_idx].astype(np.float32)
    neg_weights = Wbio[neg_idx].astype(np.float32)

    W[pos_idx] = rng.choice(pos_weights, pos_weights.size, replace=True)
    W[neg_idx] = rng.choice(neg_weights, neg_weights.size, replace=True)

    return W


def _cel_to_bin(Wbio: np.ndarray) -> np.ndarray:
    return Wbio.astype(bool).astype(np.float32)


def _signed_binary_from_cel(Wbio: np.ndarray) -> np.ndarray:
    W = _cel_to_bin(Wbio)
    W[Wbio < 0] *= -1.0
    return W.astype(np.float32, copy=False)


def _binary_to_cel_interpolation(
    Wbio: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    *,
    shuffle_magnitudes: bool = False,
) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("binary-to-CEL interpolation alpha must be between 0 and 1.")

    Wbio = Wbio.astype(np.float32, copy=False)
    W = np.zeros_like(Wbio, dtype=np.float32)
    mask = Wbio != 0
    signs = np.sign(Wbio[mask]).astype(np.float32)
    magnitudes = np.abs(Wbio[mask]).astype(np.float32)
    if shuffle_magnitudes:
        magnitudes = magnitudes.copy()
        rng.shuffle(magnitudes)

    interpolated_magnitudes = (1.0 - alpha) + alpha * magnitudes
    W[mask] = signs * interpolated_magnitudes
    return W


def _cel_to_shuffled_cel_interpolation(
    Wbio: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            "CE-to-shuffled-CE interpolation alpha must be between 0 and 1."
        )

    Wbio = Wbio.astype(np.float32, copy=False)
    W = np.zeros_like(Wbio, dtype=np.float32)
    mask = Wbio != 0
    signs = np.sign(Wbio[mask]).astype(np.float32)
    magnitudes = np.abs(Wbio[mask]).astype(np.float32)
    shuffled_magnitudes = magnitudes.copy()
    rng.shuffle(shuffled_magnitudes)

    interpolated_magnitudes = (1.0 - alpha) * magnitudes + alpha * shuffled_magnitudes
    W[mask] = signs * interpolated_magnitudes
    return W


def apply_percent_negative(W: np.ndarray, per_neg: float, rng: np.random.Generator):
    nz = np.nonzero(W)
    n_neg = int(per_neg * len(nz[0]))
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    sel = (nz[0][idx[:n_neg]], nz[1][idx[:n_neg]])
    W = np.abs(W)
    W[sel] = -1 * W[sel]
    return W


def scale_weights(
    Wbio: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    *,
    preserve_sign: bool = False,
):
    assert alpha >= 0

    W = Wbio.astype(np.float32, copy=True)
    mask = W != 0
    if alpha != 0:
        # Add signed Gaussian noise to existing edges only.
        # Keeps sparsity pattern fixed while allowing sign/magnitude to change.
        noise = rng.normal(0.0, 1.0, size=mask.sum()).astype(np.float32)
        W[mask] = W[mask] + alpha * noise
    if preserve_sign:
        W[mask] = np.sign(Wbio[mask]).astype(np.float32) * np.abs(W[mask])
    return W
