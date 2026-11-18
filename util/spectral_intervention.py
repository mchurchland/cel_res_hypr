"""Real Schur interventions on the actual column-state transition matrix.

Only selected 1x1/2x2 diagonal blocks are attenuated. All off-block Schur
couplings and unselected eigenvalues are retained. These interventions generally
change topology/signs; they are model diagnostics, not biological manipulations.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import schur


@dataclass(frozen=True)
class SchurBlock:
    start: int
    size: int
    modulus: float
    norm: float


def real_schur_blocks(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("Require a finite square transition matrix.")
    t, z = schur(a, output="real")
    blocks = []
    i = 0
    while i < len(t):
        # LAPACK explicitly zeros entries outside the real Schur blocks.
        size = 2 if i + 1 < len(t) and t[i + 1, i] != 0 else 1
        diagonal = t[i:i+size, i:i+size]
        blocks.append(SchurBlock(i, size, float(np.max(np.abs(np.linalg.eigvals(diagonal)))),
                                 float(np.linalg.norm(diagonal, "fro"))))
        i += size
    return t, z, blocks


def secondary_blocks(blocks, requested, strategy="largest", rng=None, pattern=None):
    """Exclude the leading block and preserve all conjugate pairs.

    Largest selection reaches/exceeds the requested real dimension. Controls
    match its exact numbers of real singleton and conjugate-pair blocks.
    """
    if requested <= 0 or len(blocks) < 2:
        raise ValueError("Require positive dimension and secondary blocks.")
    leader = max(range(len(blocks)), key=lambda j: blocks[j].modulus)
    candidates = [j for j in range(len(blocks)) if j != leader]
    if strategy == "largest":
        chosen, dimension = [], 0
        for j in sorted(candidates, key=lambda j: blocks[j].modulus, reverse=True):
            chosen.append(j)
            dimension += blocks[j].size
            if dimension >= requested:
                return chosen
        raise ValueError("Not enough secondary dimensions.")
    if strategy not in ("random", "smallest") or pattern is None:
        raise ValueError("Controls require random/smallest strategy and a target pattern.")
    rng = np.random.default_rng() if rng is None else rng
    chosen = []
    for size in (1, 2):
        count = sum(blocks[j].size == size for j in pattern)
        pool = [j for j in candidates if blocks[j].size == size]
        if strategy == "random":
            chosen.extend(rng.choice(pool, size=count, replace=False).tolist())
        else:
            chosen.extend(sorted(pool, key=lambda j: blocks[j].modulus)[:count])
    if len(chosen) != len(pattern):
        raise ValueError("Cannot match target block sizes.")
    return chosen


def selected_norm(blocks, chosen):
    return float(np.sqrt(sum(blocks[j].norm**2 for j in chosen)))


def attenuate_blocks(matrix, t, z, blocks, chosen, multiplier):
    """Return A + Q delta(T) Q.T without introducing baseline roundoff.

    Unitary invariance gives ||delta(A)||_F=(1-multiplier)*selected_norm.
    Zero multiplier changes eigenvalues to zero; it does not delete every
    transient pathway involving the corresponding Schur coordinates.
    """
    if not np.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError("Multiplier must be in [0,1].")
    if len(set(chosen)) != len(chosen) or any(j < 0 or j >= len(blocks) for j in chosen):
        raise ValueError("Require distinct valid block indices.")
    a = np.asarray(matrix, dtype=np.float64)
    modified = a.copy()
    for j in chosen:
        b = blocks[j]
        q = z[:, b.start:b.start+b.size]
        diagonal = t[b.start:b.start+b.size, b.start:b.start+b.size]
        modified += (multiplier - 1) * (q @ diagonal @ q.T)
    return modified
