"""Ten explicit static reservoir controls for a common edge-sign sweep.

W[source, target]; all sign fractions are imposed on EDGES. Raw CE magnitudes
are expressed in units of the completed connectome's mean nonzero magnitude.
The removal condition retains those exact units, rather than renormalizing its
surviving edges. Exact signed-multiset controls are matched within each fraction.
"""
from __future__ import annotations

import hashlib
import numpy as np
import networkx as nx
from scipy.linalg import eigvals
from util.util import ws_directed_adjacency_exact

FAMILIES = ('ce_empirical', 'ce_shuffled', 'ce_degree_rewired', 'ce_binary',
            'ce_halfnormal', 'er_signed_empirical', 'er_binary', 'er_halfnormal',
            'ws_signed_empirical', 'ce_unknown_removed')
NORMALIZATIONS = ('own_radius', 'fixed_reference', 'shared_ce_scale')
DEFINITIONS = {
    'ce_empirical': 'Completed CE topology and contact-count magnitudes; randomized edge-sign quota.',
    'ce_shuffled': 'CE topology and exact sign mask; shuffle magnitudes WITHIN sign classes, preserving exact signed multiset. Placements change across fractions.',
    'ce_degree_rewired': 'NetworkX directed swaps; preserve in/out degrees and each source signed outgoing-weight multiset; change targets.',
    'ce_binary': 'CE topology and same CE sign mask; all magnitudes one.',
    'ce_halfnormal': 'CE topology and same CE sign mask; absolute Gaussian magnitudes normalized to mean one.',
    'er_signed_empirical': 'Uniform loop-free directed fixed-edge-count random graph; exact signed CE empirical multiset.',
    'er_binary': 'Same ER mask/sign assignment as weighted ER; magnitudes one.',
    'er_halfnormal': 'Same ER mask; exact signed multiset of CE half-normal control.',
    'ws_signed_empirical': 'Directed WS-like ring with p=0.1 rewiring and random node relabeling; exact signed CE empirical multiset.',
    'ce_unknown_removed': 'Known-edge mask, 297 nodes retained, full-CE magnitude units retained; its own exact sign quota on remaining edges. Fewer edges, not density-matched to ER.',
}


def rng_for(seed, label):
    """Independent deterministic streams, unaffected by model/order selection."""
    key = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little')
    return np.random.default_rng(np.random.SeedSequence([int(seed), key & 0xffffffff, key >> 32]))


def validate_connectomes(known, unknown):
    known, unknown = np.asarray(known, float), np.asarray(unknown, float)
    if known.ndim != 2 or known.shape[0] != known.shape[1] or unknown.shape != known.shape:
        raise ValueError('Require equal square known/unknown connectomes')
    if not np.isfinite(known).all() or not np.isfinite(unknown).all() or (unknown < 0).any():
        raise ValueError('Nonfinite data or negative unknown magnitudes')
    if np.diag(known).any() or np.diag(unknown).any() or ((known != 0) & (unknown != 0)).any():
        raise ValueError('Require disjoint loop-free edge masks')
    if np.count_nonzero(known) == 0:
        raise ValueError('Known connectome is empty')
    return known, unknown


def _matrix(n, edges, values):
    w = np.zeros((n, n), dtype=np.float64)
    w[edges[:, 0], edges[:, 1]] = values
    return w


def make_assets(known, unknown, seed, swaps_per_edge=20, ws_probability=.1):
    known, unknown = validate_connectomes(known, unknown)
    if swaps_per_edge < 1 or not 0 <= ws_probability <= 1:
        raise ValueError('Invalid rewiring controls')
    magnitude = abs(known) + unknown
    edges = np.argwhere(magnitude != 0)
    n, m = len(known), len(edges)
    original_mean = float(magnitude[magnitude != 0].mean())
    values = magnitude[edges[:, 0], edges[:, 1]] / original_mean
    removed_indices = np.flatnonzero(known[edges[:, 0], edges[:, 1]] != 0)
    # Directed G(n,m), avoiding the original ER helper's sample-then-repair step.
    er_rng = rng_for(seed, 'ER fixed-edge topology')
    flat = er_rng.choice(n*(n-1), size=m, replace=False)
    er_source, compressed_target = flat // (n-1), flat % (n-1)
    er_target = compressed_target + (compressed_target >= er_source)
    er_edges = np.column_stack((er_source, er_target))
    er_map = rng_for(seed, 'ER parent weights').permutation(m)
    ws = ws_directed_adjacency_exact(n, m, ws_probability, rng_for(seed, 'WS directed topology'))
    relabel = rng_for(seed, 'WS node relabeling').permutation(n)
    ws = ws[np.ix_(relabel, relabel)]
    ws_edges = np.argwhere(ws != 0)
    ws_map = rng_for(seed, 'WS parent weights').permutation(m)
    graph = nx.from_numpy_array((magnitude != 0).astype(np.int8), create_using=nx.DiGraph)
    original_in = np.count_nonzero(magnitude, axis=0)
    original_out = np.count_nonzero(magnitude, axis=1)
    swaps = int(swaps_per_edge*m)
    nx.directed_edge_swap(graph, nswap=swaps, max_tries=max(1000, swaps*100),
                         seed=rng_for(seed, 'CE directed degree swaps'))
    rewired = nx.to_numpy_array(graph, nodelist=range(n), weight=None, dtype=np.int8)
    np.testing.assert_array_equal(np.count_nonzero(rewired, axis=0), original_in)
    np.testing.assert_array_equal(np.count_nonzero(rewired, axis=1), original_out)
    rewired_edges = np.argwhere(rewired)
    rewired_map = np.empty(m, dtype=np.int64)
    for source in range(n):
        target_positions = np.flatnonzero(rewired_edges[:, 0] == source)
        parent_positions = np.flatnonzero(edges[:, 0] == source)
        rewired_map[target_positions] = rng_for(seed, f'rewired parent weights {source}').permutation(parent_positions)
    halfnormal = np.abs(rng_for(seed, 'paired halfnormal magnitudes').normal(size=m))
    if (halfnormal == 0).any():
        raise ValueError('A half-normal draw was exactly zero; refusing to lose an edge')
    halfnormal /= halfnormal.mean()
    assets = dict(n_nodes=np.array(n), ce_edges=edges, magnitudes=values,
                  raw_ce_mean_magnitude=np.array(original_mean), known_indices=removed_indices,
                  ce_sign_order=rng_for(seed, 'CE nested sign order').permutation(m),
                  er_edges=er_edges, er_map=er_map, ws_edges=ws_edges, ws_map=ws_map,
                  rewired_edges=rewired_edges, rewired_map=rewired_map,
                  halfnormal=halfnormal,
                  reference_fraction=np.array(np.count_nonzero(known < 0)/np.count_nonzero(known)),
                  directed_swaps=np.array(swaps),
                  rewired_edge_retention=np.array(np.count_nonzero(rewired & (magnitude != 0))/m),
                  known_edge_count=np.array(len(removed_indices)))
    if np.diag(rewired).any() or graph.number_of_edges() != m:
        raise ValueError('Invalid degree-preserving null')
    return assets


def raw_networks(assets, fraction, seed):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Edge-sign fraction must be in [0,1]')
    n = int(assets['n_nodes'])
    values, edges, order = (assets[k] for k in ('magnitudes', 'ce_edges', 'ce_sign_order'))
    m = len(values)
    signs = np.ones(m)
    signs[order[:int(fraction*m)]] = -1
    empirical = values*signs
    shuffled = empirical.copy()
    # Within-sign shuffling retains CE's exact positive AND negative weights.
    for sign in [-1, 1]:
        mask = signs == sign
        shuffled[mask] = rng_for(seed, f'CE within-sign shuffle {sign}').permutation(empirical[mask])
    known = assets['known_indices']
    known_order = order[np.isin(order, known)]
    known_signs = np.ones(m)
    known_signs[known_order[:int(fraction*len(known))]] = -1
    networks = {
        'ce_empirical': _matrix(n, edges, empirical),
        'ce_shuffled': _matrix(n, edges, shuffled),
        'ce_binary': _matrix(n, edges, signs),
        'ce_halfnormal': _matrix(n, edges, assets['halfnormal']*signs),
        'ce_unknown_removed': _matrix(n, edges[known], values[known]*known_signs[known]),
        'ce_degree_rewired': _matrix(n, assets['rewired_edges'], empirical[assets['rewired_map']]),
        'er_signed_empirical': _matrix(n, assets['er_edges'], empirical[assets['er_map']]),
        'er_binary': _matrix(n, assets['er_edges'], signs[assets['er_map']]),
        'er_halfnormal': _matrix(n, assets['er_edges'], (assets['halfnormal']*signs)[assets['er_map']]),
        'ws_signed_empirical': _matrix(n, assets['ws_edges'], empirical[assets['ws_map']]),
    }
    parent = np.sort(empirical)
    for family in ['ce_shuffled', 'ce_degree_rewired', 'er_signed_empirical', 'ws_signed_empirical']:
        np.testing.assert_array_equal(np.sort(networks[family][networks[family] != 0]), parent)
    np.testing.assert_array_equal(np.sort(networks['ce_halfnormal'][networks['ce_halfnormal'] != 0]),
                                  np.sort(networks['er_halfnormal'][networks['er_halfnormal'] != 0]))
    return networks


def spectrum(matrix):
    values = eigvals(np.asarray(matrix, dtype=np.float64), check_finite=True)
    radius = float(np.abs(values).max())
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError('Zero/nilpotent/nonfinite spectral radius; no silent resampling')
    return values, radius


def scaling_factor(mode, target, own_radius, fixed_radius, ce_radius):
    if mode not in NORMALIZATIONS or not np.isfinite(target) or target <= 0:
        raise ValueError('Invalid scaling mode or target')
    reference = {'own_radius': own_radius, 'fixed_reference': fixed_radius,
                 'shared_ce_scale': ce_radius}[mode]
    if not np.isfinite(reference) or reference <= 1e-12:
        raise ValueError('Invalid normalization reference')
    return float(target/reference)
