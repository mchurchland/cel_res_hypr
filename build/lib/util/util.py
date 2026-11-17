"""Compatibility exports for the original utility module.

New code should import the responsible cel_res_hypr module directly. All
original helper names, including private helpers used by research scripts,
remain available here.
"""

from cel_res_hypr.connectome import (
    _connectome_node_path_candidates,
    _connectome_unknown_sign_path_candidates,
    _load_connectome_names,
    load_connectome_node_names,
    load_unknown_sign_weights,
    assign_random_unknown_signs,
    load_connectome,
    UNKNOWN_SIGN_INHIBITORY_FRACTION,
)

from cel_res_hypr.networks.topology import (
    _match_edge_count,
    er_adjacency,
    ws_directed_adjacency_exact,
    degree_matched_shuffle_directed,
)

from cel_res_hypr.networks.weights import (
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

from cel_res_hypr.networks.normalization import (
    spectral_radius_power,
    scale_to_sr,
    normalize_reservoir_weights,
    spectral_norm,
)

from cel_res_hypr.networks.builder import (
    build_reservoir,
)

from cel_res_hypr.reproducibility import (
    set_seed,
)
