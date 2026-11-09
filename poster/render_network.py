"""Render the actual reservoir using the thesis's connectome plotting code."""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurreps-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from architecture_variant_figures.make_weighted_graph_examples import (
    load_ce_with_project_code, build_sign_matched_unknown_connectome,
    compute_ce_kamada_layout, build_dual_colormap_norms, draw_weighted_panel,
    UNKNOWN_SIGN_INHIBITORY_FRACTION,
)

adj = str(HERE.parent / 'Connectome/ce_adj.npy')
known, _, _ = load_ce_with_project_code(adj, '')
weights = build_sign_matched_unknown_connectome(
    known, adj, None, seed=7,
    inhibitory_fraction=UNKNOWN_SIGN_INHIBITORY_FRACTION,
)
assert weights.shape == (297, 297)
assert np.count_nonzero(weights) == 3604
positions = compute_ce_kamada_layout(weights, seed=7, fallback_iters=150, layout_scale=1.5)
positive, negative = build_dual_colormap_norms(weights[weights != 0])
# Display only the largest weakly connected component; experiments use the full graph.
graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)
component = sorted(max(nx.weakly_connected_components(graph), key=len))
weights = weights[np.ix_(component, component)]
assert nx.is_weakly_connected(nx.from_numpy_array(weights, create_using=nx.DiGraph))
# Rotate the original layout without changing its relative geometry.
xy = np.array([positions[i] for i in component])
xy -= xy.mean(axis=0)
_, _, rotation = np.linalg.svd(xy, full_matrices=False)
xy = xy @ rotation.T
positions = dict(enumerate(xy))
plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(figsize=(8, 4.3))
draw_weighted_panel(
    ax, weights, positions, title='', panel_title_fontsize=14,
    node_colors=['#243e4c'] * len(weights), node_labels=None,
    label_fontsize=8, node_size=10, max_edges=0,
    keep_all_negatives=True, show_direction=False,
    pos_norm=positive['norm'], pos_cmap=positive['cmap'],
    neg_norm=negative['norm'], neg_cmap=negative['cmap'],
)
ax.set_aspect('equal')
fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
for extension in ('pdf', 'png'):
    fig.savefig(HERE / f'assets/connectome.{extension}', dpi=300)
print(f'Rendered {len(weights)} neurons, {np.count_nonzero(weights)} edges; '
      f'{np.count_nonzero(weights < 0)} negative edges. Unknown signs: seed 7.')
