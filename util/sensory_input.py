"""Sensory-label mapping and controlled placement of scalar input weights."""
import numpy as np
import pandas as pd


def sensory_annotation_map(metadata, node_names):
    """Conservative, explicit rule on the frozen WormAtlas-derived type labels.

    Include sensory-led labels and explicitly polymodal sensory neurons.
    Exclude motor-led/proprioceptive, putative motor/sensory, and interneuron-led
    labels. This is a defined sensory set, not an exhaustive biological claim.
    Map exact names to the adjacency order; never guess missing identities.
    """
    names = list(node_names)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate adjacency node names.")
    required = {"Entity1", "Relationship", "Entity2", "Evidence URL"}
    if not required.issubset(metadata.columns):
        raise ValueError("Missing annotation columns.")
    types = metadata[metadata.Relationship.str.lower().eq("type")].copy()
    text = types.Entity2.str.strip().str.lower()
    selected = types[text.str.startswith("sensory neuron") |
                     (text.str.startswith("polymodal neuron") & text.str.contains("sensory neuron", regex=False))]
    annotations = selected.groupby("Entity1", sort=True).agg(
        annotation=("Entity2", lambda x: " | ".join(sorted(set(x)))),
        evidence_url=("Evidence URL", lambda x: " | ".join(sorted(set(x.dropna())))),
    )
    missing = sorted(set(annotations.index)-set(names))
    rows = [dict(node_index=i, neuron=name, **annotations.loc[name].to_dict())
            for i,name in enumerate(names) if name in annotations.index]
    if not rows:
        raise ValueError("No sensory labels map to this adjacency.")
    return pd.DataFrame(rows), missing


def place_input_weights(dense_input, sensory_indices, target_indices, calibration):
    """Place the same ordered coefficients on any equal-size target subset.

    unchanged_coefficients preserves coefficients of the sensory entries in
    the original dense input vector. matched_total_norm rescales these to the
    full dense vector's L2 norm. No recurrent weights are changed.
    """
    dense = np.asarray(dense_input)
    if dense.ndim != 2 or dense.shape[1] != 1 or not np.isfinite(dense).all():
        raise ValueError("Require a finite N×1 input-weight array.")
    source, target = np.asarray(sensory_indices), np.asarray(target_indices)
    for indices in (source, target):
        if indices.ndim != 1 or len(indices)==0 or indices.dtype.kind not in "iu":
            raise ValueError("Require nonempty one-dimensional integer indices.")
        if len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(dense)):
            raise ValueError("Indices must be distinct and in bounds.")
    if len(source)!=len(target):
        raise ValueError("Source and target counts must match.")
    if calibration not in ("unchanged_coefficients", "matched_total_norm"):
        raise ValueError("Unknown input calibration.")
    coefficients = dense[source].copy()
    if np.linalg.norm(coefficients.astype(float))==0:
        raise ValueError("Selected coefficients have zero input norm.")
    if calibration=="matched_total_norm":
        coefficients *= np.linalg.norm(dense.astype(float))/np.linalg.norm(coefficients.astype(float))
    placed = np.zeros_like(dense)
    placed[target] = coefficients
    return placed
