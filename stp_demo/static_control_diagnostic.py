"""Reproduce the paired fixed-gain control and measure tanh saturation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from network_stats.run_one import run_reservoir_with_pre
from util.util import (
    assign_random_unknown_signs,
    build_reservoir,
    load_connectome,
    load_unknown_sign_weights,
)


torch.set_num_threads(8)
known, _, _ = load_connectome("Connectome/ce_adj.npy", "")
unknown = load_unknown_sign_weights(
    "Connectome/ce_adj.npy", "Connectome/ce_unknown_sign_weights.npy",
    n_nodes=known.shape[0],
)
trial = assign_random_unknown_signs(
    known, unknown, np.random.default_rng(12345 + 23_000_000)
)
W, Win, info = build_reservoir(
    target_sr=7.8165688,
    input_scale=1.0,
    seed=12346,
    feature_conn="cel",
    ce_W_bio=trial,
    DEVICE=torch.device("cpu"),
    normalization_mode="spectral_radius",
    return_info=True,
)
inputs = torch.rand(2500, 1) * 2.0 - 1.0
_, pre = run_reservoir_with_pre(W, Win, inputs, leak=0.6)
records = [{
    "step": 0,
    "effective_rho": float(info["post_rho"]),
    "fraction_abs_pre_ge_2": float("nan"),
}]
for end in range(100, 2501, 100):
    records.append({
        "step": end,
        "effective_rho": float(info["post_rho"]),
        "fraction_abs_pre_ge_2": float((pre[end - 100:end].abs() >= 2.0).float().mean().item()),
    })
pd.DataFrame(records).to_csv(
    Path(__file__).resolve().parent / "static_gain_u_saturation.csv", index=False
)
