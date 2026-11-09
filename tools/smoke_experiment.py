"""Run a small, seeded CPU experiment without reading or changing research data.

From the repository root: python -m tools.smoke_experiment --out-dir /tmp/cel-smoke
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from network_stats.run_one import run_reservoir_states
from reservoir_variants import (
    CSV_HEADER, SimulationParams, VariantContext, run_variant, save_rows,
)

SEED = 20261109
VARIANTS = (
    "real", "shuffle_weights", "cel_randN", "er_randN", "ws_p01_randN",
    "conn_shuf", "conn_shuf_only", "cel_sample", "local_sign",
    "local_sign+flat", "local_sign+sample", "local_sign+binary",
    "global_sign_pres", "global_sign_pres_real_weight", "binary_base",
    "binary_base_topology_shuffle", "binary+shuffle", "binary+conshuffle+wshuffle",
    "sign_test_cel0.25", "sign_test_er0.25", "sign_test_og_cel0.25",
    "weight_test0.25", "weight_test_signed0.25", "weight_test_unsigned0.25",
    "weight_test_binary_to_cel0.25", "weight_test_binary_to_shuffled_cel0.25",
    "weight_test_cel_to_shuffled_cel0.25",
)


def small_experiment(out_dir: Path) -> dict:
    """Evaluate 27 controls under both normalizations on one 8-node graph.

    Uses the production variant/evaluation/CSV pipeline with shorter time series.
    The fixture is a regression baseline, not an estimate of scientific capacity.
    """
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    adjacency = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for offset in (1, 3, 5):
            adjacency[i, (i + offset) % 8] = rng.normal()
    sim = SimulationParams(
        washout=12, t_train=80, t_test=50, mc_max_delay=3,
        ipc_max_delay=3, ipc_orders=[1, 2, 3], kr_num_streams=16,
        gr_num_streams=16,
    )
    rows = []
    for normalization in ("spectral_radius", "original_radius"):
        ctx = VariantContext(
            ce_W_bio=adjacency, ce_ei=None, ws_k=10,
            col_params=[(0.8, 0.6, 0.5, 0.1)], device=torch.device("cpu"),
            seed=SEED, sid=2, sim_params=sim, src_tag="cpu_baseline",
            per_neg=0.25, alpha=0.25, normalization_mode=normalization,
        )
        for key in VARIANTS:
            rows.extend(run_variant(key, ctx))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rows(str(out_dir / "invariance_variants.csv"), rows)
    W = torch.tensor([[0.0, 0.4], [-0.2, 0.1]])
    Win = torch.tensor([[0.5], [-0.3]])
    u = torch.tensor([[0.2], [-0.4], [0.7], [0.0]])
    trajectory = run_reservoir_states(W, Win, u, 0.6, torch.tensor([0.1, -0.1]))
    result = {
        "seed": SEED, "device": "cpu", "threads": 1,
        "python": platform.python_version(),
        "dependencies": {p: importlib.metadata.version(p) for p in (
            "torch", "numpy", "pandas", "scipy", "scikit-learn", "networkx",
            "matplotlib", "pingouin", "pytest",
        )},
        "simulation": asdict(sim), "grid": [[0.8, 0.6, 0.5, 0.1]],
        "adjacency": adjacency.tolist(), "trajectory": trajectory.tolist(),
        "columns": CSV_HEADER, "rows": [list(row) for row in rows],
    }
    (out_dir / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = small_experiment(args.out_dir)
    print(f"Wrote {len(result['rows'])} rows to {args.out_dir}")
    print(dict(zip(CSV_HEADER, result["rows"][0])))


if __name__ == "__main__":
    main()
