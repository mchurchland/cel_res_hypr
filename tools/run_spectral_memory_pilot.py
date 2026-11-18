#!/usr/bin/env python3
"""Local, paired 20%-inhibitory Schur-mode intervention pilot (MC only)."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import scipy
import torch

from mc_narma_crossover import build_condition, load_completed_connectome
from network_stats.run_one import run_reservoir_states
from network_stats.stats import compute_MC
from util.spectral_intervention import (
    attenuate_blocks, real_schur_blocks, secondary_blocks, selected_norm,
)

FAMILIES = ("cel_matched", "matched_er")
SETTINGS = dict(ei_fraction=.2, rho=.95, leak=1., input_scale=.1, bias=0.,
                washout=500, train=1500, test=500, max_delay=30, ridge_alpha=1e-4,
                requested_dimensions=[2, 6, 12], multipliers=[.75, .5, 0.],
                random_controls=2, norm_matched_dimension=6, norm_budget_fractions=[.5, 1.],
                base_seed=20261117, repeat_stride=100000, intervention_seed=20261118)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_realization(arguments):
    repeat, out, settings = arguments
    out = Path(out)
    destination = out / "cache" / f"repeat_{repeat:03d}.csv"
    if destination.exists():
        return repeat, len(pd.read_csv(destination)), "cached"
    torch.set_num_threads(1)
    started = time.monotonic()
    seed = settings["base_seed"] + repeat * settings["repeat_stride"]
    completed, _ = load_completed_connectome(ROOT / "Connectome/ce_adj.npy",
                                             ROOT / "Connectome/ce_unknown_sign_weights.npy", seed)
    steps = settings["washout"] + settings["train"] + settings["test"]
    u = torch.from_numpy(np.random.default_rng(seed+31_000_003).uniform(-1, 1, steps).astype(np.float32)[:, None])
    records, curve_records = [], []
    saved = {"input": u.numpy()}
    for family_index, family in enumerate(FAMILIES):
        base, win, info = build_condition(completed, family, .2, "spectral_radius", seed, torch.device("cpu"))
        weights = (base * settings["rho"]).numpy()
        win = win * settings["input_scale"]
        # Repository dynamics use W.T @ state, so decompose A=W.T.
        a = weights.T.astype(np.float64)
        t, z, blocks = real_schur_blocks(a)
        saved.update({f"{family}_weights": weights, f"{family}_input_weights": win.numpy(),
                      f"{family}_schur_t": t, f"{family}_schur_z": z})
        largest = max(b.modulus for b in blocks)
        baseline_score = None

        def evaluate(chosen, multiplier, protocol, strategy, requested=0, draw=-1,
                     budget_fraction=np.nan, pair_budget=np.nan):
            nonlocal baseline_score
            modified = attenuate_blocks(a, t, z, blocks, chosen, multiplier)
            matrix = np.ascontiguousarray(modified.T, dtype=np.float32)
            x = run_reservoir_states(torch.from_numpy(matrix), win, u, settings["leak"])
            start, end = settings["washout"], settings["washout"]+settings["train"]
            total, curve = compute_MC(x[start:end].double(), x[end:].double(),
                                      u[start:end].double(), u[end:].double(), settings["max_delay"],
                                      settings["ridge_alpha"], torch.device("cpu"))
            if baseline_score is None:
                baseline_score = total
            after_eigs = np.linalg.eigvals(matrix.T.astype(np.float64))
            post_radius = float(np.max(np.abs(after_eigs)))
            change = float(np.linalg.norm(matrix.astype(float)-weights.astype(float), "fro"))
            expected_change = (1-multiplier) * selected_norm(blocks, chosen)
            if abs(post_radius-largest) > 2e-5:
                raise ValueError(f"Leading radius changed unexpectedly: {largest} -> {post_radius}")
            if abs(change-expected_change) > 2e-5:
                raise ValueError("Unexpected Frobenius perturbation size.")
            if not np.isfinite(total) or not np.isfinite(curve).all():
                raise ValueError("Nonfinite memory capacity.")
            condition_id = len(records)
            record = dict(condition_id=condition_id, repeat_id=repeat, seed=seed, family=family,
                          protocol=protocol, strategy=strategy, requested_dimension=requested,
                          actual_dimension=sum(blocks[j].size for j in chosen), draw=draw,
                          multiplier=multiplier, budget_fraction=budget_fraction, pair_budget=pair_budget,
                          selected_blocks=json.dumps(chosen),
                          selected_moduli=json.dumps([blocks[j].modulus for j in chosen]),
                          MC=total, baseline_MC=baseline_score, MC_loss=baseline_score-total,
                          delta_frobenius=change, expected_delta_frobenius=expected_change,
                          relative_delta_frobenius=change/np.linalg.norm(weights, "fro"),
                          post_rho=post_radius, baseline_rho=largest,
                          mean_eigenvalue_modulus=float(np.mean(np.abs(after_eigs))),
                          saturation_fraction=float((x[start:].abs()>.9).float().mean()),
                          mean_state_variance=float(x[start:].var(dim=0, unbiased=False).mean()),
                          new_edge_fraction=float(np.mean((weights==0)&(matrix!=0))),
                          original_edge_sign_change_fraction=float(np.mean(np.sign(matrix[weights!=0])!=np.sign(weights[weights!=0]))),
                          raw_rho=info["raw_rho"])
            records.append(record)
            curve_records.extend(dict(condition_id=condition_id, repeat_id=repeat, family=family,
                                      delay=i+1, capacity=float(value)) for i,value in enumerate(curve))

        evaluate([], 1., "baseline", "baseline")
        for requested in settings["requested_dimensions"]:
            target = secondary_blocks(blocks, requested)
            smallest = secondary_blocks(blocks, requested, "smallest", pattern=target)
            selections = {"largest": target, "smallest": smallest}
            for draw in range(settings["random_controls"]):
                # Independently reproducible selection; pair same choices across doses.
                rng = np.random.default_rng(settings["intervention_seed"]+repeat*10000+family_index*1000+requested*10+draw)
                selections[f"random_{draw}"] = secondary_blocks(blocks, requested, "random", rng, target)
            for strategy, chosen in selections.items():
                for multiplier in settings["multipliers"]:
                    evaluate(chosen, multiplier, "equal_fraction", "random" if strategy.startswith("random") else strategy,
                             requested, int(strategy.split("_")[1]) if strategy.startswith("random") else -1)
            if requested == settings["norm_matched_dimension"]:
                for draw in range(settings["random_controls"]):
                    control = selections[f"random_{draw}"]
                    norm_target, norm_control = selected_norm(blocks, target), selected_norm(blocks, control)
                    for fraction in settings["norm_budget_fractions"]:
                        budget = fraction * min(norm_target, norm_control)
                        for strategy, chosen, norm in (("largest", target, norm_target), ("random", control, norm_control)):
                            multiplier = max(0., 1-budget/norm) if norm else 1.
                            evaluate(chosen, multiplier, "equal_frobenius", strategy,
                                     requested, draw, fraction, budget)
    pd.DataFrame(curve_records).to_csv(out / "cache" / f"repeat_{repeat:03d}_curves.csv", index=False)
    np.savez_compressed(out / "cache" / f"repeat_{repeat:03d}_networks.npz", **saved)
    pd.DataFrame(records).to_csv(destination, index=False)
    return repeat, len(records), f"{time.monotonic()-started:.1f}s"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT/"analysis/spectral_memory_intervention_ei20_2026-11-18")
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.n_repeats <= 0 or args.workers <= 0:
        p.error("Repeats and workers must be positive.")
    out = args.out_dir.resolve()
    (out/"cache").mkdir(parents=True, exist_ok=True)
    sources = ["Connectome/ce_adj.npy", "Connectome/ce_unknown_sign_weights.npy"]
    code = ["tools/run_spectral_memory_pilot.py", "util/spectral_intervention.py", "mc_narma_crossover.py",
            "util/util.py", "network_stats/run_one.py", "network_stats/stats.py", "network_stats/stats_util.py"]
    manifest = dict(settings=SETTINGS, n_repeats=args.n_repeats, workers=args.workers, device="cpu",
                    packages=dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__),
                    python=platform.python_version(),
                    sources={name:digest(ROOT/name) for name in sources},
                    code={name:digest(ROOT/name) for name in code},
                    interpretation="Exploratory model intervention; no global renormalization; leading block unchanged.",
                    control_details="Same singleton/pair counts. Frobenius budget per target/random pair equals fraction*min(selected block norms). Random sets may overlap target.",
                    cpu_note="All baselines rerun locally: CPU input weights need not match the original CUDA draw at the same seed.",
                    outputs="PNG/CSV/NPZ/Markdown only; no PDFs.")
    path = out/"manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[key]!=manifest[key] for key in ("settings", "n_repeats", "sources", "code")):
            raise ValueError("Existing run differs; choose a new output folder.")
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")
    jobs = [(repeat, str(out), SETTINGS) for repeat in range(args.n_repeats)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_realization, job) for job in jobs]
        for future in as_completed(futures):
            repeat, rows, duration = future.result()
            print(f"[repeat {repeat:02d}] {rows} conditions; {duration}", flush=True)
    results = pd.concat([pd.read_csv(out/"cache"/f"repeat_{r:03d}.csv") for r in range(args.n_repeats)], ignore_index=True)
    curves = pd.concat([pd.read_csv(out/"cache"/f"repeat_{r:03d}_curves.csv") for r in range(args.n_repeats)], ignore_index=True)
    results.to_csv(out/"interventions.csv", index=False)
    curves.to_csv(out/"memory_curves.csv", index=False)
    after = {name:digest(ROOT/name) for name in sources}
    if after != manifest["sources"]:
        raise ValueError("Input source changed during pilot.")
    print(f"Completed {len(results)} MC evaluations; {len(curves)} delay scores. Saved {out}", flush=True)


if __name__ == "__main__":
    main()
