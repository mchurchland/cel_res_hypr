#!/usr/bin/env python3
"""Focused MC/NARMA10 E/I sweep, preserving the existing sign-sweep models."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from network_stats.temporal_tasks import (
    TemporalTaskSettings, evaluate_temporal_tasks, make_task_data,
)
from util.util import (
    assign_random_unknown_signs, build_reservoir, negative_edge_fraction,
)

ROOT = Path(__file__).resolve().parent
FAMILIES = ("cel_matched", "matched_er")
NORMALIZATIONS = ("spectral_radius", "original_radius")
FRACTIONS = (.20, .24, .28, .32, .36, .40)
RHOS = (.6, .95, 1.05)
LEAKS = (.6, 1.)
INPUTS = (.1, .5, 1., 1.5)
PROTOCOL = "mc-narma10-crossover-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_repeats(total, split, rank):
    if total <= 0 or split <= 0 or not 0 <= rank < split:
        raise ValueError("Require positive repeats/split and 0 <= repeat-rank < repeat-split.")
    return list(range(rank, total, split))


def load_completed_connectome(known_path, unknown_path, seed):
    known = np.load(known_path).astype(np.float32)
    unknown = np.load(unknown_path).astype(np.float32)
    if known.ndim != 2 or known.shape[0] != known.shape[1] or unknown.shape != known.shape:
        raise ValueError("Known and unknown-sign matrices must be square and have the same shape.")
    if not np.isfinite(known).all() or not np.isfinite(unknown).all():
        raise ValueError("Connectome matrices must be finite; no automatic filtering.")
    if not np.count_nonzero(known) or np.any(unknown < 0):
        raise ValueError("Require known edges and nonnegative unknown-edge magnitudes.")
    if np.any((known != 0) & (unknown != 0)):
        raise ValueError("Known and unknown-sign edges must not overlap.")
    if np.any(np.diag(known)) or np.any(np.diag(unknown)):
        raise ValueError("This matched ER experiment requires loop-free input matrices.")
    fraction = negative_edge_fraction(known)
    completed = assign_random_unknown_signs(
        known, unknown, np.random.default_rng(seed + 23_000_000),
        inhibitory_fraction=fraction,
    )
    return completed, fraction


def build_condition(completed, family, fraction, normalization, seed, device):
    """Build one unit-target/unit-input network, reused across the parameter grid.

    Existing build_reservoir preserves the CE weight magnitudes, uses the
    binary matched-ER sign-sweep control, and couples sign permutations across
    fractions. Fixed-reference ER uses its own topology at the CE sign fraction.
    """
    feature = {"cel_matched": "sign_test_og_cel", "matched_er": "sign_test_er"}[family]
    return build_reservoir(
        target_sr=1., input_scale=1., seed=seed + 1, feature_conn=feature,
        N=len(completed), ce_W_bio=completed,
        nnz_target=int(np.count_nonzero(completed)) if family == "matched_er" else None,
        DEVICE=device, per_neg=fraction, normalization_mode=normalization,
        normalization_ref=completed, return_info=True,
    )


def write_shard(path, rows):
    """Publish a complete CSV atomically; never replace an existing result."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # atomic and fails if another job already wrote it
    finally:
        temporary.unlink(missing_ok=True)


def validate_shard(path, grid, normalizations, experiment_id, repeat, family, fraction):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(grid) * len(normalizations) or any(r.get("experiment_id") != experiment_id for r in rows):
        raise ValueError(f"Incomplete or incompatible existing result: {path}")
    keys = [(r["normalization"], float(r["rho_target"]), float(r["leak"]), float(r["input_scale"])) for r in rows]
    expected = {(norm, *point) for norm in normalizations for point in grid}
    if set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError(f"Missing, duplicate or unexpected configurations in {path}")
    for row in rows:
        if (int(row["repeat_id"]) != repeat or row["family"] != family
                or float(row["negative_fraction"]) != fraction or float(row["neuron_bias"]) != 0):
            raise ValueError(f"Incorrect condition identifiers in {path}")
        for key in ("MC", "NARMA10_val_nmse", "NARMA10_test_nmse"):
            if not np.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key} in {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("results/crossover_mc_narma10/local"))
    p.add_argument("--ce-adj", type=Path, default=ROOT / "Connectome/ce_adj.npy")
    p.add_argument("--ce-unknown-sign-weights", type=Path, default=ROOT / "Connectome/ce_unknown_sign_weights.npy")
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--normalization-modes", nargs="+", choices=NORMALIZATIONS, default=list(NORMALIZATIONS))
    p.add_argument("--sign-fracs", nargs="+", type=float, default=list(FRACTIONS))
    p.add_argument("--rho-values", nargs="+", type=float, default=list(RHOS))
    p.add_argument("--leak-values", nargs="+", type=float, default=list(LEAKS))
    p.add_argument("--input-scales", nargs="+", type=float, default=list(INPUTS))
    p.add_argument("--n-repeats", type=int, default=200)
    p.add_argument("--repeat-split", type=int, default=1)
    p.add_argument("--repeat-rank", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--repeat-seed-stride", type=int, default=100000)
    p.add_argument("--cuda", type=int, default=-1, help="-1 for CPU; GPU requests fail if unavailable.")
    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--train-length", type=int, default=1500)
    p.add_argument("--validation-length", type=int, default=500)
    p.add_argument("--test-length", type=int, default=500)
    p.add_argument("--mc-max-delay", type=int, default=30)
    p.add_argument("--mc-ridge-alpha", type=float, default=1e-4)
    p.add_argument("--narma-burnin", type=int, default=1000)
    p.add_argument("--narma-ridge-alphas", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-2, 1.])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print counts; no output or simulations.")
    p.add_argument("--merge", action="store_true", help="Merge a complete run below out-dir; do not simulate.")
    return p.parse_args(argv)


def experiment_spec(args, settings):
    from reservoirpy.datasets import narma

    code = ["mc_narma_crossover.py", "network_stats/temporal_tasks.py",
            "network_stats/run_one.py", "network_stats/stats.py", "network_stats/stats_util.py", "util/util.py"]
    packages = {name: version(name) for name in ("numpy", "torch", "scipy", "scikit-learn", "reservoirpy")}
    if packages["reservoirpy"] != "0.4.2":
        raise ValueError("This protocol requires reservoirpy==0.4.2; install requirements-crossover.txt.")
    spec = {
        "protocol": PROTOCOL, "families": args.families,
        "normalizations": args.normalization_modes, "sign_fracs": args.sign_fracs,
        "rho_values": args.rho_values, "leak_values": args.leak_values, "input_scales": args.input_scales,
        "neuron_bias": 0., "n_repeats": args.n_repeats, "seed": args.seed,
        "repeat_seed_stride": args.repeat_seed_stride, "tasks": asdict(settings),
        "seed_offsets": {"network": 1, "unknown_sign": 23_000_000, "mc_input": 31_000_003, "narma_input": 37_000_003},
        "sources": [{"path": str(p.resolve()), "sha256": sha(p)} for p in (args.ce_adj, args.ce_unknown_sign_weights)],
        "code_sha256": {name: sha(ROOT / name) for name in code},
        "packages": packages, "python": platform.python_version(),
        "narma_generator": "reservoirpy.datasets.narma(order=10,a1=.3,a2=.05,b=1.5,c=.1)",
        "narma_generator_sha256": hashlib.sha256(inspect.getsource(narma).encode()).hexdigest(),
        "narma_alignment": "package inputs u[9:-1] paired with returned y[10:]; discard target burn-in",
        "narma_readout": "sklearn Ridge; train-only fit; validation chooses alpha; test held out",
        "mc_readout": "existing compute_MC; float64 readout; float32 reservoir dynamics",
    }
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec, identity


def validate_args(args):
    for name in ("families", "normalization_modes", "sign_fracs", "rho_values", "leak_values", "input_scales"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate values in {name}")
    for values in (args.sign_fracs, args.rho_values, args.leak_values, args.input_scales):
        if not np.isfinite(values).all():
            raise ValueError("All grid values must be finite.")
    if any(not 0 <= x <= 1 for x in args.sign_fracs):
        raise ValueError("Sign fractions must lie in [0,1].")
    if any(not 0 < x <= 1 for x in args.leak_values):
        raise ValueError("Leak values must lie in (0,1].")
    if min(*args.rho_values, *args.input_scales) <= 0:
        raise ValueError("Radius targets and input scales must be positive.")
    if args.seed < 0 or args.repeat_seed_stride <= 0:
        raise ValueError("Require a nonnegative seed and positive repeat seed stride.")


def merge_run(root):
    import pandas as pd
    root = Path(root)
    manifests = list(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"No chunk manifests found below {root}")
    metadata = [json.loads(p.read_text()) for p in manifests]
    identity = metadata[0]["experiment_id"]
    if any(m["experiment_id"] != identity for m in metadata):
        raise ValueError("Cannot merge incompatible experiment manifests.")
    spec = metadata[0]["spec"]
    frames = [pd.read_csv(p) for p in sorted(root.rglob("repeat_*__*.csv"))]
    if not frames:
        raise ValueError("No completed result shards found.")
    frame = pd.concat(frames, ignore_index=True)
    grid_count = len(spec["rho_values"]) * len(spec["leak_values"]) * len(spec["input_scales"])
    expected = spec["n_repeats"] * len(spec["families"]) * len(spec["sign_fracs"]) * len(spec["normalizations"]) * grid_count
    keys = ["repeat_id", "family", "negative_fraction", "normalization", "rho_target", "leak", "input_scale"]
    if len(frame) != expected or frame.duplicated(keys).any() or set(frame.experiment_id) != {identity}:
        raise ValueError(f"Incomplete/duplicate/incompatible results: {len(frame)} rows, expected {expected}.")
    domains = {"repeat_id": range(spec["n_repeats"]), "family": spec["families"],
               "negative_fraction": spec["sign_fracs"], "normalization": spec["normalizations"],
               "rho_target": spec["rho_values"], "leak": spec["leak_values"], "input_scale": spec["input_scales"]}
    if any(not frame[column].isin(values).all() for column, values in domains.items()):
        raise ValueError("Unexpected configurations in result shards.")
    if not (frame.neuron_bias == 0).all() or not np.isfinite(frame[["MC", "NARMA10_val_nmse", "NARMA10_test_nmse"]]).all().all():
        raise ValueError("Nonzero reservoir bias or nonfinite scores in result shards.")
    path = root / "mc_narma_crossover.ALL.csv"
    write_shard(path, frame.sort_values(keys).to_dict("records"))
    print(f"Merged {len(frame)} rows into {path}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.merge:
        merge_run(args.out_dir)
        return
    validate_args(args)
    settings = TemporalTaskSettings(
        washout=args.washout, train=args.train_length, validation=args.validation_length,
        test=args.test_length, mc_max_delay=args.mc_max_delay, mc_ridge_alpha=args.mc_ridge_alpha,
        narma_burnin=args.narma_burnin, narma_ridge_alphas=tuple(args.narma_ridge_alphas),
    )
    repeats = partition_repeats(args.n_repeats, args.repeat_split, args.repeat_rank)
    completed, unknown_fraction = load_completed_connectome(args.ce_adj, args.ce_unknown_sign_weights, args.seed)
    spec, identity = experiment_spec(args, settings)
    grid = list(itertools.product(args.rho_values, args.leak_values, args.input_scales))
    per_repeat = len(grid) * len(args.families) * len(args.sign_fracs) * len(args.normalization_modes)
    summary = {"configurations_per_condition": len(grid), "fractions_percent": [round(f*100, 8) for f in args.sign_fracs],
               "families": args.families, "normalizations": args.normalization_modes,
               "assigned_repeats": repeats, "rows_this_task": per_repeat * len(repeats),
               "rows_full_run": per_repeat * args.n_repeats, "neuron_bias": 0,
               "known_edge_inhibitory_fraction": unknown_fraction, "completed_edges": int(np.count_nonzero(completed)),
               "experiment_id": identity}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    if args.cuda >= 0:
        if not torch.cuda.is_available() or args.cuda >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable; use --cuda -1 for CPU.")
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"experiment_id": identity, "spec": spec, "repeat_ids": repeats,
                "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
    manifest = args.out_dir / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != json.loads(json.dumps(metadata)):
            raise ValueError("Output already exists or settings changed. Use a new directory, or --resume with identical settings.")
    else:
        with manifest.open("x") as f:
            json.dump(metadata, f, indent=2)
    for repeat in repeats:
        seed = args.seed + repeat * args.repeat_seed_stride
        completed, unknown_fraction = load_completed_connectome(args.ce_adj, args.ce_unknown_sign_weights, seed)
        data = None
        for family, fraction in itertools.product(args.families, args.sign_fracs):
            percent_tag = f"{fraction * 100:.8f}".rstrip("0").rstrip(".")
            shard = args.out_dir / f"repeat_{repeat:04d}__{family}__ei_{percent_tag}.csv"
            if shard.exists():
                if not args.resume:
                    raise FileExistsError(shard)
                validate_shard(shard, grid, args.normalization_modes, identity, repeat, family, fraction)
                print(f"[resume] {shard.name}", flush=True)
                continue
            if data is None:
                data = make_task_data(settings, seed)
            rows = []
            for normalization in args.normalization_modes:
                weights, win, info = build_condition(completed, family, fraction, normalization, seed, device)
                if min(float(info["raw_rho"]), float(info["ref_rho"])) < 1e-9:
                    raise ValueError("Near-zero raw/reference radius; cannot interpret requested scaling.")
                actual_fraction = float(torch.count_nonzero(weights < 0)) / float(torch.count_nonzero(weights))
                for rho, leak, input_scale in grid:
                    scores = evaluate_temporal_tasks(weights * rho, win * input_scale, leak, data, settings)
                    rows.append({
                        "experiment_id": identity, "family": family, "normalization": normalization,
                        "repeat_id": repeat, "seed": seed, "network_seed": seed + 1,
                        "negative_fraction": fraction, "actual_negative_fraction": actual_fraction,
                        "rho_target": rho, "leak": leak, "input_scale": input_scale, "neuron_bias": 0.,
                        "n_nodes": len(completed), "n_edges": int(torch.count_nonzero(weights)),
                        "unknown_sign_inhibitory_fraction": unknown_fraction,
                        "raw_rho": info["raw_rho"], "ref_rho": info["ref_rho"],
                        "post_rho": float(info["post_rho"]) * rho,
                        "scale_factor": float(info["scale_factor"]) * rho, **scores,
                    })
            write_shard(shard, rows)
            print(f"[saved] {shard.name}: {len(rows)} configurations", flush=True)


if __name__ == "__main__":
    main()
