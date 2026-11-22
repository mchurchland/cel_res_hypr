#!/usr/bin/env python3
"""Resumable GPU cluster experiment: continuous STP × weighted CE/ER controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import tempfile
import time

# GPU trajectories are batched; keep CPU readout BLAS from oversubscribing cores.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
import torch

from mc_narma_crossover import build_condition, load_completed_connectome, partition_repeats, write_shard
from network_stats.stats import compute_MC
from network_stats.stp_weight_controls import (
    FAMILIES, NORMALIZATIONS, PARAMETER_SETS, condition_grid, parameter_spec,
    run_states, scale_matrix, spectral_radius, stp_vectors, unsigned_controls, match_signed_weights,
)
from network_stats.temporal_tasks import TemporalTaskSettings, fit_narma_readout, make_task_data
from util.sensory_input import place_input_weights

ROOT = Path(__file__).resolve().parent
PROTOCOL = "continuous-stp-weight-controls-v2"
KEYS = ["repeat_id", "family", "ei_percent", "normalization", "input_scheme",
        "input_scale", "stp_condition", "gain_control"]
SCORES = ["MC", "NARMA10_val_nmse", "NARMA10_test_nmse"]
CODE = ["stp_weight_controls.py", "network_stats/stp_weight_controls.py",
        "network_stats/stp_weight_controls_cuda.py",
        "network_stats/short_term_plasticity.py", "network_stats/temporal_tasks.py",
        "network_stats/stats.py", "network_stats/stats_util.py", "mc_narma_crossover.py",
        "util/util.py", "util/sensory_input.py", "network_stats/run_one.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_bytes(path, contents):
    """Atomic, immutable publication, safe for different array tasks sharing a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(contents)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise ValueError(f"Existing incompatible file: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def publish_json(path, obj):
    publish_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


def publish_npz(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError(f"Incompatible network/input cache: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **arrays)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            publish_npz(path, arrays)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/stp_weight_controls/local")
    p.add_argument("--ce-adj", type=Path, default=ROOT / "Connectome/ce_adj.npy")
    p.add_argument("--ce-unknown-sign-weights", type=Path, default=ROOT / "Connectome/ce_unknown_sign_weights.npy")
    p.add_argument("--sensory-labels", type=Path, default=ROOT / "configs/stp_weight_controls/sensory_neurons.csv")
    p.add_argument("--node-names", type=Path, default=ROOT / "Connectome/ce_nodes.txt")
    p.add_argument("--ei-percent", type=float, nargs="+", default=[0, 10, 20, 30, 40, 50])
    p.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    p.add_argument("--normalizations", choices=NORMALIZATIONS, nargs="+", default=list(NORMALIZATIONS))
    p.add_argument("--input-scales", type=float, nargs="+", default=[.1, .5])
    p.add_argument("--input-schemes", choices=["full", "restricted_matched_norm"], nargs="+",
                   default=["full", "restricted_matched_norm"])
    p.add_argument("--parameter-sets", choices=list(PARAMETER_SETS), nargs="+", default=list(PARAMETER_SETS))
    p.add_argument("--mixed-fractions", type=float, nargs="+", default=[.5])
    p.add_argument("--gain-control", action="store_true", help="Optional per-neuron training-mean frozen-gain controls.")
    p.add_argument("--initial-radius", type=float, default=.95, help="Single initial target; never renormalize during STP.")
    p.add_argument("--n-repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--repeat-split", type=int, default=1)
    p.add_argument("--repeat-rank", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu",
                   help="Slurm uses cuda; cpu is retained as a local validation reference.")
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument("--gpu-batch-size", type=int, default=64,
                   help="Maximum independent trajectories per GPU kernel batch.")
    p.add_argument("--check-cuda", action="store_true", help="Compile/run a tiny GPU kernel and exit.")
    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--train-length", type=int, default=1500)
    p.add_argument("--validation-length", type=int, default=500)
    p.add_argument("--test-length", type=int, default=500)
    p.add_argument("--mc-max-delay", type=int, default=30)
    p.add_argument("--narma-burnin", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--merge", action="store_true")
    return p.parse_args(argv)


def make_spec(args):
    repeats = partition_repeats(args.n_repeats, args.repeat_split, args.repeat_rank)
    for key in ("ei_percent", "families", "normalizations", "input_scales", "input_schemes", "parameter_sets", "mixed_fractions"):
        values = getattr(args, key)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty/duplicate settings: {key}")
    if any(not np.isfinite(v) or not 0 <= v <= 100 for v in args.ei_percent):
        raise ValueError("E/I percentages must be in [0,100].")
    if any(not np.isfinite(v) or v <= 0 for v in args.input_scales) or not np.isfinite(args.initial_radius) or args.initial_radius <= 0:
        raise ValueError("Input scales and initial radius must be positive and finite.")
    if args.workers < 1 or args.seed < 0 or args.gpu_batch_size < 1 or args.cuda_device < 0:
        raise ValueError("Require positive workers and nonnegative seed.")
    if args.backend == "cuda" and args.workers != 1:
        raise ValueError("Use --workers 1 for one process per GPU; conditions are batched on the device.")
    task = TemporalTaskSettings(washout=args.washout, train=args.train_length,
                                validation=args.validation_length, test=args.test_length,
                                mc_max_delay=args.mc_max_delay, narma_burnin=args.narma_burnin)
    conditions = condition_grid(args.parameter_sets, args.mixed_fractions)
    sources = {"known": args.ce_adj, "unknown": args.ce_unknown_sign_weights}
    if "restricted_matched_norm" in args.input_schemes:
        sources.update(sensory_labels=args.sensory_labels, node_names=args.node_names)
    spec = dict(protocol=PROTOCOL, n_repeats=args.n_repeats, seed=args.seed, repeat_stride=100000,
                ei_percent=args.ei_percent, families=args.families, normalizations=args.normalizations,
                input_scales=args.input_scales, input_schemes=args.input_schemes,
                stp_conditions=conditions, stp_parameters=parameter_spec(),
                gain_control=args.gain_control, initial_radius=args.initial_radius,
                task=asdict(task), source_hashes={k: sha(v) for k, v in sources.items()},
                code_hashes={k: sha(ROOT / k) for k in CODE},
                rate_proxy="abs(previous signed tanh state)", stp_gain="u*x/U",
                time_units="one input sample = one uncalibrated model time unit",
                normalization_note="shared_ce_scale uses the CE empirical radius AT EACH E/I fraction; it is not fixed across E/I fractions",
                weight_matching="exact empirical multiset before scaling; raw mean nonzero magnitude=1 in every family",
                signed_control="er_signed_empirical matches the signed CE weight multiset separately at every fraction; magnitude placement may change across fractions",
                signs="randomize ALL chemical edges with nested sign permutations across E/I; unknown edges restored first",
                assignment="same nested presynaptic-node STP assignment across all families and conditions per realization",
                dynamic_renormalization=False, leak=1., bias=0., device=args.backend,
                gpu_batch_size=args.gpu_batch_size if args.backend == "cuda" else None,
                precision="float32 weights/neural states; float64 STP integration/readout fits",
                readout_device="cpu")
    spec = json.loads(json.dumps(spec))
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec, identity, task, repeats, sources


def expected_keys(spec, repeat=None, family=None, ei=None, norm=None):
    for r, f, e, n, scheme, scale in itertools.product(
            range(spec["n_repeats"]) if repeat is None else [repeat],
            spec["families"] if family is None else [family],
            spec["ei_percent"] if ei is None else [ei],
            spec["normalizations"] if norm is None else [norm],
            spec["input_schemes"], spec["input_scales"]):
        for label, preset, mix in spec["stp_conditions"]:
            yield (r, f, e, n, scheme, scale, label, "none")
            if spec["gain_control"] and preset != "none":
                yield (r, f, e, n, scheme, scale, label, "training_mean_frozen")


def validate_frame(frame, spec, identity, expected):
    if frame.empty or set(frame.experiment_id) != {identity} or frame.duplicated(KEYS).any():
        raise ValueError("Empty, duplicate, or incompatible result shards.")
    if set(frame[KEYS].itertuples(index=False, name=None)) != set(expected):
        raise ValueError("Missing or unexpected result configurations.")
    delays = [f"MC_delay_{d:02d}" for d in range(1, spec["task"]["mc_max_delay"] + 1)]
    if not np.isfinite(frame[SCORES + delays].to_numpy(float)).all():
        raise ValueError("Nonfinite task scores or delay curves.")


def score_states(states, data, task, task_name):
    x = torch.from_numpy(states)
    start, end = task.washout, task.washout + task.train
    if task_name == "MC":
        u = data["mc_input"]
        total, curve = compute_MC(x[start:end].double(), x[end:].double(),
                                  u[start:end].double(), u[end:].double(),
                                  task.mc_max_delay, task.mc_ridge_alpha, torch.device("cpu"))
        return {"MC": float(total), **{f"MC_delay_{d+1:02d}": float(v) for d, v in enumerate(curve)}}
    val_end = end + task.validation
    y = data["narma_target"]
    return fit_narma_readout(x[start:end], y[start:end], x[end:val_end], y[end:val_end],
                            x[val_end:], y[val_end:], task.narma_ridge_alphas)


def evaluate(weights, input_weights, data, task, vectors, with_gain_control):
    records = {"none": {}}
    if with_gain_control and vectors is not None:
        records["training_mean_frozen"] = {}
    for task_name, signal_key in (("MC", "mc_input"), ("NARMA10", "narma_input")):
        signal = data[signal_key].numpy()
        states, details, mean_gain = run_states(weights, input_weights, signal, vectors,
                                                task.washout, task.washout + task.train)
        records["none"].update(score_states(states, data, task, task_name))
        records["none"].update({task_name + "_" + k: v for k, v in details.items()})
        if "training_mean_frozen" in records:
            # Coefficients only use training samples, never validation/test.
            frozen, frozen_details, _ = run_states(weights, input_weights, signal, None,
                                                    task.washout, task.washout + task.train,
                                                    fixed_gain=mean_gain)
            records["training_mean_frozen"].update(score_states(frozen, data, task, task_name))
            records["training_mean_frozen"].update({task_name + "_" + k: v for k, v in frozen_details.items()})
    return records


def evaluate_cuda(weights, conditions, data, task, with_gain_control, device, batch_size):
    """Batch independent input/STP combinations; preserve existing CPU readout fits."""
    from network_stats.stp_weight_controls_cuda import run_states_batch

    output = []
    for start_batch in range(0, len(conditions), batch_size):
        batch = conditions[start_batch:start_batch + batch_size]
        records = [{"none": {}} for _ in batch]
        for i, (_, vectors) in enumerate(batch):
            if with_gain_control and vectors is not None:
                records[i]["training_mean_frozen"] = {}
        b = np.stack([win[:, 0] for win, _ in batch])
        vectors = [v for _, v in batch]
        dynamic_indices = [i for i, v in enumerate(vectors) if v is not None]
        for task_name, signal_key in (("MC", "mc_input"), ("NARMA10", "narma_input")):
            states, details, mean_gain = run_states_batch(weights, b, data[signal_key].numpy(), vectors,
                                                          task.washout, task.washout + task.train, device=device)
            for i in range(len(batch)):
                records[i]["none"].update(score_states(states[i], data, task, task_name))
                records[i]["none"].update({task_name + "_" + k: v for k, v in details[i].items()})
            del states
            if with_gain_control and dynamic_indices:
                frozen, frozen_details, _ = run_states_batch(
                    weights, b[dynamic_indices], data[signal_key].numpy(), [None] * len(dynamic_indices),
                    task.washout, task.washout + task.train, fixed_gains=mean_gain[dynamic_indices], device=device)
                for offset, i in enumerate(dynamic_indices):
                    records[i]["training_mean_frozen"].update(score_states(frozen[offset], data, task, task_name))
                    records[i]["training_mean_frozen"].update({task_name + "_" + k: v for k, v in frozen_details[offset].items()})
                del frozen
        output.extend(records)
    return output


def sensory_indices(args, n):
    if "restricted_matched_norm" not in args.input_schemes:
        return np.empty(0, dtype=int)
    labels = pd.read_csv(args.sensory_labels)
    nodes = args.node_names.read_text().splitlines()
    indices = labels.node_index.to_numpy(dtype=int)
    if len(nodes) != n or len(set(indices)) != len(indices) or not len(indices) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("Sensory labels do not match the node list/matrix.")
    if list(labels.neuron) != [nodes[i] for i in indices]:
        raise ValueError("Sensory neuron identities disagree with matrix node order.")
    return indices


def run_repeat(job):
    repeat, args_dict, spec, identity = job
    args = argparse.Namespace(**args_dict)
    out = args.out_dir / "repeats" / f"repeat_{repeat:04d}"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if spec["device"] == "cuda":
        from network_stats.stp_weight_controls_cuda import check_cuda
        device_info = check_cuda(args.cuda_device)
        if device_info["simulated"] and os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("CUDA simulation is forbidden in a Slurm production run.")
        # A resumed shard can run on another GPU model; retain each attempt.
        attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{time.time_ns()}"
        publish_json(out / "gpu_devices" / f"{attempt}.json", device_info)
    started = time.monotonic()
    seed = spec["seed"] + repeat * spec["repeat_stride"]
    task = TemporalTaskSettings(**spec["task"])
    completed, known_fraction = load_completed_connectome(args.ce_adj, args.ce_unknown_sign_weights, seed)
    n = len(completed)
    sensory = sensory_indices(args, n)
    er_mask = np.random.default_rng(20261118 + repeat * 10000 + 1000).choice(n, len(sensory), replace=False)
    assignment = np.random.default_rng(seed + 44_000_003).permutation(n)
    er_unit, win, _ = build_condition(completed, "matched_er", 0., "spectral_radius", seed, torch.device("cpu"))
    controls = unsigned_controls(completed, er_unit.numpy() != 0, seed)
    data = make_task_data(task, seed)
    cache = {"STP_assignment_order": assignment, "sensory_indices": sensory,
             "ER_input_indices": er_mask, "unit_input_weights": win.numpy(),
             **{k: v.numpy() for k, v in data.items()},
             **{k + "_unsigned": v for k, v in controls.items()}}
    for percent in spec["ei_percent"]:
        for source, family in (("ce", "cel_matched"), ("er", "matched_er")):
            signed, same_win, _ = build_condition(completed, family, percent / 100, "spectral_radius", seed, torch.device("cpu"))
            np.testing.assert_array_equal(win, same_win)
            cache[f"{source}_signs_ei{percent:g}"] = np.sign(signed.numpy()).astype(np.int8)
        cache[f"er_signed_empirical_raw_ei{percent:g}"] = match_signed_weights(
            controls["ce_empirical"] * cache[f"ce_signs_ei{percent:g}"],
            cache[f"er_signs_ei{percent:g}"], seed)
    # Cache contains actual task signals, unsigned matrices, signs, input and STP
    # assignments: all evaluated W/Win can be reconstructed without resampling.
    publish_npz(out / "inputs_and_networks.npz", cache)
    publish_json(out / "cache_manifest.json", dict(experiment_id=identity, seed=seed,
                                                   known_inhibitory_fraction=known_fraction,
                                                   cache_sha256=sha(out / "inputs_and_networks.npz")))
    evaluated = reused = 0
    for percent in spec["ei_percent"]:
        ce_radius = spectral_radius(controls["ce_empirical"] * cache[f"ce_signs_ei{percent:g}"])
        for family, norm in itertools.product(spec["families"], spec["normalizations"]):
            shard = out / f"{family}__ei{percent:g}__{norm}.csv"
            expected = list(expected_keys(spec, repeat, family, percent, norm))
            if shard.exists():
                if not args.resume:
                    raise FileExistsError(f"Use --resume for existing output: {shard}")
                validate_frame(pd.read_csv(shard), spec, identity, expected)
                reused += len(expected)
                continue
            signs = cache[f"{'ce' if family.startswith('ce_') else 'er'}_signs_ei{percent:g}"]
            raw = cache[f"er_signed_empirical_raw_ei{percent:g}"] if family == "er_signed_empirical" else controls[family] * signs
            weights, info = scale_matrix(raw, norm, spec["initial_radius"], ce_radius)
            nz = raw != 0
            rows = []
            prepared = []
            metadata = []
            for scheme, scale in itertools.product(spec["input_schemes"], spec["input_scales"]):
                dense = (win * scale).numpy()
                indices = np.arange(n) if scheme == "full" else sensory if family.startswith("ce_") else er_mask
                input_weights = dense if scheme == "full" else place_input_weights(dense, sensory, indices, "matched_total_norm")
                for label, preset, mix in spec["stp_conditions"]:
                    vectors = None if preset == "none" else stp_vectors(n, preset, mix, assignment)
                    prepared.append((input_weights, vectors))
                    metadata.append(dict(experiment_id=identity, repeat_id=repeat, seed=seed,
                                         family=family, ei_percent=percent, normalization=norm,
                                         input_scheme=scheme, input_scale=scale, stp_condition=label,
                                         parameter_set=preset, facilitating_fraction=mix,
                                         facilitating_count=0 if vectors is None else int(vectors[3].sum()),
                                         n_nodes=n, n_edges=int(nz.sum()),
                                         realized_inhibitory_fraction=float(np.mean(raw[nz] < 0)),
                                         inhibitory_abs_weight_fraction=float(abs(raw[raw < 0]).sum() / abs(raw).sum()),
                                         initial_radius_target=spec["initial_radius"],
                                         post_abs_weight_mean=float(abs(weights[nz]).mean()),
                                         input_l2_norm=float(np.linalg.norm(input_weights.astype(float))),
                                         n_driven=len(indices),
                                         network_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                                         **info))
            if spec["device"] == "cuda":
                batches = evaluate_cuda(weights, prepared, data, task, spec["gain_control"], args.cuda_device, args.gpu_batch_size)
            else:
                batches = [evaluate(weights, inputs, data, task, vectors, spec["gain_control"])
                           for inputs, vectors in prepared]
            for base, records in zip(metadata, batches):
                for gain_control, scores in records.items():
                    rows.append(dict(**base, gain_control=gain_control, **scores))
            frame = pd.DataFrame(rows)
            validate_frame(frame, spec, identity, expected)
            write_shard(shard, rows)
            evaluated += len(rows)
            print(f"[repeat {repeat:03d}] {family} EI={percent:g}% {norm}: {len(rows)} pairs", flush=True)
    return repeat, evaluated, reused, time.monotonic() - started


def merge_run(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    spec, identity = manifest["spec"], manifest["experiment_id"]
    files = sorted(out.glob("repeats/repeat_*/*__ei*__*.csv"))
    if not files:
        raise ValueError("No completed shards.")
    frames = [pd.read_csv(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    validate_frame(frame, spec, identity, expected_keys(spec))
    for repeat in range(spec["n_repeats"]):
        base = out / "repeats" / f"repeat_{repeat:04d}"
        info = json.loads((base / "cache_manifest.json").read_text())
        if info["experiment_id"] != identity or sha(base / "inputs_and_networks.npz") != info["cache_sha256"]:
            raise ValueError(f"Invalid realization cache: {base}")
    target = out / "stp_weight_controls.ALL.csv"
    frame = frame.sort_values(KEYS)
    if target.exists():
        existing = pd.read_csv(target)
        pd.testing.assert_frame_equal(existing, frame.reset_index(drop=True), check_dtype=False)
    else:
        write_shard(target, frame.to_dict("records"))
    publish_json(out / "merge_audit.json", dict(experiment_id=identity, rows=len(frame), shards=len(files),
                                              paired_realizations=spec["n_repeats"], csv_sha256=sha(target)))
    print(f"Validated and merged {len(frame)} MC/NARMA pairs: {target}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.check_cuda:
        from network_stats.stp_weight_controls_cuda import check_cuda
        print(json.dumps(check_cuda(args.cuda_device), indent=2))
        return
    if args.merge:
        merge_run(args.out_dir)
        return
    spec, identity, task, repeats, sources = make_spec(args)
    conditions = len(spec["stp_conditions"]) + (len(spec["stp_conditions"]) - 1 if args.gain_control else 0)
    per_repeat = len(spec["families"]) * len(spec["ei_percent"]) * len(spec["normalizations"]) * len(spec["input_scales"]) * len(spec["input_schemes"]) * conditions
    print(json.dumps(dict(experiment_id=identity, assigned_repeats=repeats,
                          MC_NARMA_pairs_per_repeat=per_repeat,
                          MC_NARMA_pairs_full_run=per_repeat * args.n_repeats,
                          STP_conditions=spec["stp_conditions"], optional_gain_control=args.gain_control,
                          radius_grid=False, device=args.backend), indent=2), flush=True)
    # Validate input matrices, including zero-diagonal and disjoint known/unknown.
    completed, _ = load_completed_connectome(args.ce_adj, args.ce_unknown_sign_weights, args.seed)
    sensory_indices(args, len(completed))
    if args.dry_run:
        return
    if version("reservoirpy") != "0.4.2":
        raise ValueError("Requires ReservoirPy 0.4.2, as in the existing NARMA10 protocol.")
    if args.backend == "cuda":
        from network_stats.stp_weight_controls_cuda import check_cuda
        device_info = check_cuda(args.cuda_device)
        if device_info["simulated"] and os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("CUDA simulation is forbidden in a Slurm production run.")
        print(f"GPU startup check passed: {device_info}", flush=True)
    packages = {name: version(name) for name in ("numpy", "scipy", "numba", "torch", "reservoirpy", "scikit-learn", "pandas")}
    if args.backend == "cuda":
        try:
            packages["numba-cuda"] = version("numba-cuda")
        except PackageNotFoundError:
            packages["numba-cuda"] = "bundled numba.cuda"
    manifest = dict(experiment_id=identity, spec=spec, packages=packages)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "manifest.json").exists() and not args.resume:
        raise FileExistsError("Output already exists: use --resume or a new output directory.")
    publish_json(args.out_dir / "manifest.json", manifest)
    for path in CODE:
        publish_bytes(args.out_dir / "source_snapshot" / path, (ROOT / path).read_bytes())
    for role, path in sources.items():
        publish_bytes(args.out_dir / "source_snapshot" / "data" / (role + path.suffix), path.read_bytes())
    jobs = [(r, vars(args), spec, identity) for r in repeats]
    if args.workers == 1:
        outcomes = map(run_repeat, jobs)
        for result in outcomes:
            print(f"[complete] repeat={result[0]} new={result[1]} reused={result[2]} seconds={result[3]:.1f}", flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for future in as_completed([pool.submit(run_repeat, job) for job in jobs]):
                result = future.result()
                print(f"[complete] repeat={result[0]} new={result[1]} reused={result[2]} seconds={result[3]:.1f}", flush=True)
    if spec["source_hashes"] != {k: sha(v) for k, v in sources.items()} or spec["code_hashes"] != {k: sha(ROOT / k) for k in CODE}:
        raise ValueError("Input/code changed during execution.")


if __name__ == "__main__":
    main()
