#!/usr/bin/env python3
"""MC-only extension of the frozen architecture and ER-distribution experiments.

Reuse archived matrices, projections, seeds and exact scaling factors. Extend
the original input prefix, retain train/validation splits, score delays 1..200
on 5,000 held-out samples. NARMA and IPC are neither simulated nor refitted.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
from functools import lru_cache
import gzip
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
import architecture_mc_narma as base
from network_stats.architecture_sweep import scaling_factor as architecture_factor
from network_stats.er_weight_distributions import scaling_factor as er_factor
from network_stats.audited_temporal_tasks import (
    TaskSettings, delay_targets, simulate_batch, squared_correlations, validated_ridge)

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'arch-er-mc200-v1'
DEFAULT_PARENTS = {
    'architecture': 'results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871',
    'ws': 'results/ws_binary_halfnormal/ws_binary_halfnormal_272279',
    'er': 'results/er_weight_distribution/er_weights_274402',
}
BASELINES = {'empirical': 'er_signed_empirical', 'binary': 'er_binary', 'halfnormal': 'er_halfnormal'}


def canonical_spec(spec):
    suite, family, norm = spec
    if suite == 'er' and norm == 'frobenius_match' and family in ('empirical', 'gamma', 'lognormal'):
        return canonical_spec(('er', family, 'shared_ce_scale'))
    if suite == 'er' and family in BASELINES and norm in ('own_radius', 'shared_ce_scale'):
        return ('architecture', BASELINES[family], norm)
    return tuple(spec)


def specifications(parents):
    logical = [(s, f, n) for s, p in parents.items()
               for f in p['config']['families'] for n in p['config']['normalizations']]
    unique = list(dict.fromkeys(canonical_spec(s) for s in logical))
    if any(s not in logical for s in unique):
        raise ValueError('An alias has no canonical source')
    return logical, unique


def source_files():
    files = [ROOT/'mc200_recompute.py', ROOT/'architecture_mc_narma.py', ROOT/'MC200_RECOMPUTE.md']
    files += list(ROOT.glob('r_mc200*.sbatch'))
    files += [ROOT/'tests/test_mc200_recompute.py']
    for directory in ('network_stats', 'util', 'third_party'):
        files += list((ROOT/directory).rglob('*.py'))
    return sorted(set(files))


def prepare(a):
    parents = {}
    for suite, default in DEFAULT_PARENTS.items():
        path = Path(getattr(a, suite+'_parent', None) or ROOT/default).resolve()
        m = json.loads((path/'manifest.json').read_text())
        parents[suite] = dict(path=str(path), manifest_sha256=base.file_digest(path/'manifest.json'),
                             experiment_id=m['experiment_id'], config=m['config'])
        if base.file_digest(ROOT/'network_stats/audited_temporal_tasks.py') != m['source_sha256']['network_stats/audited_temporal_tasks.py']:
            raise ValueError('Archived task implementation differs from current source')
    c = parents['architecture']['config']
    for p in parents.values():
        for key in ('fractions', 'rhos', 'leaks', 'inputs', 'repeats', 'seed', 'seed_stride', 'tasks', 'state_dtype', 'bias'):
            if p['config'][key] != c[key]:
                raise ValueError(f'Parents differ in {key}')
        for repeat in range(c['repeats']):
            for kind in ('networks', 'tasks'):
                path = Path(p['path'])/'cache'/f'{kind}_{repeat:04d}.npz'
                if not path.is_file() or not Path(str(path)+'.json').is_file():
                    raise ValueError(f'Missing archived cache: {path}')
    logical, unique = specifications(parents)
    settings = replace(TaskSettings(**c['tasks']), mc_max_delay=200, mc_diagnostic_delay=200, test=a.test)
    sources = {str(p.relative_to(ROOT)): base.file_digest(p) for p in source_files()}
    identity = dict(protocol=PROTOCOL, parents=parents, tasks=asdict(settings), source_sha256=sources,
                    logical_specs=logical, unique_specs=unique)
    manifest = dict(**identity, experiment_id=base.digest(base.canonical(identity)),
                    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    expected_rows=len(logical)*c['repeats']*len(c['fractions'])*len(base.grid({'config': c})),
                    unique_rows=len(unique)*c['repeats']*len(c['fractions'])*len(base.grid({'config': c})),
                    groups=len(unique)*c['repeats']*len(c['fractions']),
                    mc_definition='sum of held-out squared Pearson correlations, delays 1..200; no clipping or null subtraction',
                    output='compressed per-lag float64 curves; compact merged scalar CSV; no states',
                    narma_ipc_policy='retain original results in the local backup; no recomputation')
    out = a.run_dir.resolve()
    if out.exists():
        old = json.loads((out/'manifest.json').read_text())
        if old['experiment_id'] != manifest['experiment_id']:
            raise ValueError('Existing run has a different identity')
        print(f'Resuming prepared run {out}', flush=True)
        return
    (out/'snapshot').mkdir(parents=True)
    for p in source_files():
        dest = out/'snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for name in ('checkpoints', 'preflight', 'runtime'):
        (out/name).mkdir()
    base.write_json(out/'manifest.json', manifest)
    print(f'Prepared {out}: {manifest["expected_rows"]:,} logical rows; {manifest["unique_rows"]:,} unique evaluations; MC200 only', flush=True)


def load_run(out):
    m = json.loads((out/'manifest.json').read_text())
    identity = {k: m[k] for k in ('protocol', 'parents', 'tasks', 'source_sha256', 'logical_specs', 'unique_specs')}
    if base.digest(base.canonical(identity)) != m['experiment_id']:
        raise ValueError('Run manifest identity changed')
    for name, sha in m['source_sha256'].items():
        if base.file_digest(ROOT/name) != sha:
            raise ValueError(f'Frozen source changed: {name}')
    for parent in m['parents'].values():
        if base.file_digest(Path(parent['path'])/'manifest.json') != parent['manifest_sha256']:
            raise ValueError('Parent manifest changed')
    return m


@lru_cache(maxsize=6)
def cached_arrays(parent_path, experiment_id, repeat):
    return base.read_cache(Path(parent_path)/'cache'/f'networks_{repeat:04d}.npz', experiment_id)


def extended_input(parent, repeat, settings):
    c = parent['config']
    seed = c['seed']+repeat*c['seed_stride']
    signal = np.random.default_rng(seed+31_000_003).uniform(-1, 1, (settings.length, 1))
    old = base.read_cache(Path(parent['path'])/'cache'/f'tasks_{repeat:04d}.npz', parent['experiment_id'])['mc_input']
    np.testing.assert_array_equal(signal[:len(old)], old)
    return signal


def context(m, spec, repeat, fi):
    suite, family, norm = spec
    parent = m['parents'][suite]
    c = parent['config']
    cache = cached_arrays(parent['path'], parent['experiment_id'], repeat)
    raw, _, radius = base.raw_from_cache(cache, family, fi)
    path = Path(parent['path'])/'shards'/f'repeat_{repeat:04d}__{family}__ei_{fi:02d}__{norm}.csv'
    sha = base.file_digest(path)
    sidecar = json.loads(Path(str(path)+'.json').read_text())
    if sidecar['sha256'] != sha or sidecar['experiment_id'] != parent['experiment_id']:
        raise ValueError(f'Archived shard hash/identity differs: {path}')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    indexed = {tuple(float(r[k]) for k in ('rho_target', 'leak', 'input_scale')): r for r in rows}
    points = base.grid({'config': c})
    if len(rows) != len(points) or set(indexed) != set(points):
        raise ValueError('Archived hyperparameter grid missing or duplicated')
    ordered = [indexed[p] for p in points]
    raw_hash = base.digest(raw.tobytes())
    input_hash = base.digest(cache['input_vector'].tobytes())
    factors = np.array([float(r['scaling_factor']) for r in ordered])
    ce_radius = float(cache[f'ce_empirical__{fi}__radius'])
    if suite == 'er':
        raw_f = np.linalg.norm(raw)
        ref_f = np.linalg.norm(cache[f'er_signed_empirical__{fi}__values'])
        expected = [er_factor(norm, rho, radius, ce_radius, raw_f, ref_f) for rho, _, _ in points]
    else:
        fixed = float(cache[f'{family}__{len(c["fractions"])}__radius'])
        expected = [architecture_factor(norm, rho, radius, fixed, ce_radius) for rho, _, _ in points]
    np.testing.assert_allclose(factors, expected, rtol=1e-10, atol=1e-12)
    for row in rows:
        if row['raw_network_sha256'] != raw_hash or row['input_vector_sha256'] != input_hash or int(row['repeat_id']) != repeat or float(row['negative_fraction']) != c['fractions'][fi]:
            raise ValueError('Archived matrix/input/condition mismatch')
    old = np.array([[float(r[f'MC_delay_{i:02d}']) for i in range(1, 61)] for r in ordered])
    return dict(raw=raw, input=cache['input_vector'], factors=factors, points=points, old=old,
                provenance=dict(parent_experiment_id=parent['experiment_id'], source_shard=str(path),
                                source_shard_sha256=sha, raw_network_sha256=raw_hash,
                                input_vector_sha256=input_hash, raw_radius=radius))


def alias_provenance(m, spec, repeat, fi, original):
    provenance = []
    for logical in m['logical_specs']:
        if canonical_spec(logical) == tuple(spec) and tuple(logical) != tuple(spec):
            other = context(m, logical, repeat, fi)
            for key in ('raw', 'input', 'factors'):
                # Exact identity is required for sharing scores; near equality is insufficient.
                np.testing.assert_array_equal(original[key], other[key])
            provenance.append(dict(spec=logical, **other['provenance']))
    return provenance


def decode_group(m, index):
    nf = len(m['parents']['architecture']['config']['fractions'])
    repeat, remainder = divmod(index, len(m['unique_specs'])*nf)
    si, fi = divmod(remainder, nf)
    return repeat, tuple(m['unique_specs'][si]), fi


def checkpoint_path(out, repeat, spec, fi):
    return out/'checkpoints'/f'r{repeat:04d}__{"__".join(spec)}__ei{fi:02d}.npz'


def array_digest(arrays):
    return base.digest(b''.join(base.canonical(dict(name=k, shape=list(v.shape), dtype=v.dtype.str))+v.tobytes()
                                for k, v in sorted(arrays.items())))


def write_checkpoint(path, arrays, meta):
    meta = dict(meta, arrays_sha256=array_digest(arrays))
    data = io.BytesIO()
    np.savez_compressed(data, metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)), **arrays)
    base.atomic(path, data.getvalue())


def read_checkpoint(path, m, repeat, spec, fi):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        arrays = {k: z[k] for k in z.files if k != 'metadata'}
    if (meta['experiment_id'], meta['repeat'], tuple(meta['spec']), meta['fraction_index']) != (m['experiment_id'], repeat, tuple(spec), fi):
        raise ValueError('Checkpoint identity mismatch')
    if array_digest(arrays) != meta['arrays_sha256']:
        raise ValueError('Checkpoint array checksum mismatch')
    n = len(base.grid({'config': m['parents']['architecture']['config']}))
    h = m['tasks']['mc_max_delay']
    if arrays['curves'].shape != (n, h) or arrays['alpha_index'].shape != (n, h) or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Checkpoint shape/nonfinite data')
    if np.any(arrays['curves'] < 0) or np.any(arrays['curves'] > 1+1e-12):
        raise ValueError('Recall squared correlations outside valid range')
    return arrays, meta


def score_group(m, ctx, signal, device, indices=None):
    s = TaskSettings(**m['tasks'])
    ids = list(range(len(ctx['points']))) if indices is None else list(indices)
    points = [ctx['points'][i] for i in ids]
    factors = ctx['factors'][ids]
    states, gaps = simulate_batch(np.stack([ctx['raw']*v for v in factors]),
        np.stack([ctx['input']*inp for _, _, inp in points]), [leak for _, leak, _ in points],
        signal, s.washout, device, torch.float64)
    a = s.washout; b = a+s.train; c = b+s.validation
    y = delay_targets(signal, s.mc_max_delay)
    curves = []; alpha_ids = []; scales = []; losses = []; prefix_errors = []
    old_test = m['parents']['architecture']['config']['tasks']['test']
    for j, x in enumerate(states):
        pred, alpha, loss, scale = validated_ridge(x[a:b], y[a:b], x[b:c], y[b:c], x[c:], s.ridge_alphas)
        curves.append(squared_correlations(y[c:], pred))
        alpha_ids.append([list(s.ridge_alphas).index(v) for v in alpha])
        scales.append(scale); losses.append(float(np.mean(loss)))
        prefix = squared_correlations(y[c:c+old_test, :60], pred[:old_test, :60])
        prefix_errors.append(float(np.max(abs(prefix-ctx['old'][ids[j]]))))
    return dict(curves=np.asarray(curves), alpha_index=np.asarray(alpha_ids, dtype=np.uint8),
                scaling_factor=factors, washout_gap=np.asarray(gaps), train_state_rms=np.asarray(scales),
                validation_mse=np.asarray(losses), archived_prefix_max_error=np.asarray(prefix_errors))


def preflight(a):
    m = load_run(a.run_dir)
    runtime = base.configure_device(a.device)
    if not a.device.startswith('cuda'):
        raise ValueError('GPU preflight requires CUDA; no CPU fallback')
    settings = TaskSettings(**m['tasks'])
    # Every parent/repeat stream must retain its original prefix.
    for repeat in range(m['parents']['architecture']['config']['repeats']):
        streams = [extended_input(p, repeat, settings) for p in m['parents'].values()]
        for u in streams[1:]:
            np.testing.assert_array_equal(streams[0], u)
    signal = extended_input(m['parents']['architecture'], 0, settings)
    checks = []
    smoke = [('architecture', 'ce_empirical', 'own_radius'), ('ws', 'ws_binary', 'fixed_reference'),
             ('er', 'gamma', 'frobenius_match'), ('er', 'cap90', 'own_radius')]
    for spec in smoke:
        ctx = context(m, spec, 0, 2)
        cpu = score_group(m, ctx, signal, 'cpu', [8])  # rho=.95, leak=.6, input=.01: stable parity case.
        gpu = score_group(m, ctx, signal, a.device, [8])
        error = float(abs(cpu['curves']-gpu['curves']).max())
        if error > 1e-6:
            raise ValueError(f'CPU/GPU recall disagreement: {spec}: {error}')
        if max(cpu['archived_prefix_max_error'].max(), gpu['archived_prefix_max_error'].max()) > 1e-6:
            raise ValueError(f'Archived prefix disagreement in stable preflight: {spec}')
        checks.append(dict(spec=spec, max_cpu_gpu_error=error, archive_prefix_max_error=float(gpu['archived_prefix_max_error'].max())))
        print(f'Preflight {spec}: CPU/GPU error {error:.3g}', flush=True)
    for spec in m['unique_specs']:
        ctx = context(m, spec, 0, 2)
        alias_provenance(m, spec, 0, 2, ctx)
    base.write_json(a.run_dir/'preflight/PASSED.json', dict(experiment_id=m['experiment_id'], runtime=runtime,
                    checks=checks, all_streams_checked=True))
    print('MC200 GPU preflight PASSED', flush=True)


def run(a):
    m = load_run(a.run_dir)
    runtime = base.configure_device(a.device)
    if not a.device.startswith('cuda') or not 0 <= a.rank < a.split:
        raise ValueError('Production requires a GPU and valid array partition')
    gate = json.loads((a.run_dir/'preflight/PASSED.json').read_text())
    if gate['experiment_id'] != m['experiment_id'] or gate['runtime']['environment'] != runtime['environment']:
        raise ValueError('Missing/mismatched GPU preflight')
    # A30/L40S can be mixed; same frozen code/packages and float64 are required.
    record = a.run_dir/'runtime'/f'worker_{a.rank:03d}.json'
    if not record.exists():
        base.write_json(record, dict(runtime=runtime, rank=a.rank, split=a.split,
                                    experiment_id=m['experiment_id'], slurm_job=os.environ.get('SLURM_JOB_ID')))
    signal = None; last_repeat = None
    for index in range(a.rank, m['groups'], a.split):
        repeat, spec, fi = decode_group(m, index)
        path = checkpoint_path(a.run_dir, repeat, spec, fi)
        if path.exists():
            read_checkpoint(path, m, repeat, spec, fi)
            continue
        started = time.monotonic()
        if last_repeat != repeat:
            signal = extended_input(m['parents']['architecture'], repeat, TaskSettings(**m['tasks']))
            last_repeat = repeat
        ctx = context(m, spec, repeat, fi)
        aliases = alias_provenance(m, spec, repeat, fi, ctx)
        arrays = score_group(m, ctx, signal, a.device)
        meta = dict(experiment_id=m['experiment_id'], repeat=repeat, spec=spec, fraction_index=fi,
                    provenance=ctx['provenance'], aliases=aliases, seconds=time.monotonic()-started)
        write_checkpoint(path, arrays, meta)
        read_checkpoint(path, m, repeat, spec, fi)
        print(f'{path.name}: MC200 mean={arrays["curves"].sum(axis=1).mean():.5g}, {meta["seconds"]:.1f}s', flush=True)


def merge(a):
    m = load_run(a.run_dir); c = m['parents']['architecture']['config']
    target = a.run_dir/'mc200.ALL.csv.gz'
    count = 0; max_prefix = 0.
    with tempfile.NamedTemporaryFile(dir=a.run_dir, suffix='.merge.tmp', delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with gzip.open(temp, 'wt', newline='', compresslevel=6) as f:
            writer = None
            for repeat in range(c['repeats']):
                for spec in m['logical_specs']:
                    canonical = canonical_spec(spec)
                    for fi, fraction in enumerate(c['fractions']):
                        path = checkpoint_path(a.run_dir, repeat, canonical, fi)
                        arrays, meta = read_checkpoint(path, m, repeat, canonical, fi)
                        for j, (rho, leak, inp) in enumerate(base.grid({'config': c})):
                            curve = arrays['curves'][j]
                            row = dict(experiment_id=m['experiment_id'], repeat_id=repeat, suite=spec[0], family=spec[1],
                                negative_fraction=fraction, normalization=spec[2], rho_target=rho, leak=leak, input_scale=inp,
                                MC=float(curve.sum()), MC_30=float(curve[:30].sum()), MC_60=float(curve[:60].sum()),
                                MC_100=float(curve[:100].sum()), MC_150=float(curve[:150].sum()),
                                MC_200=float(curve.sum()), MC_last25=float(curve[-25:].sum()),
                                scaling_factor=float(arrays['scaling_factor'][j]),
                                archived_prefix_max_error=float(arrays['archived_prefix_max_error'][j]),
                                washout_initialization_rms_gap=float(arrays['washout_gap'][j]),
                                curve_checkpoint=str(path.relative_to(a.run_dir)), curve_row=j)
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row); count += 1
                            max_prefix = max(max_prefix, row['archived_prefix_max_error'])
        if count != m['expected_rows']:
            raise ValueError('Incomplete merge')
        if target.exists():
            raise ValueError('Merged result already exists; refusing overwrite')
        os.link(temp, target)
        base.write_json(a.run_dir/'COMPLETE.json', dict(experiment_id=m['experiment_id'], rows=count,
                        sha256=base.file_digest(target), archived_prefix_max_error=max_prefix,
                        horizons=[30, 60, 100, 150, 200], narma_rerun=False, ipc_rerun=False))
    finally:
        temp.unlink(missing_ok=True)
    print(f'Merged {count:,} rows: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'preflight', 'run', 'merge'):
        q = sub.add_parser(name); q.add_argument('--run-dir', type=Path, required=True)
        if name == 'prepare':
            for suite in DEFAULT_PARENTS:
                q.add_argument('--'+suite+'-parent', type=Path)
            q.add_argument('--test', type=int, default=5000)
        if name in ('preflight', 'run'):
            q.add_argument('--device', default='cuda:0')
        if name == 'run':
            q.add_argument('--rank', type=int, required=True); q.add_argument('--split', type=int, required=True)
    a = p.parse_args(); globals()[a.command](a)


if __name__ == '__main__':
    main()
