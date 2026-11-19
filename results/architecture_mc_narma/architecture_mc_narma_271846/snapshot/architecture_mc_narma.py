#!/usr/bin/env python3
"""Frozen, audited ten-architecture MC/conventional-NARMA10 experiment."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import io
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import numpy as np
import torch
from network_stats.architecture_sweep import (FAMILIES, NORMALIZATIONS, DEFINITIONS,
    make_assets, raw_networks, rng_for, scaling_factor, spectrum, validate_connectomes)
from network_stats.audited_temporal_tasks import TaskSettings, make_task_data, score_states, simulate_batch

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'architecture-mc-narma-v2-conventional'
SOURCES = ('ARCHITECTURE_SWEEP.md', 'architecture_mc_narma.py', 'network_stats/architecture_sweep.py',
    'network_stats/audited_temporal_tasks.py', 'util/util.py',
    'third_party/echotorch/NARMADataset.py', 'third_party/echotorch/COPYING',
    'third_party/echotorch/README.md', 'tests/test_architecture_pipeline.py',
    'tools/analyze_architecture_mc_narma.py', 'tools/submit_architecture_mc_narma.sh',
    'r_architecture_mc_narma.sbatch', 'r_architecture_mc_narma_preflight.sbatch',
    'r_architecture_mc_narma_analyze.sbatch')
DATA = ('Connectome/ce_adj.npy', 'Connectome/ce_unknown_sign_weights.npy')
KEY = ('repeat_id', 'family', 'negative_fraction', 'normalization', 'rho_target', 'leak', 'input_scale')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic(path, data):
    """Publish complete bytes without replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b'\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse an empty CSV')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, output.getvalue().encode())


def write_cache(path, arrays, experiment_id):
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    data = out.getvalue()
    atomic(path, data)
    write_json(str(path)+'.json', dict(experiment_id=experiment_id, sha256=digest(data)))


def read_cache(path, experiment_id):
    data = Path(path).read_bytes()
    meta = json.loads(Path(str(path)+'.json').read_text())
    if meta != dict(experiment_id=experiment_id, sha256=digest(data)):
        raise ValueError(f'Cache identity/hash mismatch: {path}')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def environment():
    return dict(python=platform.python_version(), packages={p: version(p) for p in
        ('numpy', 'scipy', 'scikit-learn', 'torch', 'networkx', 'pandas', 'matplotlib')})


def task_settings(manifest):
    return TaskSettings(**manifest['config']['tasks'])


def grid(manifest):
    c = manifest['config']
    return list(itertools.product(c['rhos'], c['leaks'], c['inputs']))


def repeat_seed(manifest, repeat):
    c = manifest['config']
    return c['seed']+repeat*c['seed_stride']


def prepare(args):
    c = dict(families=args.families, normalizations=args.normalizations, fractions=args.fractions,
             rhos=args.rhos, leaks=args.leaks, inputs=args.inputs, repeats=args.repeats,
             seed=args.seed, seed_stride=100000, swaps_per_edge=args.swaps_per_edge,
             ws_probability=.1, bias=0., state_dtype=args.state_dtype,
             input_projection='all nodes; shared U[-1,1] vector per repeat',
             narma_stream_policy='Reject only nonfinite generated targets before any network evaluation; seed=repeat_seed+37000003+1009*attempt, attempt=0..99; log every rejection; share accepted stream across all conditions',
             tasks=asdict(TaskSettings(washout=args.washout, train=args.train,
                 validation=args.validation, test=args.test, narma_burnin=args.narma_burnin,
                 mc_max_delay=args.mc_max_delay, mc_diagnostic_delay=args.mc_diagnostic_delay)))
    for name in ('families', 'normalizations', 'fractions', 'rhos', 'leaks', 'inputs'):
        if not c[name] or len(c[name]) != len(set(c[name])):
            raise ValueError(f'Empty/duplicate {name}')
    if args.repeats < 1 or args.seed < 0 or args.swaps_per_edge < 1:
        raise ValueError('Invalid repeat/seed/swap settings')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in args.fractions):
        raise ValueError('Fractions must be finite and in [0,1]')
    if any(not np.isfinite(v) or v <= 0 for v in args.rhos+args.inputs) or any(not 0 < v <= 1 for v in args.leaks):
        raise ValueError('Invalid grid coordinates')
    hashes = {p: digest((ROOT/p).read_bytes()) for p in SOURCES+DATA}
    known, unknown = validate_connectomes(*(np.load(ROOT/p) for p in DATA))
    identity = dict(protocol=PROTOCOL, config=c, source_sha256=hashes)
    experiment_id = digest(canonical(identity))
    total = args.repeats*len(args.families)*len(args.normalizations)*len(args.fractions)*len(list(itertools.product(args.rhos,args.leaks,args.inputs)))
    out = args.run_dir.resolve()
    if (out/'manifest.json').exists():
        existing = load_run(out)
        if existing['experiment_id'] != experiment_id:
            raise ValueError('Existing run differs in settings/source. Use a new RUN_ID; no overwrite.')
        print(f'Unchanged frozen run: {out}', flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refuse to prepare a nonempty directory without a manifest')
    for relative in SOURCES+DATA:
        atomic(out/'snapshot'/relative, (ROOT/relative).read_bytes())
    manifest = {**identity, 'experiment_id': experiment_id, 'prepared_environment': environment(),
        'model_definitions': DEFINITIONS, 'expected_evaluations': total,
        'data': dict(nodes=len(known), known_edges=int(np.count_nonzero(known)),
                     unknown_edges=int(np.count_nonzero(unknown)),
                     reference_fraction=float(np.count_nonzero(known<0)/np.count_nonzero(known))),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(out/'manifest.json', manifest)
    print(f'Prepared {out}: {total:,} paired MC/NARMA10 evaluations; {2*total:,} trajectories.', flush=True)


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/'manifest.json').read_text())
    identity = {k: manifest[k] for k in ('protocol','config','source_sha256')}
    if manifest['protocol'] != PROTOCOL or digest(canonical(identity)) != manifest['experiment_id']:
        raise ValueError('Invalid manifest protocol/identity')
    for p, expected in manifest['source_sha256'].items():
        if digest((run_dir/'snapshot'/p).read_bytes()) != expected:
            raise ValueError(f'Frozen source/data changed: {p}')
    # The entry point must use the same implementation, even for a manual launch.
    for p in SOURCES:
        if digest((ROOT/p).read_bytes()) != manifest['source_sha256'][p]:
            raise ValueError(f'Executing different code than snapshot: {p}. Run snapshot/architecture_mc_narma.py')
    return manifest


def tasks_cached(run, m, repeat):
    path = Path(run)/'cache'/f'tasks_{repeat:04d}.npz'
    if path.exists():
        data = read_cache(path, m['experiment_id'])
    else:
        data = make_task_data(task_settings(m), repeat_seed(m, repeat))
        write_cache(path, data, m['experiment_id'])
        if len(data['narma_rejected_seeds']):
            print(f'Repeat {repeat}: rejected nonfinite NARMA seeds {data["narma_rejected_seeds"].tolist()}; accepted {int(data["narma_generator_seed"])}',flush=True)
    s = task_settings(m)
    if any(data[k].shape != (s.length,1) or not np.isfinite(data[k]).all() for k in ('mc_input','narma_input','narma_target')) or float(data['generator_residual']) > 1e-12:
        raise ValueError('Invalid task cache')
    return data


def networks_cached(run, m, repeat):
    path = Path(run)/'cache'/f'networks_{repeat:04d}.npz'
    if path.exists():
        return read_cache(path, m['experiment_id'])
    seed = repeat_seed(m, repeat)
    known, unknown = (np.load(Path(run)/'snapshot'/p) for p in DATA)
    a = make_assets(known, unknown, seed, m['config']['swaps_per_edge'])
    arrays = {'reference_fraction': a['reference_fraction'], 'n_nodes': a['n_nodes'],
              'directed_swaps': a['directed_swaps'], 'rewired_edge_retention': a['rewired_edge_retention'],
              'raw_ce_mean_magnitude': a['raw_ce_mean_magnitude'],
              'input_vector': rng_for(seed,'shared all-node input projection').uniform(-1,1,len(known))}
    fractions = list(m['config']['fractions'])+[float(a['reference_fraction'])]
    for index, fraction in enumerate(fractions):
        nets = raw_networks(a, fraction, seed)
        for family in FAMILIES:  # Generate all controls so subset selection never changes draws.
            w = nets[family]
            e = np.argwhere(w != 0)
            eig, rho = spectrum(w)
            prefix = f'{family}__{index}'
            arrays[prefix+'__edges'] = e.astype(np.int32)
            arrays[prefix+'__values'] = w[e[:,0],e[:,1]]
            arrays[prefix+'__eigenvalues'] = eig
            arrays[prefix+'__radius'] = np.array(rho)
    write_cache(path, arrays, m['experiment_id'])
    return arrays


def raw_from_cache(cache, family, index):
    prefix = f'{family}__{index}'
    edges, values = cache[prefix+'__edges'], cache[prefix+'__values']
    n = int(cache['n_nodes'])
    w = np.zeros((n,n))
    w[edges[:,0],edges[:,1]] = values
    return w, cache[prefix+'__eigenvalues'], float(cache[prefix+'__radius'])


def configure_device(device):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    return dict(device=device, cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None,
                environment=environment(), tf32=False)


def condition_rows(m, cache, data, repeat, family, fi, normalization, points, device, batch_size):
    s = task_settings(m)
    w, eigenvalues, radius = raw_from_cache(cache, family, fi)
    ref = float(cache[f'{family}__{len(m["config"]["fractions"])}__radius'])
    ce = float(cache[f'ce_empirical__{fi}__radius'])
    n, edge_count = len(w), np.count_nonzero(w)
    state_dtype = m['config'].get('state_dtype','float64')
    raw_hash = digest(w.tobytes())
    rows = []
    for begin in range(0,len(points),batch_size):
        batch = points[begin:begin+batch_size]
        factors = [scaling_factor(normalization,rho,radius,ref,ce) for rho,leak,inp in batch]
        ws = np.stack([w*factor for factor in factors])
        win = np.stack([cache['input_vector']*inp for rho,leak,inp in batch])
        leaks = [leak for rho,leak,inp in batch]
        scores = [dict() for _ in batch]
        for task, input_key in [('MC','mc_input'),('NARMA10','narma_input')]:
            states, gaps = simulate_batch(ws,win,leaks,data[input_key],s.washout,device,getattr(torch,state_dtype))
            for j in range(len(batch)):
                scores[j].update(score_states(states[j],data,s,task))
                scores[j][f'{task}_washout_initialization_rms_gap'] = float(gaps[j])
            del states
        for j, ((rho,leak,inp), factor) in enumerate(zip(batch,factors)):
            row = dict(experiment_id=m['experiment_id'], protocol=PROTOCOL, repeat_id=repeat,
                seed=repeat_seed(m,repeat), family=family, negative_fraction=m['config']['fractions'][fi],
                normalization=normalization, rho_target=rho, leak=leak, input_scale=inp, neuron_bias=0.,
                nodes=n, edges=int(edge_count), realized_negative_fraction=float(np.count_nonzero(w<0)/edge_count),
                inhibitory_magnitude_fraction=float(abs(w[w<0]).sum()/abs(w).sum()),
                raw_radius=radius, fixed_reference_radius=ref, shared_ce_radius=ce,
                reference_fraction=float(cache['reference_fraction']), scaling_factor=factor,
                post_scaling_radius=radius*factor,
                leaky_linearized_radius=float(np.max(abs((1-leak)+leak*factor*eigenvalues))),
                mean_nonzero_raw_magnitude=float(abs(w[w!=0]).mean()),
                raw_network_sha256=raw_hash, input_vector_sha256=digest(cache['input_vector'].tobytes()),
                state_dtype=state_dtype, readout_dtype='float64', backend=device,
                narma_generator_seed=int(data['narma_generator_seed']),
                narma_rejected_streams=len(data['narma_rejected_seeds']),
                directed_swaps=int(cache['directed_swaps']), rewired_edge_retention=float(cache['rewired_edge_retention']))
            row.update(scores[j])
            rows.append(row)
    return rows


def shard_path(run, repeat, family, fi, norm):
    return Path(run)/'shards'/f'repeat_{repeat:04d}__{family}__ei_{fi:02d}__{norm}.csv'


def validate_rows(rows, m, repeat, family, fi, norm):
    expected = set(grid(m))
    if len(rows) != len(expected):
        raise ValueError('Incomplete/duplicated shard')
    seen = set()
    settings = task_settings(m)
    for r in rows:
        point = tuple(float(r[k]) for k in ('rho_target','leak','input_scale'))
        seen.add(point)
        if r['experiment_id'] != m['experiment_id'] or int(r['repeat_id']) != repeat or r['family'] != family or r['normalization'] != norm or float(r['negative_fraction']) != m['config']['fractions'][fi] or float(r['neuron_bias']) != 0:
            raise ValueError('Shard identity mismatch')
        for k,v in r.items():
            if k not in ('experiment_id','protocol','family','normalization','raw_network_sha256','input_vector_sha256','state_dtype','readout_dtype','backend','MC_ridge_alphas'):
                if not np.isfinite(float(v)):
                    raise ValueError(f'Nonfinite {k}')
        curve = np.array([float(r[f'MC_delay_{i:02d}']) for i in range(1,settings.mc_diagnostic_delay+1)])
        if np.any(curve<0) or np.any(curve>1+1e-12) or not np.isclose(float(r['MC']),curve[:settings.mc_max_delay].sum(),rtol=1e-10,atol=1e-10) or not np.isclose(float(r['MC_extended']),curve.sum(),rtol=1e-10,atol=1e-10):
            raise ValueError('Invalid memory capacity/curve')
        if float(r['NARMA10_test_nmse']) < 0 or float(r['NARMA10_val_nmse']) < 0 or not np.isclose(float(r['NARMA10_test_nrmse'])**2,float(r['NARMA10_test_nmse'])):
            raise ValueError('Invalid NARMA scores')
        fac = scaling_factor(norm,point[0],float(r['raw_radius']),float(r['fixed_reference_radius']),float(r['shared_ce_radius']))
        if not np.isclose(fac,float(r['scaling_factor']),rtol=1e-12) or not np.isclose(float(r['post_scaling_radius']),fac*float(r['raw_radius']),rtol=1e-12):
            raise ValueError('Invalid spectral normalization')
    if seen != expected:
        raise ValueError('Missing/unexpected grid settings')


def read_shard(path,m,repeat,family,fi,norm):
    data = Path(path).read_bytes()
    meta = json.loads(Path(str(path)+'.json').read_text())
    if meta != dict(experiment_id=m['experiment_id'], sha256=digest(data)):
        raise ValueError(f'Shard changed: {path}')
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    validate_rows(rows,m,repeat,family,fi,norm)
    return rows


def save_shard(path,rows,m,repeat,family,fi,norm):
    validate_rows(rows,m,repeat,family,fi,norm)
    write_csv(path,rows)
    write_json(str(path)+'.json',dict(experiment_id=m['experiment_id'],sha256=digest(Path(path).read_bytes())))


def run_sweep(args):
    m = load_run(args.run_dir)
    c = m['config']
    if args.split < 1 or not 0 <= args.rank < args.split or args.batch_size < 1:
        raise ValueError('Invalid array partition or batch size')
    runtime = configure_device(args.device)
    gate = json.loads((args.run_dir/'preflight/PASSED.json').read_text())
    if gate['experiment_id'] != m['experiment_id'] or gate['runtime']['device'] != args.device:
        raise ValueError('Preflight gate missing/incompatible')
    if digest((args.run_dir/'preflight/smoke.csv').read_bytes()) != gate['smoke_sha256']:
        raise ValueError('Preflight smoke output changed')
    if runtime['environment'] != gate['runtime']['environment']:
        raise ValueError('Environment differs from preflight')
    for repeat in range(args.rank,c['repeats'],args.split):
        data, cache = tasks_cached(args.run_dir,m,repeat), networks_cached(args.run_dir,m,repeat)
        for family,fi,norm in itertools.product(c['families'],range(len(c['fractions'])),c['normalizations']):
            path = shard_path(args.run_dir,repeat,family,fi,norm)
            if path.exists():
                read_shard(path,m,repeat,family,fi,norm)
                continue
            start = time.monotonic()
            rows = condition_rows(m,cache,data,repeat,family,fi,norm,grid(m),args.device,args.batch_size)
            save_shard(path,rows,m,repeat,family,fi,norm)
            print(f'{path.name}: {len(rows)} settings in {time.monotonic()-start:.1f}s',flush=True)
    print(f'Array rank {args.rank}/{args.split} complete',flush=True)


def parity_check(device):
    """Both mathematical CPU reference and actual CUDA, with a directed fixture."""
    rng = np.random.default_rng(87654)
    w = rng.normal(0,.035,(3,12,12)); b = rng.normal(0,.2,(3,12))
    leaks = np.array([.6,1.,.8]); u = rng.uniform(-1,1,(120,1))
    ref = np.zeros((3,12)); second = np.full((3,12),.5); history=[]
    for t in range(len(u)):
        ref=(1-leaks[:,None])*ref+leaks[:,None]*np.tanh(np.einsum('bi,bij->bj',ref,w)+b*u[t,0])
        history.append(ref.copy())
        if t<30:
            second=(1-leaks[:,None])*second+leaks[:,None]*np.tanh(np.einsum('bi,bij->bj',second,w)+b*u[t,0])
            if t==29: expected_gap=np.sqrt(np.mean((second-ref)**2,axis=1))
    expected=np.transpose(history,(1,0,2))
    states,gap=simulate_batch(w,b,leaks,u,30,device,torch.float64)
    error=float(np.max(abs(states-expected)))
    if not np.allclose(states,expected,rtol=1e-10,atol=1e-12) or not np.allclose(gap,expected_gap,rtol=1e-8,atol=1e-12):
        raise ValueError(f'Recurrence/reference mismatch on {device}: {error}')
    return dict(max_absolute_state_error=error)


def preflight(args):
    m=load_run(args.run_dir)
    if not args.device.startswith('cuda'):
        raise ValueError('Production preflight requires an actual GPU')
    runtime=configure_device(args.device)
    passed=args.run_dir/'preflight/PASSED.json'
    if passed.exists():
        gate=json.loads(passed.read_text())
        if gate['experiment_id']!=m['experiment_id'] or gate['runtime']['environment']!=runtime['environment']:
            raise ValueError('Existing preflight is incompatible')
        if digest((args.run_dir/'preflight/smoke.csv').read_bytes()) != gate['smoke_sha256']:
            raise ValueError('Existing preflight smoke output changed')
        print('Existing preflight already passed',flush=True)
        return
    import subprocess,sys
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_architecture_pipeline.py'],check=True,cwd=ROOT)
    parity=parity_check(args.device)
    # Screen every prespecified task stream before starting the expensive array.
    for repeat in range(m['config']['repeats']):
        tasks_cached(args.run_dir,m,repeat)
    cache=networks_cached(args.run_dir,m,0); data=tasks_cached(args.run_dir,m,0)
    # Check the complete CPU/GPU scoring path on a real, stable CE condition.
    stable_fi=int(np.argmin(abs(np.asarray(m['config']['fractions'])-.2)))
    check_point=[(.6,1.,.1)]
    cpu=condition_rows(m,cache,data,0,'ce_empirical',stable_fi,'own_radius',check_point,'cpu',1)[0]
    gpu=condition_rows(m,cache,data,0,'ce_empirical',stable_fi,'own_radius',check_point,args.device,1)[0]
    scoring_parity={k:abs(cpu[k]-gpu[k]) for k in ('MC','MC_extended','NARMA10_test_nmse')}
    tolerance=1e-6 if m['config']['state_dtype']=='float64' else .02
    if any(error>tolerance for error in scoring_parity.values()):
        raise ValueError(f'CPU/GPU metric mismatch: {scoring_parity}')
    points=grid(m); selected=[points[i] for i in sorted({0,len(points)//2,len(points)-1})]
    fractions=m['config']['fractions']
    indices=sorted({int(np.argmin(abs(np.asarray(fractions)-f))) for f in (0,.2,.5)})
    smoke=[]; start=time.monotonic()
    for family,fi,norm in itertools.product(m['config']['families'],indices,m['config']['normalizations']):
        smoke.extend(condition_rows(m,cache,data,0,family,fi,norm,selected,args.device,args.batch_size))
        print(f'Preflight {family} EI={fractions[fi]:.0%} {norm}',flush=True)
    write_csv(args.run_dir/'preflight/smoke.csv',smoke)
    gate=dict(experiment_id=m['experiment_id'],runtime=runtime,parity=parity,scoring_parity=scoring_parity,
              tested_conditions=len(smoke),all_task_streams_checked=m['config']['repeats'],
              smoke_sha256=digest((args.run_dir/'preflight/smoke.csv').read_bytes()),
              smoke_seconds=time.monotonic()-start)
    write_json(passed,gate)
    print('GPU preflight PASSED; production array may start',flush=True)


def merge(args):
    m=load_run(args.run_dir); c=m['config']; rows=[]
    for repeat,family,fi,norm in itertools.product(range(c['repeats']),c['families'],range(len(c['fractions'])),c['normalizations']):
        rows.extend(read_shard(shard_path(args.run_dir,repeat,family,fi,norm),m,repeat,family,fi,norm))
    if len(rows)!=m['expected_evaluations']:
        raise ValueError('Unexpected merged observation count')
    out=args.run_dir/'architecture_mc_narma.ALL.csv'
    if out.exists():
        meta=json.loads(Path(str(out)+'.json').read_text())
        if meta!=dict(experiment_id=m['experiment_id'],sha256=digest(out.read_bytes())):
            raise ValueError('Existing merged result changed')
        with out.open() as f:
            old=list(csv.DictReader(f))
        if old!=rows:
            raise ValueError('Existing merged result differs from shards')
    else:
        write_csv(out,rows)
        write_json(str(out)+'.json',dict(experiment_id=m['experiment_id'],sha256=digest(out.read_bytes())))
    print(f'Validated/merged {len(rows):,} rows: {out}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--run-dir',type=Path,required=True)
    q.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    q.add_argument('--normalizations',nargs='+',choices=NORMALIZATIONS,default=list(NORMALIZATIONS))
    q.add_argument('--fractions',nargs='+',type=float,default=[0,.1,.2,.3,.4,.5])
    q.add_argument('--rhos',nargs='+',type=float,default=[.6,.95,1.05])
    q.add_argument('--leaks',nargs='+',type=float,default=[.6,1.])
    q.add_argument('--inputs',nargs='+',type=float,default=[.01,.1,.5,1.5])
    q.add_argument('--repeats',type=int,default=100)
    q.add_argument('--seed',type=int,default=20261119)
    q.add_argument('--swaps-per-edge',type=int,default=20)
    q.add_argument('--state-dtype',choices=['float64','float32'],default='float64')
    for name,default in [('washout',500),('train',1500),('validation',500),('test',500),('narma-burnin',1000),('mc-max-delay',30),('mc-diagnostic-delay',60)]:
        q.add_argument('--'+name,type=int,default=default)
    for command in ('preflight','run','merge'):
        q=sub.add_parser(command); q.add_argument('--run-dir',type=Path,required=True)
        if command!='merge':
            q.add_argument('--device',default='cuda:0'); q.add_argument('--batch-size',type=int,default=24)
        if command=='run':
            q.add_argument('--rank',type=int,default=0); q.add_argument('--split',type=int,default=1)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    {'prepare':prepare,'preflight':preflight,'run':run_sweep,'merge':merge}[args.command](args)

if __name__=='__main__':
    main()
