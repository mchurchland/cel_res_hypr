#!/usr/bin/env python3
"""Immutable, paired ER weight-distribution MC/NARMA experiment for Orca GPUs."""
import argparse
import csv
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import architecture_mc_narma as base
from network_stats.er_weight_distributions import FAMILIES,NORMALIZATIONS,PARENT_FAMILIES,LABELS,SETTINGS,build_cache,scaling_factor

ROOT=Path(__file__).resolve().parent
PROTOCOL='er-weight-distribution-mc-narma-v1'
EXTRA=('er_weight_distribution.py','network_stats/er_weight_distributions.py','tools/analyze_er_weight_distribution.py',
       'tests/test_er_weight_distribution.py','ER_WEIGHT_DISTRIBUTION.md','r_er_weight_distribution.sbatch',
       'r_er_weight_distribution_preflight.sbatch','r_er_weight_distribution_run.sbatch','r_er_weight_distribution_analyze.sbatch','util/ei_auc.py')
FILES=tuple(dict.fromkeys(base.SOURCES+base.DATA+EXTRA))
STRING_FIELDS={'experiment_id','protocol','family','normalization','raw_network_sha256','input_vector_sha256','state_dtype',
               'readout_dtype','backend','MC_ridge_alphas','MC_NARMA_source_experiment_id','MC_NARMA_parent_shard_sha256'}
INTEGER_FIELDS={'repeat_id','seed','nodes','edges','narma_generator_seed','narma_rejected_streams','directed_swaps',
                'MC_NARMA_reused','baseline_scores_reused','scaling_scores_reused'}


def prepare(a):
    parent=a.parent_run.resolve();old=json.loads((parent/'manifest.json').read_text())
    if base.digest(base.canonical({k:old[k] for k in ['protocol','config','source_sha256']}))!=old['experiment_id']:
        raise ValueError('Invalid parent manifest identity')
    for name in base.DATA+('network_stats/architecture_sweep.py','network_stats/audited_temporal_tasks.py','util/util.py','third_party/echotorch/NARMADataset.py'):
        if old['source_sha256'][name]!=base.file_digest(ROOT/name):raise ValueError(f'Incompatible parent scientific source: {name}')
    path=parent/'architecture_mc_narma.ALL.csv';metadata=json.loads(Path(str(path)+'.json').read_text())
    if metadata!=dict(experiment_id=old['experiment_id'],sha256=base.file_digest(path)):raise ValueError('Parent merged CSV changed')
    c={key:old['config'][key] for key in ['tasks','rhos','leaks','inputs','seed','seed_stride','state_dtype']}
    c.update(families=a.families,normalizations=a.normalizations,fractions=a.fractions,repeats=a.repeats,bias=0.,
             distributions=SETTINGS,input_projection=old['config']['input_projection'],
             comparison_parent=dict(path=str(parent),manifest_sha256=base.file_digest(parent/'manifest.json'),
                 experiment_id=old['experiment_id'],merged_sha256=metadata['sha256']))
    if a.repeats<1 or a.repeats>old['config']['repeats']:raise ValueError('Parent lacks requested paired repeats')
    for key in ['fractions','normalizations','families']:
        if not c[key] or len(set(c[key]))!=len(c[key]):raise ValueError(f'Empty/duplicate {key}')
    c['normalizations']=[norm for norm in NORMALIZATIONS if norm in a.normalizations]
    if any(f in PARENT_FAMILIES for f in c['families']) and not set(n for n in c['normalizations'] if n!='frobenius_match').issubset(old['config']['normalizations']):
        raise ValueError('Parent lacks a requested baseline normalization')
    if not set(c['fractions']).issubset(old['config']['fractions']):raise ValueError('Parent lacks requested E/I fractions')
    if not all(np.isfinite(c['fractions'])):raise ValueError('Invalid E/I coordinates')
    hashes={name:base.file_digest(ROOT/name) for name in FILES}
    identity=dict(protocol=PROTOCOL,config=c,source_sha256=hashes);experiment_id=base.digest(base.canonical(identity))
    run=a.run_dir.resolve()
    if (run/'manifest.json').exists():
        if load_run(run)['experiment_id']!=experiment_id:raise ValueError('Changed run settings/code; use a new RUN_ID')
        print(f'Unchanged frozen run: {run}');return
    if run.exists() and any(run.iterdir()):raise ValueError('Refuse nonempty directory without a manifest')
    for name in FILES:base.atomic(run/'snapshot'/name,(ROOT/name).read_bytes())
    per_family=a.repeats*len(a.fractions)*len(c['rhos'])*len(c['leaks'])*len(c['inputs'])
    total=per_family*len(a.families)*len(a.normalizations)
    parent_reuse=per_family*sum(f in PARENT_FAMILIES for f in a.families)*sum(n in ['own_radius','shared_ce_scale'] for n in a.normalizations)
    planned_scaling_reuse=per_family*sum(f in ['empirical','gamma','lognormal'] for f in a.families) if set(['shared_ce_scale','frobenius_match']).issubset(a.normalizations) else 0
    manifest=dict(**identity,experiment_id=experiment_id,expected_evaluations=total,expected_parent_reuse_rows=parent_reuse,
        expected_scaling_reuse_rows=planned_scaling_reuse,expected_new_evaluations=total-parent_reuse-planned_scaling_reuse,
        expected_new_trajectories=2*(total-parent_reuse-planned_scaling_reuse),prepared_environment=base.environment(),
        data=old['data'],model_definitions=LABELS,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    base.write_json(run/'manifest.json',manifest)
    print(f'Prepared {run}: {total:,} rows; {parent_reuse:,} archived baseline rows reused; '
          f'{planned_scaling_reuse:,} planned identical-scaling copies; {manifest["expected_new_trajectories"]:,} new trajectories.',flush=True)


def load_run(run):
    run=Path(run);m=json.loads((run/'manifest.json').read_text())
    if m['protocol']!=PROTOCOL or base.digest(base.canonical({k:m[k] for k in ['protocol','config','source_sha256']}))!=m['experiment_id']:
        raise ValueError('Invalid experiment identity')
    for name,sha in m['source_sha256'].items():
        if base.file_digest(run/'snapshot'/name)!=sha:raise ValueError(f'Frozen source changed: {name}')
    for name in FILES:
        if base.file_digest(ROOT/name)!=m['source_sha256'][name]:raise ValueError(f'Use frozen snapshot code: {name}')
    return m


def networks_cached(run,m,repeat):
    path=Path(run)/'cache'/f'networks_{repeat:04d}.npz'
    if path.exists():return base.read_cache(path,m['experiment_id'])
    parent,old=base.parent_manifest(m);source=base.read_cache(parent/'cache'/path.name,old['experiment_id'])
    cache=build_cache(source,old['config']['fractions'],m['config']['fractions'],base.repeat_seed(m,repeat),m['config']['families'])
    base.write_cache(path,cache,m['experiment_id']);return cache


def factor_from_row(row):
    return scaling_factor(row['normalization'],float(row['rho_target']),float(row['raw_radius']),float(row['shared_ce_radius']),
                          float(row['raw_frobenius_norm']),float(row['reference_raw_frobenius_norm']))


def validate_rows(rows,m,repeat,family,fi,norm):
    base.validate_rows(rows,m,repeat,family,fi,norm,factor_check=factor_from_row)
    for row in rows:
        if not np.isclose(float(row['scaled_frobenius_norm']),float(row['scaling_factor'])*float(row['raw_frobenius_norm']),rtol=1e-12):
            raise ValueError('Incorrect scaled Frobenius norm')
        if norm=='frobenius_match':
            target=float(row['rho_target'])/float(row['shared_ce_radius'])*float(row['reference_raw_frobenius_norm'])
            if not np.isclose(float(row['scaled_frobenius_norm']),target,rtol=2e-10,atol=1e-12):raise ValueError('Frobenius matching failed')
        if family in ['gamma','lognormal']:
            if not np.allclose([float(row['mean_nonzero_raw_magnitude']),float(row['raw_magnitude_variance'])],
                [float(row['empirical_magnitude_mean']),float(row['empirical_magnitude_variance'])],rtol=1e-10,atol=1e-12):
                raise ValueError('Realized moments do not match the empirical distribution')
        reused=int(row['baseline_scores_reused']) or int(row['scaling_scores_reused'])
        if reused and (len(row['MC_NARMA_source_experiment_id'])!=64 or len(row['MC_NARMA_parent_shard_sha256'])!=64):
            raise ValueError('Missing score reuse provenance')


def read_shard(path,m,repeat,family,fi,norm):
    raw=Path(path).read_bytes();meta=json.loads(Path(str(path)+'.json').read_text())
    if meta!=dict(experiment_id=m['experiment_id'],sha256=base.digest(raw)):raise ValueError(f'Changed shard: {path}')
    rows=list(csv.DictReader(io.StringIO(raw.decode())));validate_rows(rows,m,repeat,family,fi,norm);return rows


def save_shard(path,rows,m,repeat,family,fi,norm):
    validate_rows(rows,m,repeat,family,fi,norm);base.write_csv(path,rows)
    base.write_json(str(path)+'.json',dict(experiment_id=m['experiment_id'],sha256=base.file_digest(path)))


def condition_rows(run,m,cache,data,repeat,family,fi,norm,points,device,batch_size,force_simulate=False):
    w,eigen,radius=base.raw_from_cache(cache,family,fi);magnitudes=np.abs(w[w!=0]);raw_f=float(np.linalg.norm(magnitudes))
    empirical=np.abs(cache[f'er_signed_empirical__{fi}__values']);reference_f=float(np.linalg.norm(empirical))
    ce_radius=float(cache[f'ce_empirical__{fi}__radius']);fraction=m['config']['fractions'][fi]
    factors=[scaling_factor(norm,p[0],radius,ce_radius,raw_f,reference_f) for p in points]
    rows=None;baseline_reuse=0;scaling_reuse=0;source_id='';source_sha=''
    if not force_simulate and family in PARENT_FAMILIES and norm in ['own_radius','shared_ce_scale']:
        parent_rows,source_id,source_sha=base.parent_condition(m,repeat,PARENT_FAMILIES[family],fraction,norm)
        rows=[dict(parent_rows[tuple(p)]) for p in points];baseline_reuse=1
    if not force_simulate and rows is None and norm=='frobenius_match':
        source=base.shard_path(run,repeat,family,fi,'shared_ce_scale')
        shared_factors=[scaling_factor('shared_ce_scale',p[0],radius,ce_radius,raw_f,reference_f) for p in points]
        if source.exists() and factors==shared_factors:
            previous=read_shard(source,m,repeat,family,fi,'shared_ce_scale')
            lookup={tuple(float(r[k]) for k in ['rho_target','leak','input_scale']):r for r in previous}
            rows=[dict(lookup[tuple(p)]) for p in points];scaling_reuse=1
            baseline_reuse=int(rows[0]['baseline_scores_reused']);source_id=m['experiment_id'];source_sha=base.file_digest(source)
    if rows is not None:
        for row,factor in zip(rows,factors):
            if row['raw_network_sha256']!=base.digest(w.tobytes()) or row['input_vector_sha256']!=base.digest(cache['input_vector'].tobytes()) or not np.isclose(float(row['scaling_factor']),factor,rtol=1e-12):
                raise ValueError('Score reuse requires identical matrix, input and scaling')
    else:
        rows=base.condition_rows(m,cache,data,repeat,family,fi,norm,points,device,batch_size,reuse_parent=False,factors_override=factors)
    result=[]
    for row,factor in zip(rows,factors):
        # Legacy parent files may contain IPC; this experiment uses MC/NARMA only.
        row={key:(value if key in STRING_FIELDS else (int(float(value)) if key in INTEGER_FIELDS else float(value)))
             for key,value in row.items() if not key.startswith('IPC')}
        row.update(experiment_id=m['experiment_id'],protocol=PROTOCOL,family=family,normalization=norm,
            backend='parent_reuse' if baseline_reuse and not scaling_reuse else ('same_matrix_reuse' if scaling_reuse else device),
            MC_NARMA_reused=0,MC_NARMA_source_experiment_id=source_id,MC_NARMA_parent_shard_sha256=source_sha,
            baseline_scores_reused=baseline_reuse,scaling_scores_reused=scaling_reuse,
            scaling_factor=factor,post_scaling_radius=radius*factor,
            leaky_linearized_radius=float(np.max(abs((1-float(row['leak']))+float(row['leak'])*factor*eigen))),
            raw_frobenius_norm=raw_f,reference_raw_frobenius_norm=reference_f,scaled_frobenius_norm=raw_f*factor,
            raw_magnitude_variance=float(magnitudes.var()),raw_magnitude_cv=float(magnitudes.std()/magnitudes.mean()),
            raw_max_magnitude=float(magnitudes.max()),raw_mean_eigenvalue_magnitude=float(np.abs(eigen).mean()),
            scaled_mean_eigenvalue_magnitude=float(np.abs(eigen).mean()*factor),
            empirical_magnitude_mean=float(empirical.mean()),empirical_magnitude_variance=float(empirical.var()),
            fraction_below_empirical_min=float((magnitudes<empirical.min()).mean()),fraction_above_empirical_max=float((magnitudes>empirical.max()).mean()),
            distribution_fit_shape=float(cache[f'{family}__{fi}__fit_shape']),cap_cutoff_before_mean_restoration=float(cache[f'{family}__{fi}__cap_cutoff']),
            mean_restoration_factor=float(cache[f'{family}__{fi}__mean_restoration']))
        result.append(row)
    return result


def preflight(a):
    m=load_run(a.run_dir)
    if not a.device.startswith('cuda'):raise ValueError('Production preflight requires an actual GPU')
    runtime=base.configure_device(a.device);passed=a.run_dir/'preflight/PASSED.json'
    if passed.exists():
        gate=json.loads(passed.read_text())
        if gate['experiment_id']!=m['experiment_id'] or gate['runtime']!=runtime or gate['smoke_sha256']!=base.file_digest(a.run_dir/'preflight/smoke.csv'):
            raise ValueError('Existing preflight differs')
        print('Existing GPU preflight passed');return
    for pattern in ['test_architecture*.py','test_er_weight_distribution.py']:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',pattern],cwd=ROOT,check=True)
    parity=base.parity_check(a.device)
    for repeat in range(m['config']['repeats']):base.tasks_cached(a.run_dir,m,repeat)
    cache=networks_cached(a.run_dir,m,0);data=base.tasks_cached(a.run_dir,m,0);c=m['config']
    fi=int(np.argmin(abs(np.asarray(c['fractions'])-.2)));point=[(min(c['rhos']),max(c['leaks']),min(c['inputs'],key=lambda v:abs(v-.1)))]
    parity_errors=[]
    for family,norm in [(c['families'][0],'own_radius'),('gamma' if 'gamma' in c['families'] else c['families'][0],'frobenius_match')]:
        cpu=condition_rows(a.run_dir,m,cache,data,0,family,fi,norm,point,'cpu',1,True)[0]
        gpu=condition_rows(a.run_dir,m,cache,data,0,family,fi,norm,point,a.device,1,True)[0]
        errors={key:abs(cpu[key]-gpu[key]) for key in ['MC','MC_extended','NARMA10_test_nmse']}
        if max(errors.values())>1e-6:raise ValueError(f'CPU/GPU score mismatch: {errors}')
        parity_errors.append(dict(family=family,normalization=norm,errors=errors))
    smoke=[];points=base.grid(m);selection=[points[i] for i in sorted({0,len(points)//2,len(points)-1})]
    for family,fi,norm in itertools.product(c['families'],sorted({0,len(c['fractions'])//2,len(c['fractions'])-1}),c['normalizations']):
        smoke.extend(condition_rows(a.run_dir,m,cache,data,0,family,fi,norm,selection,a.device,a.batch_size,True))
        print(f'Preflight {family} EI={c["fractions"][fi]:.0%} {norm}',flush=True)
    # Explicitly verify archived baseline reuse against fresh GPU scores.
    baseline_checks=[]
    fi=int(np.argmin(abs(np.asarray(c['fractions'])-.2)))
    for family in set(c['families'])&set(PARENT_FAMILIES):
        inherited=condition_rows(a.run_dir,m,cache,data,0,family,fi,'own_radius',point,a.device,1)[0]
        fresh=condition_rows(a.run_dir,m,cache,data,0,family,fi,'own_radius',point,a.device,1,True)[0]
        errors={k:abs(inherited[k]-fresh[k]) for k in ['MC','NARMA10_test_nmse']}
        if max(errors.values())>1e-6:raise ValueError(f'Parent baseline differs: {errors}')
        baseline_checks.append(dict(family=family,errors=errors))
    base.write_csv(a.run_dir/'preflight/smoke.csv',smoke)
    base.write_json(passed,dict(experiment_id=m['experiment_id'],runtime=runtime,parity=parity,scoring_parity=parity_errors,
        baseline_checks=baseline_checks,all_task_streams_checked=c['repeats'],smoke_conditions=len(smoke),smoke_sha256=base.file_digest(a.run_dir/'preflight/smoke.csv')))
    print('GPU preflight PASSED',flush=True)


def run(a):
    m=load_run(a.run_dir);c=m['config'];runtime=base.configure_device(a.device)
    if not a.device.startswith('cuda'):raise ValueError('Production sweep requires GPU; refusing CPU fallback')
    if a.split<1 or not 0<=a.rank<a.split or a.batch_size<1:raise ValueError('Invalid array partition')
    gate=json.loads((a.run_dir/'preflight/PASSED.json').read_text())
    if gate['experiment_id']!=m['experiment_id'] or gate['runtime']!=runtime or gate['smoke_sha256']!=base.file_digest(a.run_dir/'preflight/smoke.csv'):
        raise ValueError('Missing or incompatible GPU preflight')
    for repeat in range(a.rank,c['repeats'],a.split):
        cache=networks_cached(a.run_dir,m,repeat);data=base.tasks_cached(a.run_dir,m,repeat)
        for family,fi,norm in itertools.product(c['families'],range(len(c['fractions'])),c['normalizations']):
            path=base.shard_path(a.run_dir,repeat,family,fi,norm)
            if path.exists():read_shard(path,m,repeat,family,fi,norm);continue
            start=time.monotonic();rows=condition_rows(a.run_dir,m,cache,data,repeat,family,fi,norm,base.grid(m),a.device,a.batch_size)
            save_shard(path,rows,m,repeat,family,fi,norm)
            print(f'{path.name}: {len(rows)} settings in {time.monotonic()-start:.1f}s',flush=True)


def merge(a):
    m=load_run(a.run_dir);c=m['config'];target=a.run_dir/'architecture_mc_narma.ALL.csv';count=0;fields=None
    with tempfile.NamedTemporaryFile(mode='w',newline='',dir=a.run_dir,suffix='.merge.tmp',delete=False) as f:
        temporary=Path(f.name)
        try:
            for repeat,family,fi,norm in itertools.product(range(c['repeats']),c['families'],range(len(c['fractions'])),c['normalizations']):
                rows=read_shard(base.shard_path(a.run_dir,repeat,family,fi,norm),m,repeat,family,fi,norm)
                if fields is None:fields=list(rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                if any(set(r)!=set(fields) for r in rows):raise ValueError('Shard schemas differ')
                writer.writerows(rows);count+=len(rows)
            if count!=m['expected_evaluations']:raise ValueError('Incomplete merged data')
            f.flush();os.fsync(f.fileno())
        except BaseException:temporary.unlink(missing_ok=True);raise
    try:
        sha=base.file_digest(temporary);metadata=dict(experiment_id=m['experiment_id'],sha256=sha)
        if target.exists():
            if json.loads(Path(str(target)+'.json').read_text())!=metadata or base.file_digest(target)!=sha:raise ValueError('Existing merge differs')
        else:os.link(temporary,target);base.write_json(str(target)+'.json',metadata)
    finally:temporary.unlink(missing_ok=True)
    print(f'Validated/merged {count:,} rows: {target}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--parent-run',type=Path,required=True)
    q.add_argument('--repeats',type=int,default=100);q.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    q.add_argument('--normalizations',nargs='+',choices=NORMALIZATIONS,default=list(NORMALIZATIONS))
    q.add_argument('--fractions',nargs='+',type=float,default=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
    for command in ['preflight','run','merge']:
        q=sub.add_parser(command);q.add_argument('--run-dir',type=Path,required=True)
        if command!='merge':q.add_argument('--device',default='cuda:0');q.add_argument('--batch-size',type=int,default=24)
        if command=='run':q.add_argument('--rank',type=int,default=0);q.add_argument('--split',type=int,default=1)
    return p


if __name__=='__main__':
    a=parser().parse_args();{'prepare':prepare,'preflight':preflight,'run':run,'merge':merge}[a.command](a)
