#!/usr/bin/env python3
"""Consolidate completed paired WS controls and redraw verified cached analyses.

Original run manifests, merged CSVs and analyses are never modified. CSV fields
are reordered as strings, preserving every original numeric value and row ID.
Cached bootstrap intervals are reused only after checking provenance, hashes,
complete raw paired surfaces, and reconstructed point estimates.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from architecture_mc_narma import canonical,digest,file_digest,write_json
from tools.analyze_architecture_mc_narma import PARAMETERS,plots
from tools.plot_architecture_duels_by_model import main as duel_plots
from tools.replot_ipc_mean_zoom import main as ipc_zoom

METRICS=['MC','NARMA10_test_nmse','IPC']
LEVELS=['repeat_id','family','negative_fraction','normalization',*PARAMETERS]
CONFIG_KEYS=['repeats','fractions','normalizations','rhos','leaks','inputs','seed',
             'seed_stride','tasks','ipc','state_dtype','bias','ws_probability',
             'input_projection','narma_stream_policy','swaps_per_edge']


def checked_manifest(root):
    m=json.loads((root/'manifest.json').read_text())
    identity={k:m[k] for k in ['protocol','config','source_sha256']}
    if digest(canonical(identity))!=m['experiment_id']:
        raise ValueError(f'Invalid source manifest: {root}')
    return m


def paired_config(parent,child,parent_root):
    a,b=parent['config'],child['config'];record=b['comparison_parent']
    if parent['protocol']!=child['protocol'] or record['experiment_id']!=parent['experiment_id']:
        raise ValueError('Parent identity/protocol mismatch')
    if record['manifest_sha256']!=file_digest(parent_root/'manifest.json'):
        raise ValueError('Parent manifest hash mismatch')
    if set(a['families'])&set(b['families']):raise ValueError('Overlapping model families')
    for key in CONFIG_KEYS:
        if canonical(a[key])!=canonical(b[key]):raise ValueError(f'Unpaired {key}')
    for name in ['Connectome/ce_adj.npy','Connectome/ce_unknown_sign_weights.npy',
                 'network_stats/architecture_sweep.py','network_stats/audited_temporal_tasks.py',
                 'network_stats/ipc_capacity.py','third_party/echotorch/NARMADataset.py','util/util.py']:
        if parent['source_sha256'][name]!=child['source_sha256'][name]:
            raise ValueError(f'Incompatible scientific source: {name}')
    return dict(a,families=a['families']+b['families'])


def validate_raw(paths,manifests,c):
    """Verify all raw keys and paired input vectors; retain compact score cube."""
    levels=[list(range(c['repeats'])),c['families'],c['fractions'],c['normalizations'],
            c['rhos'],c['leaks'],c['inputs']]
    shape=tuple(map(len,levels));seen=np.zeros(np.prod(shape),dtype=bool)
    values=np.empty((len(seen),len(METRICS)));input_hashes={};sources=[]
    for path,m in zip(paths,manifests):
        meta=json.loads(Path(str(path)+'.json').read_text());sha=file_digest(path)
        if meta!=dict(experiment_id=m['experiment_id'],sha256=sha):
            raise ValueError(f'Changed raw CSV: {path}')
        count=0
        with pd.read_csv(path,usecols=LEVELS+METRICS+['experiment_id','protocol','seed','input_vector_sha256'],chunksize=25000) as chunks:
            for frame in chunks:
                if not frame.experiment_id.eq(m['experiment_id']).all() or not frame.protocol.eq(m['protocol']).all():
                    raise ValueError('Unexpected row experiment/protocol')
                if not set(frame.family).issubset(m['config']['families']):raise ValueError('Unexpected source family')
                indices=[pd.Categorical(frame[key],categories=level).codes for key,level in zip(LEVELS,levels)]
                if any((index<0).any() for index in indices):raise ValueError('Unexpected configuration/realization')
                linear=np.ravel_multi_index(indices,shape)
                if len(np.unique(linear))!=len(linear) or seen[linear].any():raise ValueError('Duplicate raw configuration')
                expected_seed=c['seed']+frame.repeat_id.to_numpy()*c['seed_stride']
                if not np.array_equal(frame.seed,expected_seed):raise ValueError('Unpaired random seeds')
                for repeat,h in frame[['repeat_id','input_vector_sha256']].drop_duplicates().itertuples(index=False,name=None):
                    if repeat in input_hashes and input_hashes[repeat]!=h:raise ValueError('Unpaired input projections')
                    input_hashes[repeat]=h
                scores=frame[METRICS].to_numpy()
                if not np.isfinite(scores).all():raise ValueError('Nonfinite score')
                if (scores[:,0]<0).any() or (scores[:,0]>c['tasks']['mc_max_delay']+1e-8).any() or (scores[:,1]<0).any() or (scores[:,2]>c['ipc']['target_count']+1e-8).any():
                    raise ValueError('Invalid metric bounds')
                values[linear]=scores;seen[linear]=True;count+=len(frame)
        if count!=m['expected_evaluations']:raise ValueError('Incomplete source run')
        sources.append(dict(path=str(path.resolve()),sha256=sha,experiment_id=m['experiment_id'],rows=count,
                            manifest_sha256=file_digest(path.parent/'manifest.json')))
        print(f'Validated {count:,} rows: {path}',flush=True)
    if not seen.all():raise ValueError('Missing paired configurations')
    return values.reshape((*shape,len(METRICS))),sources


def merge_csvs(paths,destination):
    """Keep parent bytes; reorder extension columns without parsing numbers."""
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.merge.tmp',delete=False) as f:tmp=Path(f.name)
    try:
        with paths[0].open(newline='') as stream:fields=next(csv.reader(stream))
        with tmp.open('wb') as target,paths[0].open('rb') as original:shutil.copyfileobj(original,target)
        with tmp.open('a',newline='') as target:
            writer=csv.writer(target)
            for path in paths[1:]:
                with path.open(newline='') as stream:
                    reader=csv.reader(stream);other=next(reader)
                    if len(other)!=len(fields) or set(other)!=set(fields):raise ValueError('CSV schemas differ')
                    order=[other.index(name) for name in fields]
                    for row in reader:
                        if len(row)!=len(fields):raise ValueError('Malformed CSV record')
                        writer.writerow([row[i] for i in order])
        os.link(tmp,destination)
    finally:tmp.unlink(missing_ok=True)
    return file_digest(destination)


def validate_cache(cache,sources,c,cube):
    manifest=json.loads((cache/'manifest.json').read_text())
    old=manifest['comparison_parent']
    if (manifest['experiment_id']!=sources[1]['experiment_id'] or manifest['source_sha256']!=sources[1]['sha256']
        or old['experiment_id']!=sources[0]['experiment_id'] or old['sha256']!=sources[0]['sha256']
        or old['analysis_rows']!=sum(s['rows'] for s in sources) or old['analysis_families']!=c['families']):
        raise ValueError('Cached analysis does not describe these sources')
    if manifest['analysis_sha256']!=file_digest(ROOT/'tools/analyze_architecture_mc_narma.py'):
        raise ValueError('Cached statistics were produced by a different analyzer')
    names=[name for name in manifest['outputs'] if name.endswith('.csv')]
    for name in names:
        if file_digest(cache/name)!=manifest['outputs'][name]:raise ValueError(f'Changed analysis table: {name}')
    tables={name:pd.read_csv(cache/name) for name in names}
    averaged=cube.mean(axis=0)
    means=averaged.mean(axis=(3,4,5))
    def index(row):
        return (c['families'].index(row.family),c['fractions'].index(row.negative_fraction),
                c['normalizations'].index(row.normalization),METRICS.index(row.metric))
    table=tables['mean_performance.csv']
    if len(table)!=len(c['families'])*len(c['fractions'])*len(c['normalizations'])*len(METRICS):raise ValueError('Incomplete cached means')
    for row in table.itertuples():
        if not np.isclose(row.mean,means[index(row)],rtol=1e-9,atol=1e-8):raise ValueError('Cached mean differs from raw observations')
    ce=c['families'].index('ce_empirical');advantages=averaged[ce][None,...]-averaged
    advantages[...,1]*=-1  # Lower NARMA NMSE wins.
    a=tables['configuration_advantages.csv']
    expected=(len(c['families'])-1)*len(c['fractions'])*len(c['normalizations'])*len(c['rhos'])*len(c['leaks'])*len(c['inputs'])*3
    if len(a)!=expected or a.duplicated(['control','negative_fraction','normalization',*PARAMETERS,'metric']).any():raise ValueError('Incomplete/duplicate cached advantages')
    for row in a.itertuples():
        ind=(c['families'].index(row.control),c['fractions'].index(row.negative_fraction),c['normalizations'].index(row.normalization),
             c['rhos'].index(row.rho_target),c['leaks'].index(row.leak),c['inputs'].index(row.input_scale),METRICS.index(row.metric))
        if not np.isclose(row.ce_advantage,advantages[ind],rtol=1e-9,atol=1e-8):raise ValueError('Cached duel advantage differs from raw observations')
    wins=tables['ce_winning_fraction.csv']
    for row in wins.itertuples():
        ind=(c['families'].index(row.control),c['fractions'].index(row.negative_fraction),c['normalizations'].index(row.normalization))
        actual=(advantages[ind][...,METRICS.index(row.metric)]>0).mean()
        if not np.isclose(row.ce_winning_fraction,actual,atol=1e-12):raise ValueError('Cached win fraction differs from raw observations')
    return manifest,tables


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--ws-run',type=Path,required=True)
    p.add_argument('--cached-analysis',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--analysis-dir',type=Path,required=True);args=p.parse_args(argv)
    for target in [args.out_dir,args.analysis_dir]:
        if target.exists():raise FileExistsError(f'Use a new output folder: {target}')
    parent,child=[checked_manifest(root) for root in [args.parent_run,args.ws_run]]
    c=paired_config(parent,child,args.parent_run)
    paths=[root/'architecture_mc_narma.ALL.csv' for root in [args.parent_run,args.ws_run]]
    cube,sources=validate_raw(paths,[parent,child],c)
    if sources[0]['sha256']!=child['config']['comparison_parent']['merged_sha256']:raise ValueError('Wrong parent merged data')
    cached,tables=validate_cache(args.cached_analysis,sources,c,cube);del cube
    print('All cached mean scores and paired duel point estimates match the raw data.',flush=True)
    dataset_id=digest(canonical(dict(sources=sources,config=c)))
    target=args.out_dir/'architecture_mc_narma.ALL.csv';combined_sha=merge_csvs(paths,target)
    write_json(Path(str(target)+'.json'),dict(dataset_id=dataset_id,sha256=combined_sha,rows=sum(s['rows'] for s in sources)))
    write_json(args.out_dir/'manifest.json',dict(kind='consolidated completed runs; not a new simulation',dataset_id=dataset_id,
        protocol=parent['protocol'],config=c,sources=sources,rows=sum(s['rows'] for s in sources),merged_sha256=combined_sha,
        original_row_experiment_ids_preserved=True,script_sha256=file_digest(Path(__file__))))
    shutil.copyfile(Path(__file__),args.out_dir/'consolidate_ws_results.py')
    args.analysis_dir.mkdir(parents=True)
    for name in tables:shutil.copyfile(args.cached_analysis/name,args.analysis_dir/name)
    m=dict(protocol=parent['protocol'],config=c)
    plots(tables['mean_performance.csv'],tables['ce_winning_fraction.csv'],tables['configuration_advantages.csv'],
          tables['hyperparameter_variance_components.csv'],args.analysis_dir,m)
    report=(args.cached_analysis/'REPORT.md').read_text()
    report+='\n## Consolidated local dataset\n\n'
    report+=f'All 950,400 original observations from the ten-model run and the two new WS controls are stored together in `{target.resolve()}`. Source experiment IDs and every original score are preserved. Both source runs and earlier plots remain unchanged.\n\n'
    report+=f'Plots were regenerated from checksum-verified sufficient-statistic tables in `{args.cached_analysis.resolve()}`. All grid means, configuration advantages and winning fractions were independently reconstructed from the complete raw observations and matched the cached tables. The original {cached["bootstrap"]} paired-realization bootstrap draws and their intervals are reused; no simulations or test-dependent IPC filtering were performed. See the dataset manifest for exact local sources and SHA-256 checksums.\n'
    (args.analysis_dir/'REPORT.md').write_text(report)
    write_json(args.analysis_dir/'manifest.json',dict(dataset_id=dataset_id,source=str(target.resolve()),source_sha256=combined_sha,
        source_runs=sources,config=c,bootstrap=cached['bootstrap'],bootstrap_seed=cached['bootstrap_seed'],bootstrap_unit=cached['bootstrap_unit'],
        cached_statistics=dict(path=str(args.cached_analysis.resolve()),manifest_sha256=file_digest(args.cached_analysis/'manifest.json'),
                               point_estimates_reconstructed_from_raw=True),
        plot_sha256=file_digest(ROOT/'tools/analyze_architecture_mc_narma.py'),script_sha256=file_digest(Path(__file__)),
        outputs={f.name:file_digest(f) for f in args.analysis_dir.iterdir() if f.is_file()}))
    # Retain the requested readable IPC axis, with a full-range inset and backup.
    original_argv=sys.argv
    try:
        sys.argv=['replot_ipc_mean_zoom.py','--analysis-dir',str(args.analysis_dir)]
        ipc_zoom()
    finally:sys.argv=original_argv
    duel_plots(['--analysis-dir',str(args.analysis_dir),'--out-dir',str(args.analysis_dir/'duel_plots_by_model')])
    (args.out_dir/'README.md').write_text(f'# Consolidated 12-model results\n\n950,400 observations; 100 paired realizations; 24 hyperparameter settings; 11 inhibitory-edge fractions; MC, NARMA10 and IPC; three normalization protocols.\n\nRaw scores: `architecture_mc_narma.ALL.csv`. Original row experiment IDs are preserved. This is a dataset bundle, not a newly simulated run. `manifest.json` records both original sources and complete settings.\n\n[Regenerated figures and report]({os.path.relpath(args.analysis_dir,args.out_dir)}/REPORT.md). Both original runs and their earlier analyses are preserved.\n')
    print(f'Combined dataset: {target}\nRegenerated plots: {args.analysis_dir}',flush=True)


if __name__=='__main__':main()
