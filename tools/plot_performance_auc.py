#!/usr/bin/env python3
"""Additional mean-performance-curve AUC figures in an existing AUC directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,SymLogNorm
import numpy as np
import pandas as pd
from architecture_mc_narma import file_digest,write_json
from tools.consolidate_ws_results import checked_manifest,validate_raw,METRICS
from tools.analyze_architecture_mc_narma import LABELS,COLORS
from tools.plot_architecture_duels_by_model import MODES
from util.ei_auc import curve_areas

METRIC_NAMES={'MC':'MC','NARMA10_test_nmse':'NARMA10','IPC':'IPC'}


def performance_aucs(cube,c,counts):
    """Integrate scores in their own units; average settings before integration."""
    n=len(cube);counts=np.asarray(counts)
    if counts.ndim!=2 or counts.shape[1]!=n or len(counts)<2 or (counts<0).any() or not np.all(counts.sum(axis=1)==n):
        raise ValueError('Invalid whole-realization bootstrap counts')
    x=np.asarray(c['fractions']);order=np.argsort(x);x=x[order]
    if len(x)<2 or x[-1]<=x[0]:raise ValueError('Need at least two inhibitory fractions')
    span=x[-1]-x[0];summary=[];individual=[]
    for ni,norm in enumerate(c['normalizations']):
        for mi,metric in enumerate(METRICS):
            for fi,family in enumerate(c['families']):
                values=cube[:,fi,:,ni,:,:,:,mi][:,order]
                surfaces=[('ALL',values.mean(axis=(2,3,4)),len(c['rhos'])*len(c['leaks'])*len(c['inputs']))]
                surfaces += [(f'{value:g}',values[...,j].mean(axis=(2,3)),len(c['rhos'])*len(c['leaks'])) for j,value in enumerate(c['inputs'])]
                for level,surface,nsettings in surfaces:
                    per_seed=curve_areas(surface,x)[0]/span
                    lo,hi=np.quantile(counts@per_seed/n,[.025,.975])
                    common=dict(normalization=norm,metric=metric,family=family,input_scale=level)
                    summary.append(dict(**common,performance_auc=float(per_seed.mean()),ci_low=float(lo),ci_high=float(hi),
                        n_realizations=n,n_configurations=nsettings,fraction_min=float(x[0]),fraction_max=float(x[-1])))
                    individual.extend(dict(**common,repeat_id=repeat,performance_auc=float(value)) for repeat,value in enumerate(per_seed))
    return pd.DataFrame(summary),pd.DataFrame(individual)


def short_label(family):
    return LABELS[family].replace('exact CE signed weights','CE weight multiset').replace('magnitudes','weights')


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)


def performance_panel(ax,d,c,metric):
    d=d.set_index('family').loc[c['families']]
    for y,(family,row) in enumerate(d.iterrows()):
        color=COLORS[family]
        ax.hlines(y,row.ci_low,row.ci_high,color=color,lw=1.5)
        ax.scatter(row.performance_auc,y,color=color,marker='D' if family=='ce_empirical' else 'o',s=45 if family=='ce_empirical' else 30,zorder=3)
    if metric=='IPC':ax.set_xscale('symlog',linthresh=1);ax.axvline(0,color='.5',lw=.7)
    ax.set_xlabel('NARMA10 error AUC (lower is better)' if metric=='NARMA10_test_nmse' else f'{METRIC_NAMES[metric]} performance AUC (higher is better)')
    ax.grid(axis='x',alpha=.15)


def plot_performance(summary,c,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    all_grid=summary[summary.input_scale=='ALL'];families=c['families'];y=np.arange(len(families))
    for norm,(label,_,_) in MODES.items():
        fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True,layout='constrained')
        for ax,metric in zip(axes,METRICS):
            performance_panel(ax,all_grid[(all_grid.normalization==norm)&(all_grid.metric==metric)],c,metric)
            ax.set_title(METRIC_NAMES[metric])
        axes[0].set(yticks=y,yticklabels=[short_label(f) for f in families]);axes[0].invert_yaxis()
        fig.suptitle(f'{label}: AUC of mean-performance curves across E/I balance\n95% paired-realization intervals; IPC uses symmetric-log scaling and retains all estimates',fontsize=13)
        save(fig,out,f'performance_auc_{norm}.png')
    for metric in METRICS:
        fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True,sharex=True,layout='constrained')
        for ax,(norm,(label,_,_)) in zip(axes,MODES.items()):
            performance_panel(ax,all_grid[(all_grid.normalization==norm)&(all_grid.metric==metric)],c,metric)
            ax.set_title(label)
        axes[0].set(yticks=y,yticklabels=[short_label(f) for f in families]);axes[0].invert_yaxis()
        note='IPC uses a symmetric-log axis; all negative estimates retained' if metric=='IPC' else 'Same performance scale across all three normalization protocols'
        fig.suptitle(f'{METRIC_NAMES[metric]}: AUC of mean-performance curves\n{note}; 95% paired-realization bootstrap intervals',fontsize=13)
        save(fig,out,f'performance_auc_{METRIC_NAMES[metric]}.png')
    inputs=[f'{value:g}' for value in c['inputs']];conditional=summary[summary.input_scale!='ALL']
    for norm,(label,_,_) in MODES.items():
        fig,axes=plt.subplots(1,3,figsize=(17,8),sharey=True,layout='constrained')
        for ax,metric in zip(axes,METRICS):
            d=conditional[(conditional.metric==metric)&(conditional.normalization==norm)]
            matrix=d.pivot(index='family',columns='input_scale',values='performance_auc').loc[families,inputs]
            # Within each metric the same colour scale is used across normalizations.
            all_values=conditional[conditional.metric==metric].performance_auc
            if metric=='IPC':
                limit=max(1.,abs(all_values).max());scale=SymLogNorm(linthresh=1,vmin=-limit,vmax=limit);cmap='RdBu'
            else:
                scale=Normalize(vmin=0,vmax=max(float(all_values.max()),1e-12));cmap='viridis_r' if metric=='NARMA10_test_nmse' else 'viridis'
            im=ax.imshow(matrix,norm=scale,cmap=cmap,aspect='auto')
            for (i,j),value in np.ndenumerate(matrix.to_numpy()):
                rgb=np.asarray(im.cmap(im.norm(value))[:3]);lightness=rgb@np.array([.2126,.7152,.0722])
                ax.text(j,i,f'{value:.3g}',ha='center',va='center',fontsize=8,color='black' if lightness>.5 else 'white')
            ax.set(xticks=range(len(inputs)),xticklabels=inputs,xlabel='Input scale',title=METRIC_NAMES[metric]+(' (lower is better)' if metric=='NARMA10_test_nmse' else ' (higher is better)'))
            fig.colorbar(im,ax=ax,shrink=.7,label='Normalized performance AUC')
        axes[0].set(yticks=y,yticklabels=[short_label(f) for f in families])
        fig.suptitle(f'{label}: performance-curve AUC within each input scale\nEach input averages {len(c["rhos"])*len(c["leaks"])} radius/leak settings; IPC uses symmetric-log colours',fontsize=13)
        save(fig,out,f'performance_auc_by_input_{norm}.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--auc-dir',type=Path,required=True);args=p.parse_args(argv)
    out=args.auc_dir.resolve();old_manifest=out/'manifest.json';old=json.loads(old_manifest.read_text())
    reserved=['performance_auc_manifest.json','PERFORMANCE_AUC.md','performance_curve_auc_summary.csv',
              'performance_curve_auc_per_realization.csv','README.before_performance_auc.md','manifest.before_performance_auc.json']
    if any((out/name).exists() for name in reserved) or any(out.glob('performance_auc_*.png')):
        raise FileExistsError('Performance AUC extension already exists or is incomplete; preserve existing outputs')
    base=Path(old['source_analysis']);m=json.loads((base/'manifest.json').read_text());c=m['config']
    if file_digest(base/'manifest.json')!=old['source_analysis_manifest_sha256']:raise ValueError('Source analysis changed')
    for name,sha in old['outputs'].items():
        if file_digest(out/name)!=sha:raise ValueError(f'Existing AUC artifact changed: {name}')
    previous_files={f.name:file_digest(f) for f in out.iterdir() if f.is_file()}
    paths=[Path(s['path']) for s in old['source_runs']]
    cube,sources=validate_raw(paths,[checked_manifest(p.parent) for p in paths],c)
    if sources!=old['source_runs']:raise ValueError('Source runs changed')
    counts=np.random.default_rng(old['seed']).multinomial(c['repeats'],np.full(c['repeats'],1/c['repeats']),size=old['bootstrap'])
    summary,individual=performance_aucs(cube,c,counts)
    cached=pd.read_csv(out/'mean_performance_auc.csv')
    check=summary[summary.input_scale=='ALL'].merge(cached,on=['normalization','metric','family'],validate='one_to_one',suffixes=('','_previous'))
    if len(check)!=len(cached):raise ValueError('Missing full-grid AUCs')
    for name,previous in [('performance_auc','mean_performance_auc'),('ci_low','ci_low_previous'),('ci_high','ci_high_previous')]:
        np.testing.assert_allclose(check[name],check[previous],rtol=1e-9,atol=1e-8)
    curves_path=base/'mean_performance.csv'
    if file_digest(curves_path)!=m['outputs'][curves_path.name]:raise ValueError('Mean-performance curve table changed')
    curves=pd.read_csv(curves_path)
    for row in check.itertuples():
        curve=curves[(curves.normalization==row.normalization)&(curves.metric==row.metric)&(curves.family==row.family)].sort_values('negative_fraction')
        expected=curve_areas(curve['mean'],curve.negative_fraction)[0]/(row.fraction_max-row.fraction_min)
        np.testing.assert_allclose(row.performance_auc,expected,rtol=1e-9,atol=1e-8)
    summary.to_csv(out/'performance_curve_auc_summary.csv',index=False);individual.to_csv(out/'performance_curve_auc_per_realization.csv',index=False)
    plot_performance(summary,c,out)
    text='''# AUC of mean-performance curves

These figures integrate each model's mean performance over inhibitory-edge
fraction, then divide by the observed E/I range. They retain the performance
metric's units. MC and IPC are better when higher; NARMA10 NMSE is better when
lower. The integral uses trapezoidal interpolation over the eleven sampled
fractions from 0 to 1. It is not a percentage of hyperparameter wins.

| Scaling | All tasks | By input scale |
|---|---|---|
'''
    for norm,(label,_,_) in MODES.items():text+=f'| {label} | [PNG](performance_auc_{norm}.png) | [PNG](performance_auc_by_input_{norm}.png) |\n'
    text+='\nSeparate task figures, with the same task scale across normalization panels:\n\n'
    for metric in METRICS:text+=f'- [{METRIC_NAMES[metric]}](performance_auc_{METRIC_NAMES[metric]}.png)\n'
    text+=f'''
All twelve models are included. Full-grid results average 24 equally weighted
settings within each realization; each input-specific result averages six
radius/leak settings. Each realization's mean curve is integrated before
averaging AUCs across the {c['repeats']} paired realizations. Intervals resample
whole realizations jointly across fractions, models, inputs and tasks:
{old['bootstrap']} draws, seed {old['seed']}. No pointwise intervals were integrated.
Input heatmaps show point estimates; their intervals are recorded in the CSV.

IPC uses a symmetric-log display (linear between −1 and 1), retaining severe
negative held-out estimates and their influence on the mean. Input heatmaps use
the same colour normalization across scaling protocols within each metric.
No observations were removed, clipped, or replaced. Curves, weights and task
scores are unchanged. Statistical intervals are descriptive percentile intervals;
they do not establish that one model is significantly better than another.

The CSVs contain 540 mean-AUC estimates and 54,000 realization AUCs. All 108
full-grid point estimates and intervals reproduce the earlier performance-AUC
table; point estimates also match integration of the actual mean-curve table.
Uniform E/I weighting and linear interpolation between observations are explicit
analysis choices. Scores and AUCs have separate units across tasks.
'''
    (out/'PERFORMANCE_AUC.md').write_text(text)
    new_names=[f.name for f in out.iterdir() if f.is_file() and f.name not in previous_files]
    write_json(out/'performance_auc_manifest.json',dict(source_analysis=str(base),source_analysis_manifest_sha256=file_digest(base/'manifest.json'),
        previous_auc_manifest_sha256=previous_files['manifest.json'],source_runs=sources,bootstrap=old['bootstrap'],seed=old['seed'],
        bootstrap_unit='paired whole realization',estimand='AUC of mean-performance curve, normalized by observed E/I span',
        full_grid_estimates_and_intervals_verified=108,script_sha256=file_digest(Path(__file__)),utility_sha256=file_digest(ROOT/'util/ei_auc.py'),
        outputs={name:file_digest(out/name) for name in new_names}))
    # Extend the index while preserving the previous index and manifest verbatim.
    shutil.copyfile(out/'README.md',out/'README.before_performance_auc.md');shutil.copyfile(old_manifest,out/'manifest.before_performance_auc.json')
    with (out/'README.md').open('a') as f:f.write('\n## Additional performance-curve AUC figures\n\n[All nine performance-curve AUC figures and tables](PERFORMANCE_AUC.md): individual tasks, normalization protocols, and input-scale breakdowns.\n')
    old['outputs']['README.md']=file_digest(out/'README.md')
    old.setdefault('extensions',[]).append(dict(manifest='performance_auc_manifest.json',sha256=file_digest(out/'performance_auc_manifest.json'),
        previous_manifest='manifest.before_performance_auc.json',previous_readme='README.before_performance_auc.md'))
    old_manifest.write_text(json.dumps(old,indent=2)+'\n')
    for name,sha in previous_files.items():
        if name not in ['README.md','manifest.json'] and file_digest(out/name)!=sha:raise ValueError(f'Previous output changed: {name}')
    print(f'{out}: added 9 performance-AUC PNGs; {len(summary)} estimates and {len(individual)} per-realization AUCs.',flush=True)


if __name__=='__main__':main()
