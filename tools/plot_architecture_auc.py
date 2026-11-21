#!/usr/bin/env python3
"""Paired-realization AUCs of matched-setting win curves for the 12-model sweep."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from architecture_mc_narma import file_digest,write_json
from tools.consolidate_ws_results import checked_manifest,validate_raw,METRICS
from tools.analyze_architecture_mc_narma import LABELS
from tools.plot_architecture_duels_by_model import MODES
from util.ei_auc import curve_areas,win_fraction_auc

METRIC_LABELS={'MC':'MC','NARMA10_test_nmse':'NARMA10','IPC':'IPC'}


def compute_auc(cube,c,counts):
    """Count strict wins AFTER seed averaging; integrate observed E/I curves."""
    x=np.asarray(c['fractions']);order=np.argsort(x);x=x[order];span=x[-1]-x[0]
    if span<=0:raise ValueError('At least two distinct inhibitory fractions required')
    rows=[];performance=[];ce=c['families'].index('ce_empirical')
    for ni,norm in enumerate(c['normalizations']):
        for mi,metric in enumerate(METRICS):
            # Axes: realization, family, fraction, rho, leak, input.
            values=cube[:,:,:,ni,:,:,:,mi][:,:,order]
            for fi,family in enumerate(c['families']):
                per_seed=curve_areas(values[:,fi].mean(axis=(2,3,4)),x)[0]/span
                boot=counts@per_seed/len(per_seed)
                lo,hi=np.quantile(boot,[.025,.975])
                performance.append(dict(normalization=norm,metric=metric,family=family,
                    mean_performance_auc=float(per_seed.mean()),ci_low=float(lo),ci_high=float(hi)))
                if fi==ce:continue
                delta=(values[:,ce]-values[:,fi])*(-1 if metric=='NARMA10_test_nmse' else 1)
                subsets=[('ALL',delta.reshape(len(delta),len(x),-1))]
                subsets += [(f'{value:g}',delta[...,j].reshape(len(delta),len(x),-1)) for j,value in enumerate(c['inputs'])]
                for level,y in subsets:
                    result=win_fraction_auc(y,counts,x)
                    # Existing utility names the comparator "er"; here it is any control.
                    result={key.replace('er_','control_',1):value/span for key,value in result.items()}
                    if not np.isclose(sum(result[f'{name}_win_auc_percent'] for name in ['ce','control','tie']),100):
                        raise ValueError('Win/tie AUCs do not partition the full domain')
                    rows.append(dict(normalization=norm,metric=metric,control=family,input_scale=level,
                        n_realizations=len(y),n_configurations=y.shape[-1],fraction_min=x[0],fraction_max=x[-1],**result))
    return pd.DataFrame(rows),pd.DataFrame(performance)


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)


def draw(auc,performance,c,out):
    controls=[family for family in c['families'] if family!='ce_empirical']
    short={family:LABELS[family].replace('exact CE signed weights','CE weight multiset').replace('magnitudes','weights') for family in c['families']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    primary=auc[auc.input_scale=='ALL'];y=np.arange(len(controls))
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
    for ax,metric in zip(axes,METRICS):
        for j,(norm,(label,color,marker)) in enumerate(MODES.items()):
            d=primary[(primary.metric==metric)&(primary.normalization==norm)].set_index('control').loc[controls]
            ypos=y+(j-1)*.22
            ax.hlines(ypos,d.ce_ci_low,d.ce_ci_high,color=color,lw=1.2)
            ax.scatter(d.ce_win_auc_percent,ypos,color=color,marker=marker,s=35,label=label,zorder=3)
        ax.axvline(50,color='.45',ls=':',lw=.8);ax.set(xlim=(-2,102),xticks=[0,25,50,75,100],title=METRIC_LABELS[metric],xlabel='CE win AUC (%)')
        ax.grid(axis='x',alpha=.15)
    axes[0].set(yticks=y,yticklabels=[short[f] for f in controls]);axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('AUC of matched hyperparameter settings won by empirical CE\nUniform E/I weighting; 95% paired-realization bootstrap intervals',fontsize=14)
    save(fig,out,'auc_win_fraction_overview.png')
    for norm,(label,_,_) in MODES.items():
        fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
        for ax,metric in zip(axes,METRICS):
            d=primary[(primary.normalization==norm)&(primary.metric==metric)].set_index('control').loc[controls]
            ce=d.ce_win_auc_percent.to_numpy();control=d.control_win_auc_percent.to_numpy();ties=d.tie_win_auc_percent.to_numpy()
            ax.barh(y,ce,color='#009e73',label='CE wins');ax.barh(y,control,left=ce,color='#d55e00',alpha=.7,label='Control wins')
            ax.barh(y,ties,left=ce+control,color='.7',label='Ties')
            ax.hlines(y,d.ce_ci_low,d.ce_ci_high,color='black',lw=1.1)
            ax.scatter(ce,y,s=14,color='black',zorder=4)
            for pos,value in zip(y,ce):
                ax.text(value+1 if value<85 else value-1,pos,f'{value:.1f}',ha='left' if value<85 else 'right',va='center',fontsize=8,
                        bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.2))
            ax.axvline(50,color='.3',lw=.7,ls=':');ax.set(xlim=(0,100),xlabel='Normalized win-curve AUC (%)',title=METRIC_LABELS[metric])
        axes[0].set(yticks=y,yticklabels=[short[f] for f in controls]);axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f'{label}: area under the fraction-of-settings-won curves\nWhiskers: CE AUC 95% paired-realization intervals',fontsize=14)
        save(fig,out,f'auc_win_fraction_{norm}.png')
        fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
        inputs=[f'{v:g}' for v in c['inputs']]
        for ax,metric in zip(axes,METRICS):
            d=auc[(auc.normalization==norm)&(auc.metric==metric)&(auc.input_scale!='ALL')]
            matrix=d.pivot(index='control',columns='input_scale',values='ce_win_auc_percent').loc[controls,inputs]
            im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
            ax.set(xticks=range(len(inputs)),xticklabels=inputs,xlabel='Input scale',title=METRIC_LABELS[metric])
            for (i,j),value in np.ndenumerate(matrix.to_numpy()):ax.text(j,i,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white' if value<55 else 'black')
        axes[0].set(yticks=y,yticklabels=[short[f] for f in controls])
        fig.colorbar(im,ax=axes,shrink=.75,label='CE win AUC (%)')
        fig.suptitle(f'{label}: CE win AUC within each input scale\nEach input counts {len(c["rhos"])*len(c["leaks"])} matched radius/leak settings',fontsize=14)
        save(fig,out,f'auc_by_input_{norm}.png')
    # Supporting task performance: separate metric axes retain their own units.
    families=c['families'];y=np.arange(len(families))
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
    for ax,metric in zip(axes,METRICS):
        for j,(norm,(label,color,marker)) in enumerate(MODES.items()):
            d=performance[(performance.metric==metric)&(performance.normalization==norm)].set_index('family').loc[families]
            ypos=y+(j-1)*.22;ax.hlines(ypos,d.ci_low,d.ci_high,color=color,lw=1.2)
            ax.scatter(d.mean_performance_auc,ypos,color=color,marker=marker,s=30,label=label,zorder=3)
        if metric=='IPC':ax.set_xscale('symlog',linthresh=1);ax.axvline(0,color='.5',lw=.7)
        ax.set(title=METRIC_LABELS[metric],xlabel='Mean-error AUC (lower is better)' if metric=='NARMA10_test_nmse' else 'Mean-score AUC (higher is better)')
        ax.grid(axis='x',alpha=.15)
    axes[0].set(yticks=y,yticklabels=[short[f] for f in families]);axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Supporting comparison: AUC of grid-mean performance\nSeparate metric scales; IPC uses a symmetric-log axis and retains extreme negative estimates',fontsize=13)
    save(fig,out,'auc_mean_performance.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261119);args=p.parse_args(argv)
    base=args.analysis_dir.resolve();out=args.out_dir or base/'auc'
    if out.exists():raise FileExistsError('Use a new AUC directory; no overwrite')
    if args.bootstrap<2:raise ValueError('Need at least two bootstrap draws')
    m=json.loads((base/'manifest.json').read_text());c=m['config'];sources=m['source_runs']
    paths=[Path(s['path']) for s in sources];manifests=[checked_manifest(path.parent) for path in paths]
    cube,actual_sources=validate_raw(paths,manifests,c)
    if actual_sources!=sources:raise ValueError('Raw sources differ from the consolidated analysis')
    counts=np.random.default_rng(args.seed).multinomial(c['repeats'],np.full(c['repeats'],1/c['repeats']),size=args.bootstrap)
    auc,performance=compute_auc(cube,c,counts)
    plotted=base/'ce_winning_fraction.csv'
    if file_digest(plotted)!=m['outputs'][plotted.name]:raise ValueError('Win-curve table changed')
    curves=pd.read_csv(plotted)
    for row in auc[auc.input_scale=='ALL'].itertuples():
        curve=curves[(curves.normalization==row.normalization)&(curves.metric==row.metric)&(curves.control==row.control)].sort_values('negative_fraction')
        if len(curve)!=len(c['fractions']):raise ValueError('Incomplete original win curve')
        expected=100*curve_areas(curve.ce_winning_fraction,curve.negative_fraction)[0]/(row.fraction_max-row.fraction_min)
        if not np.isclose(expected,row.ce_win_auc_percent,atol=1e-10):raise ValueError('AUC does not match the plotted curve')
    out.mkdir(parents=True);auc.to_csv(out/'win_fraction_auc.csv',index=False);performance.to_csv(out/'mean_performance_auc.csv',index=False)
    draw(auc,performance,c,out)
    text='''# Area under the matched-setting win curves

The primary AUC integrates CE's fraction of matched hyperparameter settings won
across the observed inhibitory-edge sweep, divided by the sweep width. Values
run from 0 to 100%. Higher favors CE against the named control; 50% is half of
the integrated fixed-grid/EI domain. Strict wins are determined after averaging
performance across paired realizations. Lower NARMA10 NMSE determines a win.
Exact ties count separately. This is not ROC AUC.

[All controls and normalization protocols](auc_win_fraction_overview.png).

| Scaling | Full grid | By input scale |
|---|---|---|
'''
    for norm,(label,_,_) in MODES.items():text+=f'| {label} | [PNG](auc_win_fraction_{norm}.png) | [PNG](auc_by_input_{norm}.png) |\n'
    text+=f'''
The grid has {len(c['rhos'])*len(c['leaks'])*len(c['inputs'])} settings and
{c['repeats']} paired realizations; {len(c['fractions'])} observed E/I fractions
cover {100*min(c['fractions']):g}–{100*max(c['fractions']):g}% inhibition.
Trapezoidal integration assumes linear interpolation between observed curve
points and gives endpoints half the weight of equally spaced interior points.
Uniform weighting of E/I and this finite grid is an analysis choice, not an
empirical biological distribution. These AUCs measure breadth of relative wins;
small advantages at low absolute performance count equally to large advantages.

Intervals use {args.bootstrap} whole-realization bootstrap draws, seed {args.seed}.
The same resampling counts are shared across models, metrics, fractions and
settings. Each draw recomputes mean performance, strict wins and the complete
curve's AUC. Pointwise curve intervals are not integrated. These are descriptive
percentile intervals, with no multiple-comparison adjustment. The by-input
plots show point estimates; their intervals are in the CSV.

[Supporting mean-performance AUC](auc_mean_performance.png) summarizes score
magnitude on separate MC, NARMA10 and IPC axes. Its intervals resample whole
realizations. It is a distinct estimand; NARMA10 error AUC is better when lower.
The IPC axis uses symmetric-log scaling (linear between −1 and 1) to retain
extreme negative held-out estimates. No IPC estimates were removed or clipped.
All {len(c['families'])-1} full-grid control comparisons were checked against the
existing plotted win-fraction tables for every task and normalization protocol.
'''
    (out/'README.md').write_text(text)
    write_json(out/'manifest.json',dict(source_analysis=str(base),source_analysis_manifest_sha256=file_digest(base/'manifest.json'),
        source_runs=sources,bootstrap=args.bootstrap,seed=args.seed,bootstrap_unit='paired whole realization',
        normalized_by_observed_fraction_span=True,primary_estimand='AUC of CE seed-mean matched-setting win-fraction curve',
        script_sha256=file_digest(Path(__file__)),utility_sha256=file_digest(ROOT/'util/ei_auc.py'),
        outputs={f.name:file_digest(f) for f in out.iterdir() if f.is_file()}))
    print(f'{out}: {len(list(out.glob("*.png")))} PNGs; {len(auc)} win-AUC estimates.',flush=True)


if __name__=='__main__':main()
