#!/usr/bin/env python3
"""PNG/CSV paired performance and matched-setting comparisons for ER weights."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-er-weights')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import er_weight_distribution as run
import architecture_mc_narma as base
from network_stats.er_weight_distributions import LABELS,COLORS
from util.ei_auc import curve_areas,win_fraction_auc

METRICS={'MC':1,'NARMA10_test_nmse':-1}
MODES={'own_radius':'Own radius','shared_ce_scale':'Shared CE multiplier','frobenius_match':'Matched Frobenius norm'}


def summarize(data,m,weights):
    c=m['config'];levels=[list(range(c['repeats'])),c['families'],c['fractions'],c['normalizations'],c['rhos'],c['leaks'],c['inputs']]
    keys=['repeat_id','family','negative_fraction','normalization','rho_target','leak','input_scale'];shape=tuple(map(len,levels))
    if data.duplicated(keys).any() or len(data)!=np.prod(shape):raise ValueError('Incomplete/duplicated data')
    indices=[pd.Categorical(data[key],categories=level).codes for key,level in zip(keys,levels)]
    if any((i<0).any() for i in indices):raise ValueError('Unexpected grid coordinates')
    cube=np.empty((*shape,len(METRICS)));flat=cube.reshape(-1,len(METRICS))
    flat[np.ravel_multi_index(indices,shape)]=data[list(METRICS)].to_numpy()
    if not np.isfinite(cube).all():raise ValueError('Nonfinite scores')
    curves=cube.mean(axis=(4,5,6));summary=[];wins=[];advantages=[];auc=[];win_auc=[]
    x=np.asarray(c['fractions']);order=np.argsort(x);span=x.max()-x.min()
    if span<=0:raise ValueError('AUC requires at least two distinct E/I fractions')
    emp=c['families'].index('empirical') if 'empirical' in c['families'] else None
    for ni,norm in enumerate(c['normalizations']):
        for mi,(metric,direction) in enumerate(METRICS.items()):
            for j,family in enumerate(c['families']):
                curve=curves[:,j,:,ni,mi];bootstrap=weights@curve;intervals=np.quantile(bootstrap,[.025,.975],axis=0)
                for fi,f in enumerate(c['fractions']):summary.append(dict(normalization=norm,metric=metric,family=family,negative_fraction=f,
                    mean=float(curve[:,fi].mean()),ci_low=float(intervals[0,fi]),ci_high=float(intervals[1,fi])))
                per_seed=curve_areas(curve[:,order],x[order])[0]/span;ci=np.quantile(weights@per_seed,[.025,.975])
                auc.append(dict(normalization=norm,metric=metric,family=family,performance_auc=float(per_seed.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
                if emp is None or j==emp:continue
                delta=direction*(cube[:,emp,:,ni,:,:,:,mi]-cube[:,j,:,ni,:,:,:,mi]);surface=delta.reshape(c['repeats'],len(x),-1)
                boot=(weights@surface.reshape(c['repeats'],-1)).reshape(len(weights),len(x),-1)
                means=surface.mean(axis=0);intervals=np.quantile((boot>0).mean(axis=-1),[.025,.975],axis=0)
                for fi,f in enumerate(c['fractions']):
                    wins.append(dict(normalization=norm,metric=metric,control=family,negative_fraction=f,
                        empirical_winning_fraction=float((means[fi]>0).mean()),ties=int((means[fi]==0).sum()),ci_low=float(intervals[0,fi]),ci_high=float(intervals[1,fi]),
                        n_configurations=means.shape[-1],n_realizations=c['repeats']))
                    for pi,point in enumerate(base.grid(m)):
                        ci=np.quantile(boot[:,fi,pi],[.025,.975])
                        advantages.append(dict(normalization=norm,metric=metric,control=family,negative_fraction=f,
                            **dict(zip(['rho_target','leak','input_scale'],point)),empirical_advantage=float(means[fi,pi]),ci_low=float(ci[0]),ci_high=float(ci[1])))
                result=win_fraction_auc(surface[:,order],np.rint(weights*c['repeats']).astype(int),x[order])
                win_auc.append(dict(normalization=norm,metric=metric,control=family,empirical_win_auc_percent=result['ce_win_auc_percent']/span,
                    ci_low=result['ce_ci_low']/span,ci_high=result['ce_ci_high']/span,control_win_auc_percent=result['er_win_auc_percent']/span,tie_auc_percent=result['tie_win_auc_percent']/span))
    return tuple(pd.DataFrame(v) for v in [summary,wins,advantages,auc,win_auc])


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)


def plots(summary,wins,auc,win_auc,c,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for norm in c['normalizations']:
        for metric in METRICS:
            for kind,table,value,prefix in [('Mean performance',summary,'mean','mean'),('Empirical wins against each condition',wins,'empirical_winning_fraction','win_fraction')]:
                if table.empty:continue
                fig,ax=plt.subplots(figsize=(10,6));column='family' if kind=='Mean performance' else 'control'
                for family in c['families']:
                    if column=='control' and family=='empirical':continue
                    d=table[(table.normalization==norm)&(table.metric==metric)&(table[column]==family)].sort_values('negative_fraction')
                    multiplier=1 if column=='family' else 100
                    ax.plot(d.negative_fraction*100,d[value]*multiplier,color=COLORS[family],marker='o',ms=3,label=LABELS[family])
                    ax.fill_between(d.negative_fraction*100,d.ci_low*multiplier,d.ci_high*multiplier,color=COLORS[family],alpha=.1)
                ax.set(xlabel='Inhibitory edges (%)',title=f'{MODES[norm]}: {metric}',ylabel=(metric+' (higher is better)' if metric=='MC' else 'NARMA10 NMSE (lower is better)') if column=='family' else 'Matched settings empirical weights win (%)')
                if column=='control':ax.set_ylim(-2,102);ax.axhline(50,color='.5',ls=':',lw=.8)
                else:ax.set_ylim(bottom=0)
                ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
                save(fig,out,f'{prefix}_{metric}_{norm}.png')
    for metric in METRICS:
        fig,axes=plt.subplots(1,len(c['normalizations']),figsize=(5*len(c['normalizations'])+3,7),sharey=True,squeeze=False,layout='constrained')
        y=np.arange(len(c['families']))
        for ax,norm in zip(axes[0],c['normalizations']):
            d=auc[(auc.normalization==norm)&(auc.metric==metric)].set_index('family').loc[c['families']]
            for pos,(family,row) in enumerate(d.iterrows()):
                ax.hlines(pos,row.ci_low,row.ci_high,color=COLORS[family]);ax.scatter(row.performance_auc,pos,color=COLORS[family],s=30)
            ax.set(title=MODES[norm],xlabel='Performance AUC (higher is better)' if metric=='MC' else 'Error AUC (lower is better)');ax.grid(axis='x',alpha=.15)
        axes[0,0].set(yticks=y,yticklabels=[LABELS[f] for f in c['families']]);axes[0,0].invert_yaxis()
        fig.suptitle(f'{metric}: area under mean-performance curves; 95% paired-realization intervals')
        save(fig,out,f'performance_auc_{metric}.png')
        if win_auc.empty:continue
        fig,axes=plt.subplots(1,len(c['normalizations']),figsize=(5*len(c['normalizations'])+3,7),sharey=True,squeeze=False,layout='constrained')
        controls=[f for f in c['families'] if f!='empirical'];y=np.arange(len(controls))
        for ax,norm in zip(axes[0],c['normalizations']):
            d=win_auc[(win_auc.normalization==norm)&(win_auc.metric==metric)].set_index('control').loc[controls]
            for pos,(family,row) in enumerate(d.iterrows()):
                ax.hlines(pos,row.ci_low,row.ci_high,color=COLORS[family]);ax.scatter(row.empirical_win_auc_percent,pos,color=COLORS[family],s=30)
            ax.axvline(50,color='.5',ls=':',lw=.7);ax.set(xlim=(-2,102),title=MODES[norm],xlabel='Empirical matched-setting win AUC (%)');ax.grid(axis='x',alpha=.15)
        axes[0,0].set(yticks=y,yticklabels=[LABELS[f] for f in controls]);axes[0,0].invert_yaxis()
        fig.suptitle(f'{metric}: empirical magnitudes versus each control; 95% paired-realization intervals')
        save(fig,out,f'win_fraction_auc_{metric}.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261119);a=p.parse_args(argv)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise FileExistsError('Use a new analysis folder')
    if a.bootstrap<2:raise ValueError('Require at least two bootstrap draws')
    m=run.load_run(a.run_dir);path=a.run_dir/'architecture_mc_narma.ALL.csv';meta=json.loads(Path(str(path)+'.json').read_text())
    if meta!=dict(experiment_id=m['experiment_id'],sha256=base.file_digest(path)):raise ValueError('Merged results changed')
    keys=['repeat_id','family','negative_fraction','normalization','rho_target','leak','input_scale']
    diagnostics=['raw_radius','scaling_factor','post_scaling_radius','raw_frobenius_norm','scaled_frobenius_norm','raw_magnitude_cv',
                 'raw_magnitude_variance','raw_mean_eigenvalue_magnitude','scaled_mean_eigenvalue_magnitude','inhibitory_magnitude_fraction',
                 'distribution_fit_shape','cap_cutoff_before_mean_restoration','mean_restoration_factor','fraction_below_empirical_min','fraction_above_empirical_max']
    data=pd.read_csv(path,usecols=keys+list(METRICS)+diagnostics+['baseline_scores_reused','scaling_scores_reused'])
    weights=np.random.default_rng(a.seed).multinomial(m['config']['repeats'],np.full(m['config']['repeats'],1/m['config']['repeats']),size=a.bootstrap)/m['config']['repeats']
    tables=summarize(data,m,weights);a.out_dir.mkdir(parents=True,exist_ok=True)
    for name,table in zip(['mean_performance','empirical_winning_fraction','configuration_advantages','performance_auc','win_fraction_auc'],tables):
        table.to_csv(a.out_dir/f'{name}.csv',index=False)
    data.groupby(['family','normalization','negative_fraction','rho_target'])[diagnostics].mean().reset_index().to_csv(a.out_dir/'weight_and_scaling_diagnostics.csv',index=False)
    reuse=data[['baseline_scores_reused','scaling_scores_reused']].value_counts().reset_index(name='rows');reuse.to_csv(a.out_dir/'score_reuse_counts.csv',index=False)
    plots(tables[0],tables[1],tables[3],tables[4],m['config'],a.out_dir)
    text=f'''# Paired ER weight-distribution experiment

{len(data):,} rows; {m['config']['repeats']} paired realizations; {len(base.grid(m))} hyperparameter settings;
{len(m['config']['fractions'])} inhibitory-edge fractions. All topology masks,
sign masks, input projections and task streams come from parent 271871.
MC and NARMA10 use its audited float64 scoring protocol. Higher MC is better;
lower NARMA10 NMSE is better. IPC is not part of this experiment.

Ten conditions: empirical, binary, half-normal; empirical caps at Q99, Q95 and
Q90 with mean restoration; gamma and lognormal matched to realized mean and
population variance; heterogeneity a=0.5 and 0.75. Synthetic controls use fixed
mid-quantile vectors from SciPy distributions and preserve empirical ranks,
with random tie-breaking paired across conditions and fractions. They are not
iid sampled distributions. Moment matching also matches raw squared-weight
energy; it does not match the entire shape or minimum weight.

Own-radius scaling fixes each current radius. Shared-CE scaling freezes the
unmodified parent's multiplier at each fraction across all weight conditions.
Frobenius matching fixes the norm to that of the shared-scaled, unmodified CE
reference; actual spectral radii can differ. A matched norm is a sum-of-squares
constraint, not a claim of matched dynamical gain. Equal-norm cases use an exact
shared multiplier within the documented 1e-10 relative tolerance.

Baseline scores are reused only for identical matrices, inputs and scaling;
identical-matrix scaling copies are checked and recorded separately. Raw
metadata record both reuse flags and source shard hashes. Original experiments
and outputs are preserved. Refitting synthetic distribution parameters does not
use task scores, validation scores or test targets.

Win curves compare empirical ER against each condition at identical settings,
averaging scores across realizations before counting strict wins. Performance
AUCs and win-fraction AUCs are distinct, normalized by the observed E/I span.
Trapezoidal interpolation connects observed E/I points. No hyperparameter
configuration is discarded. The chosen grid and uniform E/I weighting are
analysis choices, not biological sampling distributions.

Intervals resample whole paired realizations jointly across all conditions,
fractions, settings and tasks: {a.bootstrap} draws, seed {a.seed}. They are
pointwise descriptive percentile intervals with no multiplicity adjustment.
Inspect diagnostics for actual radii, applied factors, raw/scaled Frobenius
norms, weight variance, fitted shapes and changes in magnitude-weighted E/I.
'''
    (a.out_dir/'REPORT.md').write_text(text)
    base.write_json(a.out_dir/'manifest.json',dict(experiment_id=m['experiment_id'],source=str(path.resolve()),source_sha256=meta['sha256'],
        bootstrap=a.bootstrap,bootstrap_seed=a.seed,bootstrap_unit='paired whole realization',analysis_sha256=base.file_digest(Path(__file__)),
        outputs={f.name:base.file_digest(f) for f in a.out_dir.iterdir() if f.is_file()}))
    print(a.out_dir,flush=True)


if __name__=='__main__':main()
