#!/usr/bin/env python3
"""PNG/CSV reports for the frozen ten-model MC/NARMA experiment.

Bootstrap whole realizations jointly across conditions. Hyperparameter points
are a fixed design, never independent biological-network samples.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/architecture-sweep-xdg-cache')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from architecture_mc_narma import load_run, merge, digest, write_json, file_digest

PARAMETERS=['rho_target','leak','input_scale']
METRICS={'MC':1,'NARMA10_test_nmse':-1}


def metric_directions(data):
    return {name:direction for name,direction in dict(METRICS,IPC=1).items() if name in data.columns}

LABELS={'ce_empirical':'CE empirical','ce_shuffled':'CE shuffled magnitudes',
 'ce_degree_rewired':'CE degree rewired','ce_binary':'CE binary',
 'ce_halfnormal':'CE half-normal','er_signed_empirical':'ER exact CE signed weights',
 'er_binary':'ER binary','er_halfnormal':'ER half-normal',
 'ws_signed_empirical':'WS-like exact CE signed weights','ce_unknown_removed':'CE unknown edges removed'}
COLORS={'ce_empirical':'#d55e00','ce_unknown_removed':'#0072b2','er_binary':'#009e73',
 'ce_shuffled':'#e69f00','ce_degree_rewired':'#cc79a7','ce_binary':'#882255',
 'ce_halfnormal':'#aa4499','er_signed_empirical':'#44aa99','er_halfnormal':'#117733',
 'ws_signed_empirical':'#332288'}


def bootstrap_weights(n,count,seed):
    if n<1 or count<2: raise ValueError('Require observations and >=2 bootstrap replicates')
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=count)/n


def variance_components(surface):
    """Exact balanced-grid functional ANOVA, including interactions.

    Returns raw variance components. A constant surface has zero components;
    percentage shares are undefined and must be reported as missing.
    """
    levels=[np.unique(surface[p]) for p in PARAMETERS]
    expected=pd.MultiIndex.from_product(levels,names=PARAMETERS)
    if surface.duplicated(PARAMETERS).any() or len(surface)!=len(expected):
        raise ValueError('ANOVA requires a complete unique Cartesian grid')
    series=surface.set_index(PARAMETERS)['score'].reindex(expected)
    if not np.isfinite(series).all(): raise ValueError('Nonfinite/incomplete ANOVA surface')
    values=series.to_numpy().reshape([len(v) for v in levels])
    center=values.mean(); components={}; records=[]
    for size in range(1,4):
        for subset in itertools.combinations(range(3),size):
            other=tuple(i for i in range(3) if i not in subset)
            effect=(values.mean(axis=other,keepdims=True) if other else values.copy())-center
            for smaller,term in components.items():
                if set(smaller).issubset(subset): effect=effect-term
            components[subset]=effect
            records.append(dict(component=' × '.join(PARAMETERS[i] for i in subset),
                                dimensions=subset,variance=float(np.mean(effect**2))))
    total=float(np.var(values))
    if not np.isclose(sum(r['variance'] for r in records),total,rtol=1e-8,atol=1e-12):
        raise ValueError('Functional ANOVA does not reconstruct total variance')
    return records,total


def summarize(d,m,weights):
    records=[]; wins=[]; advantages=[]; decomposition=[]
    repeats=range(m['config']['repeats'])
    for (norm,fraction,family),g in d.groupby(['normalization','negative_fraction','family'],sort=False):
        for metric in metric_directions(d):
            values=g.groupby('repeat_id')[metric].mean().reindex(repeats).to_numpy()
            ci=np.quantile(weights@values,[.025,.975]) if len(values)>1 else [np.nan,np.nan]
            records.append(dict(normalization=norm,negative_fraction=fraction,family=family,metric=metric,
                mean=float(values.mean()),ci_low=ci[0],ci_high=ci[1],n_realizations=len(values),
                n_configurations=len(g)//len(values)))
            for repeat,rep in g.groupby('repeat_id'):
                components,total=variance_components(rep[PARAMETERS+[metric]].rename(columns={metric:'score'}))
                for comp in components:
                    decomposition.append(dict(normalization=norm,negative_fraction=fraction,family=family,
                        metric=metric,repeat_id=repeat,component=comp['component'],
                        variance=comp['variance'],total_variance=total,
                        variance_share=comp['variance']/total if total>0 else np.nan))
    for (norm,fraction),g in d.groupby(['normalization','negative_fraction'],sort=False):
        if 'ce_empirical' not in set(g.family): continue
        for metric,direction in metric_directions(d).items():
            ce=g[g.family=='ce_empirical'].pivot(index='repeat_id',columns=PARAMETERS,values=metric).reindex(repeats)
            for family in m['config']['families']:
                if family=='ce_empirical': continue
                control=g[g.family==family].pivot(index='repeat_id',columns=PARAMETERS,values=metric).reindex(index=ce.index,columns=ce.columns)
                delta=direction*(ce.to_numpy()-control.to_numpy())
                if not np.isfinite(delta).all(): raise ValueError('Incomplete paired comparison')
                means=delta.mean(axis=0)
                boot=weights@delta
                proportions=(boot>0).mean(axis=1)
                ci=np.quantile(proportions,[.025,.975]) if len(repeats)>1 else [np.nan,np.nan]
                wins.append(dict(normalization=norm,negative_fraction=fraction,control=family,metric=metric,
                    ce_winning_fraction=float((means>0).mean()),ties=int((means==0).sum()),
                    ci_low=ci[0],ci_high=ci[1],n_configurations=len(means),n_realizations=len(repeats)))
                for j,point in enumerate(ce.columns):
                    interval=np.quantile(boot[:,j],[.025,.975]) if len(repeats)>1 else [np.nan,np.nan]
                    advantages.append(dict(normalization=norm,negative_fraction=fraction,control=family,metric=metric,
                        **dict(zip(PARAMETERS,point)),ce_advantage=means[j],ci_low=interval[0],ci_high=interval[1]))
    return pd.DataFrame(records),pd.DataFrame(wins),pd.DataFrame(advantages),pd.DataFrame(decomposition)


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plots(summary,wins,advantages,components,out,m):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
    for norm in m['config']['normalizations']:
        for metric in summary.metric.unique():
            panel=summary[(summary.normalization==norm)&(summary.metric==metric)]
            fig,ax=plt.subplots(figsize=(9,5.8))
            for family in m['config']['families']:
                g=panel[panel.family==family].sort_values('negative_fraction')
                x=100*g.negative_fraction
                ax.plot(x,g['mean'],marker='o',ms=3,color=COLORS[family],label=LABELS[family])
                ax.fill_between(x,g.ci_low,g.ci_high,color=COLORS[family],alpha=.12)
            ylabel={'MC':'MC (higher is better)','NARMA10_test_nmse':'NARMA10 NMSE (lower is better)','IPC':'Held-out IPC estimate (higher is better)'}[metric]
            ax.set(xlabel='Inhibitory edges (%)',ylabel=ylabel,title=f'{norm}: mean performance over the grid')
            if metric!='IPC': ax.set_ylim(bottom=0)
            else: ax.axhline(0,color='.5',lw=.7)
            ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False)
            ax.grid(axis='y',alpha=.15)
            fig.text(.5,-.02,'Pointwise 95% whole-realization bootstrap intervals; equal weight per grid configuration.',ha='center',fontsize=8)
            save(fig,out,f'mean_{metric}_{norm}.png')
            if wins.empty: continue
            fig,ax=plt.subplots(figsize=(9,5.8))
            for family in m['config']['families']:
                if family=='ce_empirical': continue
                g=wins[(wins.normalization==norm)&(wins.metric==metric)&(wins.control==family)].sort_values('negative_fraction')
                ax.plot(100*g.negative_fraction,100*g.ce_winning_fraction,marker='o',ms=3,color=COLORS[family],label=LABELS[family])
                ax.fill_between(100*g.negative_fraction,100*g.ci_low,100*g.ci_high,color=COLORS[family],alpha=.1)
            ax.set(xlabel='Inhibitory edges (%)',ylabel='Matched hyperparameter settings CE wins (%)',
                   title=f'{metric} / {norm}: CE empirical versus each control',ylim=(-2,102))
            ax.axhline(50,color='.5',lw=.7,ls=':'); ax.grid(axis='y',alpha=.15)
            ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False)
            save(fig,out,f'winning_fraction_{metric}_{norm}.png')
            a=advantages[(advantages.normalization==norm)&(advantages.metric==metric)]
            controls=[f for f in m['config']['families'] if f!='ce_empirical']
            if not controls: continue
            limit=float(abs(a.ce_advantage).max()) or 1.
            fig,axes=plt.subplots(3,3,figsize=(12,10),squeeze=False,layout='constrained')
            for ax,family in zip(axes.flat,controls):
                matrix=a[a.control==family].groupby(['negative_fraction','input_scale']).ce_advantage.mean().unstack().sort_index()
                im=ax.imshow(matrix,aspect='auto',origin='lower',cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
                ax.set(title=LABELS[family],xlabel='Input scale',ylabel='Inhibitory edges (%)',
                    xticks=range(len(matrix.columns)),xticklabels=[f'{v:g}' for v in matrix.columns],
                    yticks=range(len(matrix.index)),yticklabels=[f'{100*v:g}' for v in matrix.index])
            for ax in list(axes.flat)[len(controls):]: ax.set_visible(False)
            fig.colorbar(im,ax=list(axes.flat),shrink=.6,label='Mean CE advantage (positive favors CE)')
            fig.suptitle(f'{metric} / {norm}; average over radius and leak; no configurations excluded')
            save(fig,out,f'input_advantage_{metric}_{norm}.png')
    shares=[]
    for keys,g in components.groupby(['normalization','negative_fraction','family','metric','repeat_id']):
        for parameter in PARAMETERS:
            share=sum(r.variance_share/len(r.component.split(' × ')) for r in g.itertuples() if parameter in r.component.split(' × '))
            shares.append(dict(zip(['normalization','negative_fraction','family','metric','repeat_id'],keys),parameter=parameter,share=share))
    pd.DataFrame(shares).to_csv(out/'hyperparameter_allocated_shares.csv',index=False)
    for norm in m['config']['normalizations']:
        metrics=list(summary.metric.unique())
        fig,axes=plt.subplots(1,len(metrics),figsize=(6*len(metrics),5.5),layout='constrained',squeeze=False)
        axes=axes[0]
        for ax,metric in zip(axes,metrics):
            g=pd.DataFrame(shares)
            table=g[(g.normalization==norm)&(g.metric==metric)].groupby(['family','parameter']).share.mean().unstack().reindex(m['config']['families'])
            bottom=np.zeros(len(table))
            for parameter,color in zip(PARAMETERS,['#0072b2','#e69f00','#009e73']):
                values=table[parameter].to_numpy()*100
                ax.barh(range(len(table)),values,left=bottom,label=parameter,color=color)
                bottom+=values
            ax.set(yticks=range(len(table)),yticklabels=[LABELS[f] for f in table.index],xlabel='Allocated performance variance (%)',title=metric,xlim=(0,100))
        axes[-1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
        fig.suptitle(f'{norm}: interaction variance divided equally among participating parameters')
        save(fig,out,f'variance_contributions_{norm}.png')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000); p.add_argument('--seed',type=int,default=20261119)
    args=p.parse_args(argv)
    m=load_run(args.run_dir); merge(argparse.Namespace(run_dir=args.run_dir))
    if args.out_dir.exists() and any(args.out_dir.iterdir()): raise FileExistsError('Use a new analysis directory; no overwrite')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    source=args.run_dir/'architecture_mc_narma.ALL.csv'
    d=pd.read_csv(source)
    weights=bootstrap_weights(m['config']['repeats'],args.bootstrap,args.seed)
    summary,wins,advantages,components=summarize(d,m,weights)
    for name,table in [('mean_performance',summary),('ce_winning_fraction',wins),('configuration_advantages',advantages),('hyperparameter_variance_components',components)]:
        table.to_csv(args.out_dir/f'{name}.csv',index=False)
    diagnostics=d.groupby(['normalization','family','negative_fraction']).agg(
        max_post_radius=('post_scaling_radius','max'),max_echo_MC=('MC_washout_initialization_rms_gap','max'),
        max_echo_NARMA=('NARMA10_washout_initialization_rms_gap','max'),max_NARMA_NMSE=('NARMA10_test_nmse','max'),
        mean_MC_tail=('MC_tail','mean')).reset_index()
    diagnostics.to_csv(args.out_dir/'dynamics_diagnostics.csv',index=False)
    streams=d[['repeat_id','narma_generator_seed','narma_rejected_streams']].drop_duplicates()
    streams.to_csv(args.out_dir/'narma_streams.csv',index=False)
    if 'IPC' in d:
        columns=['IPC','IPC_positive_part','IPC_linear','IPC_nonlinear','IPC_crossdelay','IPC_single_delay','IPC_mean_predictor_baseline','IPC_excess_over_mean','IPC_negative_terms']
        d.groupby(['normalization','family','negative_fraction'])[columns].mean().reset_index().to_csv(args.out_dir/'ipc_component_diagnostics.csv',index=False)
    plots(summary,wins,advantages,components,args.out_dir,m)
    report=f'''# Unified architecture sweep\n\nProtocol: `{m['protocol']}`. {len(d):,} paired observations; {m['config']['repeats']} realizations; {len(m['config']['rhos'])*len(m['config']['leaks'])*len(m['config']['inputs'])} grid settings.\n\nThe primary comparison is CE empirical versus each control at identical hyperparameters. MC is the sum of held-out squared correlations over delays 1–{task_delay(m)}; larger is better. NARMA10 NMSE is test MSE divided by test-target variance; smaller is better. Readout penalties use separate validation data. Mean curves average every grid setting; winning fractions compare realization-averaged scores at each configuration. They are different summaries.\n\nIntervals resample whole paired realizations jointly across models, fractions and configurations ({args.bootstrap} draws, seed {args.seed}); they are pointwise, descriptive intervals. The fixed grid is not a random hyperparameter population. One-realization intervals are undefined.\n\n`hyperparameter_variance_components.csv` contains exact balanced-grid main effects and interactions within each realization. The allocated-share plot splits each interaction equally across its participating parameters; that allocation is descriptive and depends on the grid and score scale. Constant surfaces have undefined percentage shares.\n\nInspect `configuration_advantages.csv` for the settings driving any crossover and `dynamics_diagnostics.csv` for initialization sensitivity, actual scaled radii, memory beyond the primary cutoff, and large errors. No poor-performing configurations were discarded. NARMA generator-only nonfinite streams were replaced by the prespecified recorded seed rule before network evaluation; `narma_streams.csv` records counts and accepted seeds, and task caches contain rejected seed lists.\n\nOwn-radius scaling equalizes current raw radius. Fixed-reference uses each family's radius at the known-edge CE reference fraction, held fixed across the sweep. Shared-CE scaling applies the CE multiplier at the current fraction to all families; it varies across fractions. Only the shared multiplier preserves exact signed-weight matching after scaling. The removed-edge CE control has fewer edges and is not density-matched to the full-CE ER controls. WS is a directed WS-like control. All edge signs are randomized quotas; these are reservoir controls, not inferred biological sign balances.\n\nThe protocol uses conventional NARMA10 and a new stabilized readout; it must not be silently pooled with archived NARMA results. Performance differences do not alone isolate eigenmode causality or establish biological optimality.\n'''
    if 'ipc' in m['config']:
        spec=m['config']['ipc']
        report+=f"\nIPC is the signed sum of held-out 1 - MSE/target-second-moment estimates for {spec['target_count']} orthonormal Legendre-product targets, lags 1–{spec['max_delay']}, total degree 1–{spec['max_degree']}. Cross-delay products are included. Higher is better; finite estimates may be negative and are retained. This is a finite-basis, regularized adaptation of Dambre et al. (2012), not full population IPC. IPC_positive_part is a separately named diagnostic with a positive clipping floor; it is not used for the primary win curves. Linear/nonlinear, cross-delay and mean-predictor diagnostics are included. IPC covers every fraction in this run with the same definition; do not substitute archived restricted IPC scores.\n"
    if 'reuse_parent_through' in m['config']:
        reused=int(d.MC_NARMA_reused.sum())
        report+=f"\nMC/NARMA scores in {reused:,} rows through {100*m['config']['reuse_parent_through']:g}% inhibition are reused from the recorded parent experiment. IPC was newly computed from regenerated, paired MC state trajectories for those rows. Original NARMA simulations and MC/NARMA readout fitting were skipped for reused rows. Row-level reuse flags, parent experiment IDs and parent-shard hashes preserve provenance.\n"
    report+='\nPrimary comparison: fraction of identical hyperparameter settings won at each E/I fraction, after averaging each setting over paired realizations. Grid-averaged performance is supporting context.\n'
    (args.out_dir/'REPORT.md').write_text(report)
    write_json(args.out_dir/'manifest.json',dict(experiment_id=m['experiment_id'],source=str(source.resolve()),
        source_sha256=file_digest(source),analysis_sha256=digest(Path(__file__).read_bytes()),
        bootstrap=args.bootstrap,bootstrap_seed=args.seed,bootstrap_unit='paired whole realization',
        outputs={f.name:digest(f.read_bytes()) for f in args.out_dir.iterdir() if f.is_file()}))
    print(args.out_dir)


def task_delay(m): return m['config']['tasks']['mc_max_delay']

if __name__=='__main__': main()
