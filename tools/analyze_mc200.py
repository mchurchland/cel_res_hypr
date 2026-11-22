#!/usr/bin/env python3
"""MC200 and archived NARMA analysis, preserving paired realization structure."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-mc200-analysis')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/mc200-analysis-cache')
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from tools.analyze_architecture_mc_narma import LABELS as ARCH_LABELS, COLORS as ARCH_COLORS
from network_stats.er_weight_distributions import LABELS as ER_LABELS, COLORS as ER_COLORS
from util.ei_auc import curve_areas, win_fraction_auc

KEYS = ['repeat_id', 'family', 'normalization', 'negative_fraction', 'rho_target', 'leak', 'input_scale']
METRICS = ['MC_30', 'MC_60', 'MC_100', 'MC_150', 'MC_200', 'MC_last25', 'archived_MC30', 'archived_MC60', 'NARMA10_test_nmse']
PRIMARY = {'MC_200': 1, 'NARMA10_test_nmse': -1}
NAMES = {'MC_200': 'MC200', 'NARMA10_test_nmse': 'NARMA10'}
MODES = {'own_radius': 'Own radius', 'fixed_reference': 'Fixed reference', 'shared_ce_scale': 'Shared CE multiplier', 'frobenius_match': 'Matched Frobenius norm'}
PARAMETERS = ['Spectral-radius target', 'Leak rate', 'Input scale']
INPUT_COLORS = ['#332288', '#0072B2', '#E69F00', '#CC3311']
ARCH_COLORS = dict(ARCH_COLORS, er_binary='#0072B2', er_halfnormal='#D18B00', er_signed_empirical='#7A3E8E')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def interval(values, weights):
    values = np.asarray(values)
    boot = (weights@values.reshape(len(values), -1)).reshape(len(weights), *values.shape[1:])
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    return values.mean(axis=0), lo, hi


def validate_cube(data, config, columns=METRICS):
    levels = [range(config['repeats']), config['families'], config['normalizations'], config['fractions'],
              config['rhos'], config['leaks'], config['inputs']]
    shape = tuple(map(len, levels))
    if len(data) != np.prod(shape) or data.duplicated(KEYS).any():
        raise ValueError('Incomplete or duplicated paired grid')
    codes = [pd.Categorical(data[k], categories=l).codes for k, l in zip(KEYS, levels)]
    if any((v < 0).any() for v in codes):
        raise ValueError('Unexpected grid coordinates')
    flat = np.empty((np.prod(shape), len(columns)))
    flat[np.ravel_multi_index(codes, shape)] = data[columns].to_numpy()
    if not np.isfinite(flat).all():
        raise ValueError('Missing/nonfinite scores; no filtering allowed')
    return flat.reshape(*shape[:4], np.prod(shape[4:]), len(columns))


def anova(values):
    """Balanced-grid ANOVA on the final three axes, preserving leading axes."""
    values = np.asarray(values, float)
    if values.ndim < 3 or not np.isfinite(values).all():
        raise ValueError('Finite complete three-dimensional grids required')
    axes = tuple(range(values.ndim-3, values.ndim))
    center = values.mean(axis=axes, keepdims=True)
    effects = {}; variances = []
    subsets = [s for size in (1, 2, 3) for s in itertools.combinations(range(3), size)]
    for subset in subsets:
        other = tuple(axes[i] for i in range(3) if i not in subset)
        effect = (values.mean(axis=other, keepdims=True) if other else values.copy())-center
        for smaller, term in effects.items():
            if set(smaller).issubset(subset):
                effect = effect-term
        effects[subset] = effect
        variances.append(np.mean(effect**2, axis=axes))
    total = np.var(values, axis=axes)
    components = np.stack(variances, axis=-1)
    components = np.where(total[..., None] > 0, components, 0.)
    np.testing.assert_allclose(components.sum(axis=-1), total, rtol=1e-8, atol=1e-12)
    return subsets, components, total


def load_data(run, out):
    manifest = json.loads((run/'manifest.json').read_text())
    complete = json.loads((run/'COMPLETE.json').read_text())
    source = run/'mc200.ALL.csv.gz'
    if sha(source) != complete['sha256'] or complete['experiment_id'] != manifest['experiment_id']:
        raise ValueError('MC200 completion checksum differs')
    use = ['suite', *KEYS, *METRICS[:6]]
    new = pd.read_csv(source, usecols=use, dtype={'suite':'category','family':'category','normalization':'category'})
    if len(new) != manifest['expected_rows']:
        raise ValueError('Wrong merged MC200 row count')
    provenance = [dict(path=str(source.resolve()), sha256=complete['sha256'], rows=len(new), kind='new MC200')]
    frames = []
    for suite, record in manifest['parents'].items():
        path = ROOT/Path(record['path']).relative_to('/home/milesc/Thesis')
        if sha(path/'manifest.json') != record['manifest_sha256']:
            raise ValueError(f'Original manifest differs: {suite}')
        oldpath = path/'architecture_mc_narma.ALL.csv'
        sidecar = json.loads(Path(str(oldpath)+'.json').read_text())
        digest = sha(oldpath)
        if sidecar['experiment_id'] != record['experiment_id'] or sidecar['sha256'] != digest:
            raise ValueError(f'Original scores differ: {suite}')
        old = pd.read_csv(oldpath, usecols=KEYS+['MC', 'MC_extended', 'NARMA10_test_nmse'],
                          dtype={'family':'category','normalization':'category'})
        old = old.rename(columns={'MC':'archived_MC30', 'MC_extended':'archived_MC60'})
        selected = new[new.suite==suite].copy()
        if len(old) != len(selected) or old.duplicated(KEYS).any():
            raise ValueError(f'Original/new pairing differs: {suite}')
        joined = selected.merge(old, on=KEYS, how='outer', validate='one_to_one', indicator=True)
        if not (joined['_merge']=='both').all():
            raise ValueError('Unmatched original/new configurations')
        joined = joined.drop(columns='_merge')
        joined['suite'] = 'architecture' if suite=='ws' else suite
        frames.append(joined)
        provenance.append(dict(path=str(oldpath.resolve()), sha256=digest, rows=len(old), kind=f'archived {suite} MC30/60 and NARMA10'))
        print(f'Paired {suite}: {len(joined):,} rows', flush=True)
    all_data = pd.concat(frames, ignore_index=True)
    configs = {}; cubes = {}
    for suite in ('architecture', 'er'):
        config = dict(manifest['parents'][suite]['config'])
        if suite=='architecture':
            config['families'] = config['families']+manifest['parents']['ws']['config']['families']
        configs[suite] = config
        cubes[suite] = validate_cube(all_data[all_data.suite==suite], config)
        np.savez_compressed(out/'cache'/f'{suite}_scores.npz', scores=cubes[suite], metrics=np.array(METRICS),
                            config=np.array(json.dumps(config)), sources=np.array(json.dumps(provenance)))
    return manifest, configs, cubes, provenance


def summarize(cube, config, weights, baseline):
    x = np.array(config['fractions']); span = x[-1]-x[0]
    if span <= 0 or not (np.diff(x)>0).all():
        raise ValueError('Need strictly increasing fractions')
    repeats = config['repeats']; points = list(itertools.product(config['rhos'], config['leaks'], config['inputs']))
    counts = np.rint(weights*repeats).astype(int)
    means = []; aucs = []; individual = []; wins = []; win_aucs = []; advantages = []; pairs = []; variance = []
    bi = config['families'].index(baseline)
    for ni, norm in enumerate(config['normalizations']):
        for metric, direction in PRIMARY.items():
            mi = METRICS.index(metric)
            for fi, family in enumerate(config['families']):
                surface = cube[:, fi, ni, :, :, mi]
                for level in ['ALL', *config['inputs']]:
                    ids = list(range(len(points))) if level=='ALL' else [j for j,p in enumerate(points) if p[2]==level]
                    curve = surface[..., ids].mean(axis=-1)
                    mu, lo, hi = interval(curve, weights)
                    for j, fraction in enumerate(x):
                        means.append(dict(family=family, normalization=norm, metric=metric, input_scale=level,
                                          negative_fraction=fraction, mean=mu[j], ci_low=lo[j], ci_high=hi[j]))
                    areas = curve_areas(curve, x)[0]/span
                    mu, lo, hi = interval(areas, weights)
                    aucs.append(dict(family=family, normalization=norm, metric=metric, input_scale=level,
                                     performance_auc=float(mu), ci_low=float(lo), ci_high=float(hi)))
                    if level=='ALL':
                        individual.extend(dict(repeat_id=r, family=family, normalization=norm, metric=metric, performance_auc=v) for r,v in enumerate(areas))
                grid = surface.reshape(repeats, len(x), len(config['rhos']), len(config['leaks']), len(config['inputs']))
                subsets, components, total = anova(grid)
                pooled_total = total.sum()
                allocated = np.zeros(3); equal_allocated = np.zeros((repeats, len(x), 3))
                for j, subset in enumerate(subsets):
                    value = components[..., j].sum()/pooled_total if pooled_total>0 else np.nan
                    for p in subset:
                        allocated[p] += value/len(subset)
                        equal_allocated[..., p] += np.divide(components[...,j], total, out=np.full_like(total,np.nan), where=total>0)/len(subset)
                    variance.append(dict(family=family, normalization=norm, metric=metric, component=' × '.join(PARAMETERS[p] for p in subset),
                                         variance_share=value, raw_variance_mean=components[...,j].mean(), kind='main effect or interaction'))
                for p in range(3):
                    variance.append(dict(family=family, normalization=norm, metric=metric, component=PARAMETERS[p], variance_share=allocated[p],
                                         equal_surface_mean_share=np.nanmean(equal_allocated[...,p]), kind='allocated including interactions'))
                if fi==bi:
                    continue
                delta = direction*(cube[:,bi,ni,:,:,mi]-surface)
                boot = (weights@delta.reshape(repeats,-1)).reshape(len(weights),len(x),len(points))
                observed = delta.mean(axis=0)
                for j, fraction in enumerate(x):
                    for k, point in enumerate(points):
                        ci = np.quantile(boot[:,j,k],[.025,.975])
                        advantages.append(dict(reference=baseline,control=family,normalization=norm,metric=metric,negative_fraction=fraction,
                            rho_target=point[0],leak=point[1],input_scale=point[2],reference_advantage=observed[j,k],ci_low=ci[0],ci_high=ci[1]))
                for level in ['ALL', *config['inputs']]:
                    ids = list(range(len(points))) if level=='ALL' else [j for j,p in enumerate(points) if p[2]==level]
                    observed_wins = (observed[:,ids]>0).mean(axis=-1)
                    boot_wins = (boot[...,ids]>0).mean(axis=-1)
                    ci = np.quantile(boot_wins,[.025,.975],axis=0)
                    strong = (np.quantile(boot[...,ids],.025,axis=0)>0).mean(axis=-1)
                    for j, fraction in enumerate(x):
                        wins.append(dict(reference=baseline,control=family,normalization=norm,metric=metric,input_scale=level,
                            negative_fraction=fraction,winning_fraction=observed_wins[j],ci_low=ci[0,j],ci_high=ci[1,j],
                            positive_interval_fraction=strong[j],ties=int((observed[j,ids]==0).sum()),n_configurations=len(ids)))
                    result = win_fraction_auc(delta[...,ids],counts,x)
                    win_aucs.append(dict(reference=baseline,control=family,normalization=norm,metric=metric,input_scale=level,
                        reference_win_auc_percent=result['ce_win_auc_percent']/span,ci_low=result['ce_ci_low']/span,ci_high=result['ce_ci_high']/span,
                        control_win_auc_percent=result['er_win_auc_percent']/span,tie_auc_percent=result['tie_win_auc_percent']/span))
                    seed_gain = curve_areas(delta[...,ids].mean(axis=-1),x)[0]/span
                    mu,lo,hi = interval(seed_gain,weights)
                    pairs.append(dict(reference=baseline,control=family,normalization=norm,metric=metric,input_scale=level,
                                      reference_advantage_auc=float(mu),ci_low=float(lo),ci_high=float(hi)))
            print(f'Summarized {baseline}: {norm} / {metric}',flush=True)
    tables = [means,aucs,individual,wins,win_aucs,advantages,pairs,variance]
    names = ['mean_performance','performance_auc','performance_auc_per_realization','winning_fraction','win_fraction_auc','configuration_advantages','paired_performance_auc_advantages','hyperparameter_variance']
    return dict(zip(names,map(pd.DataFrame,tables)))


def horizon_tables(cube, config, weights):
    rows = []; tails = []; changes = []
    x = np.array(config['fractions']); span = x[-1]-x[0]
    for fi,family in enumerate(config['families']):
        for ni,norm in enumerate(config['normalizations']):
            for metric in METRICS[:5]+['archived_MC30','archived_MC60']:
                areas = curve_areas(cube[:,fi,ni,:,:,METRICS.index(metric)].mean(axis=-1),x)[0]/span
                mu,lo,hi = interval(areas,weights)
                rows.append(dict(family=family,normalization=norm,metric=metric,performance_auc=float(mu),ci_low=float(lo),ci_high=float(hi)))
            for comparison,a,b in [('extra delays 31–200','MC_200','MC_30'),('test-length change at delay 30','MC_30','archived_MC30')]:
                delta = cube[:,fi,ni,:,:,METRICS.index(a)]-cube[:,fi,ni,:,:,METRICS.index(b)]
                areas = curve_areas(delta.mean(axis=-1),x)[0]/span
                mu,lo,hi = interval(areas,weights)
                changes.append(dict(family=family,normalization=norm,comparison=comparison,change_auc=float(mu),ci_low=float(lo),ci_high=float(hi)))
            surface = cube[:,fi,ni,:,:,METRICS.index('MC_last25')]
            avg = surface.mean(axis=0); idx = np.unravel_index(np.argmax(avg),avg.shape)
            tails.append(dict(family=family,normalization=norm,mean_last25=float(surface.mean()),
                maximum_configuration_seed_mean_last25=float(avg.max()),configuration_fraction_above_005=float((avg>.05).mean()),
                maximum_location_fraction=x[idx[0]],maximum_location_setting_index=int(idx[1])))
    h = pd.DataFrame(rows)
    h['rank'] = h.groupby(['normalization','metric']).performance_auc.rank(ascending=False,method='min')
    return h,pd.DataFrame(changes),pd.DataFrame(tails)


def save(fig, path):
    fig.savefig(path,dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def figure_style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':12,'legend.fontsize':8,'figure.titlesize':15})


def overview_plots(tables, config, out, suite):
    labels,colors = (ARCH_LABELS,ARCH_COLORS) if suite=='architecture' else (ER_LABELS,ER_COLORS)
    ref = 'CE empirical' if suite=='architecture' else 'Empirical CE weights on ER'
    families = config['families']; controls = [f for f in families if f!=('ce_empirical' if suite=='architecture' else 'empirical')]
    for metric in PRIMARY:
        label = NAMES[metric]
        for norm in config['normalizations']:
            fig,ax = plt.subplots(figsize=(10.5,6.5))
            for j,family in enumerate(families):
                g=tables['mean_performance'].query('metric==@metric and normalization==@norm and input_scale=="ALL" and family==@family').sort_values('negative_fraction')
                ax.plot(g.negative_fraction*100,g['mean'],label=labels[family],color=colors[family],marker=['o','s','^','D'][j%4],ms=3,lw=1.6,ls='-' if j<6 else '--')
                ax.fill_between(g.negative_fraction*100,g.ci_low,g.ci_high,color=colors[family],alpha=.09)
            ax.set(xlabel='Inhibitory edges (%)',ylabel='MC200 (higher is better)' if metric=='MC_200' else 'NARMA10 NMSE (lower is better)',
                   title=f'{label}: mean over all 24 settings — {MODES[norm]}',xticks=np.arange(0,101,10),ylim=(0,None))
            ax.grid(axis='y',alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
            save(fig,out/f'mean_{label}_{norm}.png')
            fig,axes=plt.subplots(4,3,figsize=(13,12),sharex=True,sharey=True)
            for ax,family in zip(axes.flat,controls):
                for level,color in zip(['ALL',*config['inputs']],['#222222',*INPUT_COLORS]):
                    g=tables['winning_fraction'].query('metric==@metric and normalization==@norm and input_scale==@level and control==@family').sort_values('negative_fraction')
                    ax.plot(g.negative_fraction*100,g.winning_fraction*100,color=color,lw=1.8 if level=='ALL' else 1.1,marker='o',ms=2,label='All 24 settings' if level=='ALL' else f'Input {level:g}')
                    if level=='ALL':ax.fill_between(g.negative_fraction*100,g.ci_low*100,g.ci_high*100,color=color,alpha=.12)
                ax.axhline(50,color='.6',ls=':',lw=.7);ax.set(title=labels[family],ylim=(-3,103),xticks=[0,20,40,60,80,100]);ax.grid(axis='y',alpha=.12)
            for ax in list(axes.flat)[len(controls):]:ax.set_visible(False)
            fig.supxlabel('Inhibitory edges (%)');fig.supylabel(f'Matched settings {ref} wins (%)')
            fig.suptitle(f'{label}: {ref} versus each control — {MODES[norm]}')
            fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.02))
            fig.tight_layout(rect=(.025,.035,1,.965));save(fig,out/f'winning_fraction_{label}_{norm}.png')
        for table,col,lo,hi,prefix,xlabel,selection in [
            (tables['performance_auc'],'performance_auc','ci_low','ci_high','performance_auc', 'Mean MC200 across E/I sweep (higher is better)' if metric=='MC_200' else 'Mean NARMA10 error across E/I sweep (lower is better)',families),
            (tables['win_fraction_auc'],'reference_win_auc_percent','ci_low','ci_high','win_fraction_auc',f'Matched-setting win AUC for {ref} (%)',controls)]:
            fig,axes=plt.subplots(1,len(config['normalizations']),figsize=(17,7.2),sharey=True,layout='constrained')
            for ax,norm in zip(axes,config['normalizations']):
                name='family' if prefix=='performance_auc' else 'control'
                g=table.query('metric==@metric and normalization==@norm and input_scale=="ALL"').set_index(name).loc[selection]
                for j,(family,row) in enumerate(g.iterrows()):
                    ax.hlines(j,row[lo],row[hi],color=colors[family],lw=1.5);ax.scatter(row[col],j,color=colors[family],s=35,marker='D' if family==selection[0] else 'o')
                ax.set(title=MODES[norm],xlabel=xlabel);ax.grid(axis='x',alpha=.15)
                if prefix=='win_fraction_auc':ax.set_xlim(-2,102);ax.axvline(50,color='.5',ls=':',lw=.8)
                else:
                    bounds=g[[lo,hi]].to_numpy();low,high=float(bounds.min()),float(bounds.max());pad=max((high-low)*.12,.001)
                    ax.set_xlim(max(0,low-pad),high+pad)
            axes[0].set(yticks=range(len(selection)),yticklabels=[labels[f] for f in selection]);axes[0].invert_yaxis()
            fig.suptitle(f'{label}: '+('AUC of mean-performance curves' if prefix=='performance_auc' else 'AUC of matched-setting win curves')+'\n95% paired-realization bootstrap intervals')
            save(fig,out/f'{prefix}_{label}.png')


def input_plots(tables, config, out, suite):
    labels,colors = (ARCH_LABELS,ARCH_COLORS) if suite=='architecture' else (ER_LABELS,ER_COLORS)
    chosen = ['ce_empirical','er_signed_empirical','er_binary','er_halfnormal','ws_signed_empirical','ws_binary'] if suite=='architecture' else ['empirical','binary','halfnormal','cap90','gamma','lognormal','hetero50']
    for norm in config['normalizations']:
        fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey='row')
        for row,metric in enumerate(PRIMARY):
            for ax,inp in zip(axes[row],config['inputs']):
                for family in chosen:
                    g=tables['mean_performance'].query('metric==@metric and normalization==@norm and input_scale==@inp and family==@family').sort_values('negative_fraction')
                    ax.plot(g.negative_fraction*100,g['mean'],color=colors[family],label=labels[family],lw=1.7)
                ax.set(title=f'Input scale {inp:g}',xticks=[0,20,40,60,80,100]);ax.grid(axis='y',alpha=.13)
                if row==0:ax.set_ylim(bottom=0)
                if row==1:ax.set_xlabel('Inhibitory edges (%)')
            axes[row,0].set_ylabel('MC200' if metric=='MC_200' else 'NARMA10 NMSE')
        fig.suptitle(f'Performance by input strength — {MODES[norm]}')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.025))
        fig.tight_layout(rect=(0,.07,1,.96));save(fig,out/f'performance_by_input_{norm}.png')


def variance_plots(table, config, out, suite):
    labels=ARCH_LABELS if suite=='architecture' else ER_LABELS
    for norm in config['normalizations']:
        fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True,layout='constrained')
        for ax,metric in zip(axes,PRIMARY):
            g=table.query('kind=="allocated including interactions" and normalization==@norm and metric==@metric')
            values=g.pivot(index='family',columns='component',values='variance_share').loc[config['families'],PARAMETERS]*100
            left=np.zeros(len(values))
            for parameter,color in zip(PARAMETERS,['#0072B2','#E69F00','#009E73']):
                v=values[parameter].to_numpy();ax.barh(np.arange(len(v)),v,left=left,color=color,label=parameter);left+=v
            ax.set(title=NAMES[metric],xlim=(0,100),xlabel='Allocated share of within-grid variance (%)',yticks=range(len(values)),yticklabels=[labels[f] for f in values.index])
        axes[0].invert_yaxis();fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f'Hyperparameter contributions — {MODES[norm]}\nInteractions split equally; component variances summed over realizations and E/I fractions')
        save(fig,out/f'hyperparameter_variance_{norm}.png')


def horizon_plots(table,config,out,suite):
    labels,colors=(ARCH_LABELS,ARCH_COLORS) if suite=='architecture' else (ER_LABELS,ER_COLORS)
    fig,axes=plt.subplots(1,3,figsize=(17,5.8),layout='constrained')
    for ax,norm in zip(axes,config['normalizations']):
        for family in config['families']:
            g=table[(table.family==family)&(table.normalization==norm)&table.metric.str.startswith('MC_')].copy()
            g['horizon']=g.metric.str[3:].astype(int);g=g.sort_values('horizon')
            ax.plot(g.horizon,g.performance_auc,color=colors[family],marker='o',ms=3,label=labels[family])
        ax.set(title=MODES[norm],xlabel='Summation horizon (input delays)',ylabel='MC AUC across E/I sweep',xticks=[30,60,100,150,200]);ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Effect of the MC cutoff: all horizons use the same 5,000 test samples')
    save(fig,out/'horizon_dependence.png')


def gamma_plot(tables,out):
    pair=tables['paired_performance_auc_advantages'];wins=tables['win_fraction_auc']
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    records=[]
    for ni,norm in enumerate(['own_radius','shared_ce_scale','frobenius_match']):
        for row,metric in enumerate(PRIMARY):
            g=tables['configuration_advantages'].query('control=="gamma" and normalization==@norm and metric==@metric')
            # Mean curves use separate paired differences saved at each setting.
            curve=g.groupby('negative_fraction').reference_advantage.mean()
            ax=axes[row,ni];ax.axhline(0,color='.4',lw=.8)
            ax.plot(curve.index*100,-curve.to_numpy(),color='#332288',marker='o',ms=4)
            ax.set(title=MODES[norm],xlabel='Inhibitory edges (%)',ylabel='Gamma MC200 gain' if metric=='MC_200' else 'Gamma NMSE reduction');ax.grid(alpha=.15)
            a=pair.query('control=="gamma" and normalization==@norm and metric==@metric and input_scale=="ALL"').iloc[0]
            w=wins.query('control=="gamma" and normalization==@norm and metric==@metric and input_scale=="ALL"').iloc[0]
            ax.text(.03,.96,f'AUC gain: {-a.reference_advantage_auc:.4g}\n95% CI [{-a.ci_high:.4g}, {-a.ci_low:.4g}]\nGamma win AUC: {w.control_win_auc_percent:.1f}%',transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
            records.append(dict(normalization=norm,metric=metric,gamma_advantage_auc=-a.reference_advantage_auc,ci_low=-a.ci_high,ci_high=-a.ci_low,gamma_win_auc_percent=w.control_win_auc_percent))
    fig.suptitle('Moment-matched gamma versus empirical CE magnitudes on identical ER wiring\nPositive favors gamma; all 24 settings included')
    save(fig,out/'gamma_vs_empirical.png')
    pd.DataFrame(records).to_csv(out/'gamma_summary.csv',index=False)


def delay_profiles(run, m, out, weights):
    models=[('architecture','ce_empirical'),('architecture','ce_shuffled'),('architecture','er_signed_empirical'),('architecture','er_binary'),
            ('architecture','ws_signed_empirical'),('ws','ws_binary'),('er','gamma'),('er','lognormal')]
    config=m['parents']['architecture']['config'];points=list(itertools.product(config['rhos'],config['leaks'],config['inputs']))
    cases=list(itertools.product([.2,.5],[.95,1.05],[.6,1.]))
    profiles=np.empty((len(models),len(cases),config['repeats'],200))
    jobs=list(itertools.product(range(len(models)),range(config['repeats']),[.2,.5]))
    def read(job):
        mi,repeat,fraction=job;suite,family=models[mi];fi=config['fractions'].index(fraction)
        path=run/'checkpoints'/f'r{repeat:04d}__{suite}__{family}__own_radius__ei{fi:02d}.npz'
        with np.load(path,allow_pickle=False) as z:curves=z['curves']
        for ci,case in enumerate(cases):
            if case[0]==fraction:profiles[mi,ci,repeat]=curves[points.index((case[1],case[2],.01))]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(read,jobs))
    rows=[];summaries=[]
    for mi,(suite,family) in enumerate(models):
        for ci,(fraction,rho,leak) in enumerate(cases):
            mean,lo,hi=interval(profiles[mi,ci],weights)
            for k in range(200):
                rows.append(dict(family=family,suite=suite,negative_fraction=fraction,rho_target=rho,leak=leak,input_scale=.01,delay=k+1,mean=mean[k],ci_low=lo[k],ci_high=hi[k]))
            summaries.append(dict(family=family,suite=suite,negative_fraction=fraction,rho_target=rho,leak=leak,input_scale=.01,
                MC200=float(mean.sum()),MC_last25=float(mean[-25:].sum()),last_lag=float(mean[-1]),last25_per_lag=float(mean[-25:].mean())))
    d=pd.DataFrame(rows);d.to_csv(out/'selected_delay_profiles.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'selected_horizon_diagnostics.csv',index=False)
    np.savez_compressed(out/'cache'/'selected_delay_profiles.npz',profiles=profiles,cases=np.asarray(cases),models=np.asarray(models))
    def color(f):return ER_COLORS[f] if f in ['gamma','lognormal'] else ARCH_COLORS[f]
    def label(f):return ('ER '+ER_LABELS[f]) if f in ['gamma','lognormal'] else ARCH_LABELS[f]
    for detailed in (False,True):
        selected=cases if detailed else [(.2,.95,1.),(.5,.95,1.)]
        fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True) if detailed else plt.subplots(1,2,figsize=(13,5),sharey=True)
        for ax,case in zip(axes.flat,selected):
            for suite,family in models:
                g=d[(d.family==family)&(d.negative_fraction==case[0])&(d.rho_target==case[1])&(d.leak==case[2])]
                ax.plot(g.delay,g['mean'],color=color(family),label=label(family),lw=1.6)
                if not detailed:ax.fill_between(g.delay,g.ci_low,g.ci_high,color=color(family),alpha=.08)
            ax.axvline(30,color='.5',ls=':',lw=1);ax.set(title=f'{100*case[0]:g}% inhibitory; radius {case[1]:g}; leak {case[2]:g}',xlabel='Input delay',ylim=(-.02,1.03),xlim=(1,200));ax.grid(alpha=.12)
        axes.flat[0].set_ylabel('Held-out recall r²')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.04))
        fig.suptitle('Memory by delay: own-radius scaling, input scale 0.01\nDotted line marks the archived 30-delay cutoff')
        fig.tight_layout(rect=(0,.09,1,.92));save(fig,out/('memory_delay_profiles_all_settings.png' if detailed else 'memory_delay_profiles.png'))
    print('Selected delay profiles complete',flush=True)


def write_report(out, manifest, configs, all_tables, all_horizons, all_tails, sources, args):
    lines=['# MC200 reanalysis','',
        'The MC200 run contains 1,742,400 logical rows, 100 paired realizations, 11 inhibitory-edge fractions, and 24 hyperparameter settings. '
        'Architecture comparisons include 12 variants; ER weight comparisons include 10 distributions. NARMA10 is joined from the archived, identically paired runs and was not rerun.','',
        '## Mean-performance AUC rankings','',
        'AUC integrates the grid-mean curve over the observed 0–100% sweep, divided by its width. It retains performance units. '
        'Win AUC instead integrates the percentage of settings won after averaging paired realization scores. They answer different questions.','']
    for suite,tables in all_tables.items():
        labels=ARCH_LABELS if suite=='architecture' else ER_LABELS
        lines += [f'### {suite.capitalize()}: own-radius scaling','','| Model | MC200 AUC [95% CI] | NARMA10 error AUC [95% CI] |','|---|---:|---:|']
        a=tables['performance_auc'].query('normalization=="own_radius" and input_scale=="ALL"')
        ranking=a.query('metric=="MC_200"').sort_values('performance_auc',ascending=False).family
        for family in ranking:
            rows=[a.query('family==@family and metric==@metric').iloc[0] for metric in PRIMARY]
            lines.append('| '+labels[family]+' | '+' | '.join(f'{r.performance_auc:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]' for r in rows)+' |')
        lines += ['',f'[Mean curves and input effects]({suite}/performance_by_input_own_radius.png) · [Performance AUC]({suite}/performance_auc_MC200.png) · [Win AUC]({suite}/win_fraction_auc_MC200.png)','',
                  f'### {suite.capitalize()}: matched-setting wins','','| Comparator | Reference MC200 win AUC | Reference NARMA10 win AUC |','|---|---:|---:|']
        w=tables['win_fraction_auc'].query('normalization=="own_radius" and input_scale=="ALL"')
        for family in w.control.unique():
            vals=[w.query('control==@family and metric==@metric').iloc[0] for metric in PRIMARY]
            lines.append('| '+labels[family]+' | '+' | '.join(f'{r.reference_win_auc_percent:.1f}% [{r.ci_low:.1f}, {r.ci_high:.1f}]' for r in vals)+' |')
        lines += ['', 'Reference = '+('CE empirical' if suite=='architecture' else 'empirical CE magnitudes placed on ER')+'. Every strict positive difference counts, even a very small one. '
                  'The tables also record the fraction of settings with a positive lower bootstrap interval; this is a descriptive uncertainty screen, not a multiplicity-adjusted significance claim.','']
    lines += ['## What extending the MC horizon changes','',
              'All new horizons (30, 60, 100, 150, 200) use the same 5,000-sample test segment. Comparing them isolates the effect of summing additional delays within this estimator. '
              'The archived 30/60-delay scores used only 500 test samples. Their differences from the new 30/60 sums also reflect test length and sampling variation. '
              'New readouts keep the original training, validation, projections, matrices, and exact scaling factors.','',
              '| Suite / model | Archived MC30 AUC | New MC30 AUC | MC200 AUC | Rank at new MC30 → MC200 |','|---|---:|---:|---:|---:|']
    for suite,h in all_horizons.items():
        labels=ARCH_LABELS if suite=='architecture' else ER_LABELS
        g=h[h.normalization=='own_radius']
        for family in configs[suite]['families']:
            d=g[g.family==family].set_index('metric')
            lines.append(f'| {suite}: {labels[family]} | {d.loc["archived_MC30","performance_auc"]:.3f} | {d.loc["MC_30","performance_auc"]:.3f} | {d.loc["MC_200","performance_auc"]:.3f} | {int(d.loc["MC_30","rank"])} → {int(d.loc["MC_200","rank"])} |')
    lines += ['', '[Architecture horizon comparison](architecture/horizon_dependence.png) · [ER distribution horizon comparison](er/horizon_dependence.png) · [Selected memory curves](memory_delay_profiles.png)','',
              '## Hyperparameter dependence','',
              'Exact balanced-grid functional ANOVA decomposes performance variance within each realization, family, scaling branch and E/I fraction. '
              'The main plots pool raw component variances across those surfaces, then split interactions equally among their participating parameters. '
              'CSV tables additionally retain the equal-surface average of allocated percentage shares; it is a different weighting. '
              'These allocations depend on the selected parameter levels and performance scale and are descriptive, not causal effects.','',
              '[Architecture contributions](architecture/hyperparameter_variance_own_radius.png) · [ER contributions](er/hyperparameter_variance_own_radius.png)','',
              '## Horizon and protocol limitations','',
              '- MC200 remains a finite-delay estimate. Late-delay diagnostics are retained across the full grid; the selected profiles emphasize low-input long-memory regimes.',
              '- `horizon_tail_diagnostics.csv` records recall summed over delays 176–200. A seed-mean block score above 0.05 is an exploratory screen, not a validated null threshold or proof that the horizon is inadequate.',
              '- Squared correlations have positive finite-test chance contributions. No null subtraction or clipping was applied to the reported MC sums.',
              '- Intervals resample whole paired realizations jointly across all fractions and configurations: '+str(args.bootstrap)+' bootstrap replicates, seed '+str(args.seed)+'. Hyperparameter settings are a fixed design, not independent biological samples. Intervals are pointwise and unadjusted for multiple comparisons.',
              '- Own-radius normalization changes each matrix multiplier. Shared-CE uses the same multiplier at a given fraction, and Frobenius matching equalizes a matrix norm. None alone establishes an isolated eigenmode mechanism.',
              '- The ER gamma/lognormal vectors match realized mean and variance and preserve the rank placement of empirical ER weights (with paired tie-breaking); they are fixed mid-quantile vectors, not iid random samples. Mean/variance matching also matches the raw Frobenius norm for an equal edge count.',
              '- Inhibitory-edge fractions are imposed sign quotas, not estimates of biological functional E/I balance. The removed-edge CE control has fewer edges; directed WS construction follows the existing WS-like generator.',
              '- No reservoir was discarded for poor performance. Grid-mean results, conditional input comparisons and win fractions should be interpreted together. No simulations were launched for this analysis.','',
              '## Files and reproducibility','',
              'Each suite directory contains PNGs and CSVs for mean curves, per-setting advantages, win curves, both AUC definitions, input dependence, horizon dependence and ANOVA components. '
              'The top-level manifest records checksums, filters, pairing keys and bootstrap settings. Cached cubes retain all joined scalar scores. '
              'Original data, prior analyses and poster figures are preserved.','']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=ROOT/'results/mc200_recompute/mc200_274500')
    p.add_argument('--out-dir',type=Path,default=ROOT/'analysis/mc200_recompute/mc200_274500');p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=20261122)
    a=p.parse_args();out=a.out_dir.resolve()
    if a.bootstrap<2:raise ValueError('At least two bootstrap replicates required')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new analysis folder')
    (out/'cache').mkdir(parents=True)
    m,configs,cubes,sources=load_data(a.run_dir.resolve(),out)
    n=configs['architecture']['repeats'];weights=np.random.default_rng(a.seed).multinomial(n,np.full(n,1/n),size=a.bootstrap)/n
    np.savez_compressed(out/'cache'/'bootstrap_weights.npz',weights=weights)
    all_tables={};all_horizons={};all_tails={};figure_style()
    for suite,cube in cubes.items():
        dest=out/suite;dest.mkdir();config=configs[suite]
        tables=summarize(cube,config,weights,'ce_empirical' if suite=='architecture' else 'empirical')
        all_tables[suite]=tables
        for name,table in tables.items():table.to_csv(dest/f'{name}.csv',index=False)
        h,changes,tails=horizon_tables(cube,config,weights);all_horizons[suite]=h;all_tails[suite]=tails
        h.to_csv(dest/'horizon_comparison.csv',index=False);changes.to_csv(dest/'horizon_and_test_length_changes.csv',index=False);tails.to_csv(dest/'horizon_tail_diagnostics.csv',index=False)
        overview_plots(tables,config,dest,suite);input_plots(tables,config,dest,suite);variance_plots(tables['hyperparameter_variance'],config,dest,suite);horizon_plots(h,config,dest,suite)
        if suite=='er':gamma_plot(tables,dest)
        print(f'Finished {suite} PNGs',flush=True)
    delay_profiles(a.run_dir.resolve(),m,out,weights)
    write_report(out,m,configs,all_tables,all_horizons,all_tails,sources,a)
    settings=dict(experiment_id=m['experiment_id'],sources=sources,bootstrap=a.bootstrap,bootstrap_seed=a.seed,
        bootstrap_unit='whole paired realization shared across all conditions',pairing_keys=KEYS,filters='none',
        metrics=METRICS,primary_metrics=PRIMARY,normalization='AUC divided by observed E/I span',
        ties='exact zero after seed averaging; separate from strict wins',test_steps_MC200=m['tasks']['test'],test_steps_NARMA10=500,
        script_sha256=sha(Path(__file__)),outputs={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and 'cache' not in f.parts})
    (out/'manifest.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(f'Analysis complete: {out}; {len(list(out.rglob("*.png")))} PNGs',flush=True)


if __name__=='__main__':main()
