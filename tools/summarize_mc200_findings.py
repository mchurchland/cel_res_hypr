#!/usr/bin/env python3
"""Focused MC200 figures and interpretation from the checked analysis caches."""
from pathlib import Path
import itertools
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.analyze_mc200 import (METRICS, PRIMARY, NAMES, ARCH_LABELS, ARCH_COLORS,
    ER_LABELS, ER_COLORS, MODES, interval, figure_style, save, sha)
from util.ei_auc import curve_areas
import matplotlib.pyplot as plt

OUT=ROOT/'analysis/mc200_recompute/mc200_274500'


def cache(suite):
    with np.load(OUT/'cache'/f'{suite}_scores.npz',allow_pickle=False) as z:
        return z['scores'],json.loads(str(z['config']))


def focus_curves():
    w=pd.read_csv(OUT/'architecture/winning_fraction.csv')
    controls=['er_binary','er_halfnormal','er_signed_empirical']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,metric in zip(axes,PRIMARY):
        for family,marker,ls in zip(controls,['o','s','^'],['-','--','-.']):
            g=w.query('normalization=="own_radius" and input_scale=="ALL" and metric==@metric and control==@family').sort_values('negative_fraction')
            ax.plot(g.negative_fraction*100,g.winning_fraction*100,color=ARCH_COLORS[family],marker=marker,ms=4,lw=1.8,ls=ls,label=ARCH_LABELS[family])
            ax.fill_between(g.negative_fraction*100,g.ci_low*100,g.ci_high*100,color=ARCH_COLORS[family],alpha=.10)
        ax.axhline(50,color='.5',ls=':',lw=.8);ax.set(title=NAMES[metric],xlabel='Inhibitory edges (%)',ylim=(-3,103),xticks=np.arange(0,101,10));ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Matched settings empirical CE wins (%)')
    fig.suptitle('CE versus three ER weight controls: own-radius scaling')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015))
    fig.tight_layout(rect=(0,.07,1,.93));save(fig,OUT/'key_win_curves.png')


def gamma_focus(cube,c,weights):
    x=np.asarray(c['fractions']);ni=c['normalizations'].index('own_radius');gi=c['families'].index('gamma');ei=c['families'].index('empirical')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(metric,direction) in zip(axes,PRIMARY.items()):
        mi=METRICS.index(metric)
        difference=direction*(cube[:,gi,ni,:,:,mi]-cube[:,ei,ni,:,:,mi])
        mean,lo,hi=interval(difference.mean(axis=-1),weights)
        ax.plot(x*100,mean,color='#AA3377',marker='o',ms=4,lw=1.8);ax.fill_between(x*100,lo,hi,color='#AA3377',alpha=.15)
        ax.axhline(0,color='.4',ls=':',lw=1);ax.set(title=NAMES[metric],xlabel='Inhibitory edges (%)',ylabel='Gamma MC200 gain' if metric=='MC_200' else 'Gamma NMSE reduction');ax.grid(alpha=.13)
    fig.suptitle('Gamma versus empirical CE magnitudes on the same ER networks\nOwn-radius scaling; 95% paired-realization intervals')
    fig.tight_layout(rect=(0,0,1,.90));save(fig,OUT/'gamma_gain_own_radius.png')


def own_auc():
    a=pd.read_csv(OUT/'architecture/performance_auc.csv');families=list(ARCH_LABELS)
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True,layout='constrained')
    for ax,metric in zip(axes,PRIMARY):
        g=a.query('normalization=="own_radius" and input_scale=="ALL" and metric==@metric').set_index('family').loc[families]
        for j,(family,row) in enumerate(g.iterrows()):
            ax.hlines(j,row.ci_low,row.ci_high,color=ARCH_COLORS[family],lw=1.5);ax.scatter(row.performance_auc,j,color=ARCH_COLORS[family],s=35)
        ax.set(title=NAMES[metric],xlabel='Mean MC200 AUC (higher is better)' if metric=='MC_200' else 'Mean NMSE AUC (lower is better)');ax.grid(axis='x',alpha=.15)
    axes[0].set(yticks=range(len(families)),yticklabels=[ARCH_LABELS[f] for f in families]);axes[0].invert_yaxis()
    fig.suptitle('Architecture comparison: own-radius scaling\nAll 24 settings; 95% paired-realization bootstrap intervals')
    save(fig,OUT/'key_performance_auc.png')


def horizon_win_changes(cubes,configs):
    rows=[];aucs=[]
    selected=['archived_MC30','MC_30','MC_60','MC_100','MC_150','MC_200']
    for suite,cube in cubes.items():
        c=configs[suite];bi=c['families'].index('ce_empirical' if suite=='architecture' else 'empirical');x=np.asarray(c['fractions'])
        for ni,norm in enumerate(c['normalizations']):
            for fi,family in enumerate(c['families']):
                if fi==bi:continue
                for metric in selected:
                    mi=METRICS.index(metric);diff=(cube[:,bi,ni,:,:,mi]-cube[:,fi,ni,:,:,mi]).mean(axis=0)
                    curve=100*(diff>0).mean(axis=-1)
                    rows.extend(dict(suite=suite,control=family,normalization=norm,horizon_metric=metric,negative_fraction=f,reference_win_percent=v) for f,v in zip(x,curve))
                    aucs.append(dict(suite=suite,control=family,normalization=norm,horizon_metric=metric,reference_win_auc_percent=float(curve_areas(curve,x)[0]/(x[-1]-x[0]))))
    d=pd.DataFrame(rows);d.to_csv(OUT/'horizon_win_curves.csv',index=False);pd.DataFrame(aucs).to_csv(OUT/'horizon_win_auc.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharey=True)
    for ax,family in zip(axes,['er_binary','er_halfnormal','er_signed_empirical']):
        for metric,color,ls in [('MC_30','#777777','--'),('MC_200',ARCH_COLORS[family],'-')]:
            g=d.query('suite=="architecture" and control==@family and normalization=="own_radius" and horizon_metric==@metric').sort_values('negative_fraction')
            ax.plot(g.negative_fraction*100,g.reference_win_percent,color=color,ls=ls,lw=1.9,marker='o',ms=3,label='30 delays' if metric=='MC_30' else '200 delays')
        ax.axhline(50,color='.6',ls=':',lw=.7);ax.set(title=ARCH_LABELS[family],xlabel='Inhibitory edges (%)',ylim=(-3,103));ax.grid(alpha=.12);ax.legend(frameon=False)
    axes[0].set_ylabel('Matched settings CE wins (%)')
    fig.suptitle('Does the MC cutoff change the winning regions?\nSame 5,000 test samples; own-radius scaling; descriptive point estimates')
    fig.tight_layout(rect=(0,0,1,.88));save(fig,OUT/'horizon_win_comparison.png')


def peaks(cubes,configs):
    records=[];curves=[]
    for suite,cube in cubes.items():
        c=configs[suite];points=list(itertools.product(c['rhos'],c['leaks'],c['inputs']))
        for ni,norm in enumerate(c['normalizations']):
            means=cube[:,:,ni,:,:,METRICS.index('MC_200')].mean(axis=0)
            for fi,family in enumerate(c['families']):
                xi,pi=np.unravel_index(means[fi].argmax(),means[fi].shape)
                records.append(dict(suite=suite,family=family,normalization=norm,maximum_seed_mean_MC200=means[fi,xi,pi],
                    negative_fraction=c['fractions'][xi],rho_target=points[pi][0],leak=points[pi][1],input_scale=points[pi][2],
                    selection='descriptive maximum over test-scored fixed grid; not independent validation of tuned performance'))
                curves.extend(dict(suite=suite,family=family,normalization=norm,negative_fraction=f,maximum_seed_mean_MC200=means[fi,j].max()) for j,f in enumerate(c['fractions']))
    pd.DataFrame(records).to_csv(OUT/'descriptive_peak_MC200.csv',index=False)
    d=pd.DataFrame(curves);d.to_csv(OUT/'descriptive_peak_MC200_curves.csv',index=False)
    fig,ax=plt.subplots(figsize=(10.5,5.7))
    for family in ['ce_empirical','ce_shuffled','er_signed_empirical','er_binary','er_halfnormal','ws_signed_empirical','ws_binary','ws_halfnormal']:
        g=d.query('suite=="architecture" and normalization=="own_radius" and family==@family')
        ax.plot(g.negative_fraction*100,g.maximum_seed_mean_MC200,color=ARCH_COLORS[family],marker='o',ms=3,lw=1.7,label=ARCH_LABELS[family])
    ax.set(xlabel='Inhibitory edges (%)',ylabel='Largest seed-mean MC200 among 24 settings',ylim=(0,None),title='Descriptive best observed MC200 at each fraction\nOwn-radius scaling; selection uses test scores');ax.grid(alpha=.15)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
    save(fig,OUT/'best_observed_MC200.png')


def profiles_with_distinct_colors():
    d=pd.read_csv(OUT/'selected_delay_profiles.csv')
    cases=list(itertools.product([.2,.5],[.95,1.05],[.6,1.]))
    families=['ce_empirical','ce_shuffled','er_signed_empirical','er_binary','ws_signed_empirical','ws_binary','gamma','lognormal']
    colors=dict(ARCH_COLORS,gamma='#AA3377',lognormal='#44AA99')
    labels=dict(ARCH_LABELS,gamma='ER moment-matched gamma',lognormal='ER moment-matched lognormal')
    for detailed in (False,True):
        selected=cases if detailed else [(.2,.95,1.),(.5,.95,1.)]
        fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True) if detailed else plt.subplots(1,2,figsize=(13,5),sharey=True)
        for ax,case in zip(axes.flat,selected):
            for family in families:
                g=d[(d.family==family)&(d.negative_fraction==case[0])&(d.rho_target==case[1])&(d.leak==case[2])]
                ax.plot(g.delay,g['mean'],color=colors[family],label=labels[family],lw=1.6,ls='--' if family=='gamma' else '-')
                if not detailed:ax.fill_between(g.delay,g.ci_low,g.ci_high,color=colors[family],alpha=.08)
            ax.axvline(30,color='.5',ls=':',lw=1);ax.set(title=f'{100*case[0]:g}% inhibitory; radius {case[1]:g}; leak {case[2]:g}',xlabel='Input delay',ylim=(-.02,1.03),xlim=(1,200));ax.grid(alpha=.12)
        axes.flat[0].set_ylabel('Held-out recall r²')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.04))
        fig.suptitle('Memory by delay: own-radius scaling, input scale 0.01\nDotted line marks the archived 30-delay cutoff')
        fig.tight_layout(rect=(0,.09,1,.92));save(fig,OUT/('memory_delay_profiles_all_settings.png' if detailed else 'memory_delay_profiles.png'))


def validate(cubes,configs,weights):
    checks=[]
    for suite,cube in cubes.items():
        c=configs[suite];x=np.array(c['fractions']);t=pd.read_csv(OUT/suite/'performance_auc.csv');means=pd.read_csv(OUT/suite/'mean_performance.csv')
        for ni,norm in enumerate(c['normalizations']):
            for metric in PRIMARY:
                mi=METRICS.index(metric)
                expected=curve_areas(cube[:,:,ni,:,:,mi].mean(axis=-1),x)[0].mean(axis=0)/(x[-1]-x[0])
                got=t.query('normalization==@norm and metric==@metric and input_scale=="ALL"').set_index('family').loc[c['families']].performance_auc.to_numpy()
                np.testing.assert_allclose(expected,got,rtol=1e-12,atol=1e-12)
                for fi,family in enumerate(c['families']):
                    g=means.query('normalization==@norm and metric==@metric and input_scale=="ALL" and family==@family').sort_values('negative_fraction')
                    np.testing.assert_allclose(g['mean'],cube[:,fi,ni,:,:,mi].mean(axis=(0,2)),rtol=1e-12)
                checks.append(f'{suite}/{norm}/{metric}: AUCs and curves independently reconstructed')
        w=pd.read_csv(OUT/suite/'win_fraction_auc.csv');np.testing.assert_allclose(w.reference_win_auc_percent+w.control_win_auc_percent+w.tie_auc_percent,100,atol=1e-10)
        h=cube[..., :5];assert np.all(np.diff(h,axis=-1)>=-1e-12)
    a,ac=cubes['architecture'],configs['architecture'];e,ec=cubes['er'],configs['er']
    for family,arch in [('empirical','er_signed_empirical'),('binary','er_binary'),('halfnormal','er_halfnormal')]:
        for norm in ['own_radius','shared_ce_scale']:
            np.testing.assert_array_equal(a[:,ac['families'].index(arch),ac['normalizations'].index(norm)],e[:,ec['families'].index(family),ec['normalizations'].index(norm)])
    for family in ['empirical','gamma','lognormal']:
        fi=ec['families'].index(family)
        np.testing.assert_array_equal(e[:,fi,ec['normalizations'].index('shared_ce_scale'),:,:,:6],e[:,fi,ec['normalizations'].index('frobenius_match'),:,:,:6])
    checks += ['Exact baseline aliases agree across suites','All new cumulative horizons are monotonic','Shared/Frobenius moment-equal conditions have identical new MC scores','Win/tie AUC partitions sum to 100%']
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',checks=checks,new_analysis_unit_tests=6,reused_AUC_unit_tests=6,bootstrap_draws=len(weights)),indent=2)+'\n')


def main():
    figure_style();m=json.loads((OUT/'manifest.json').read_text())
    for name,digest in m['outputs'].items():
        assert sha(OUT/name)==digest,f'Primary analysis output changed: {name}'
    cubes={};configs={}
    for suite in ['architecture','er']:cubes[suite],configs[suite]=cache(suite)
    with np.load(OUT/'cache/bootstrap_weights.npz') as z:weights=z['weights']
    validate(cubes,configs,weights)
    focus_curves();gamma_focus(cubes['er'],configs['er'],weights);own_auc();horizon_win_changes(cubes,configs);peaks(cubes,configs);profiles_with_distinct_colors()
    # A paired comparison distinguishes a close ranking from an established winner.
    c=configs['architecture'];cube=cubes['architecture'];ni=c['normalizations'].index('own_radius');x=np.array(c['fractions'])
    diff=cube[:,c['families'].index('ws_halfnormal'),ni,:,:,METRICS.index('MC_200')]-cube[:,c['families'].index('ws_signed_empirical'),ni,:,:,METRICS.index('MC_200')]
    mu,lo,hi=interval(curve_areas(diff.mean(axis=-1),x)[0]/(x[-1]-x[0]),weights)
    pd.DataFrame([dict(comparison='WS half-normal minus WS empirical',metric='MC200',normalization='own_radius',advantage_auc=mu,ci_low=lo,ci_high=hi)]).to_csv(OUT/'top_WS_paired_comparison.csv',index=False)
    print(f'Focused plots and independent validation complete; WS difference {mu:.5f} [{lo:.5f}, {hi:.5f}]',flush=True)
    # Findings and gallery are added separately, then the final output inventory is refreshed.
    m['supplementary_script_sha256']=sha(Path(__file__))
    m['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and 'cache' not in p.parts}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':main()
