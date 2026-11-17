#!/usr/bin/env python3
"""Exploratory, paired analyses of existing E/I sweeps; no new simulations."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-robustness')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/robustness-xdg-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.robustness import PARAMETERS,bootstrap_mean

BASE=ROOT/'analysis/robustness_reevaluation_2026-11-16'
OUT=ROOT/'analysis/exploratory_followup_2026-11-17'
METRICS=['MC','IPC','KR','IPC_minus_MC']
FAMILIES=['cel_matched','cel_removed','matched_er']
LABELS={'cel_matched':'C. elegans: unknown signs matched','cel_removed':'C. elegans: unknown signs removed','matched_er':'Matched ER'}
COMMON=[0.,20.,40.,60.,80.,100.]
NBOOT=1000


def sha(path):
    with open(path,'rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def load_surfaces():
    plotdir=BASE/'figures/ei_hyperparameter_dots'
    manifest=json.loads((plotdir/'manifest.json').read_text())
    for source in manifest['sources']:
        assert sha(ROOT/source['path'])==source['sha256']
    identity=pd.read_csv(BASE/'tables/per_realization.csv',usecols=['dataset','architecture','normalization'])
    points=pd.read_csv(plotdir/'plotted_points.csv.gz')
    for name in PARAMETERS:points[name]=points[name].round(12)
    frames=[];caches=[]
    for prefix in ['sign','norm']:
        for family in FAMILIES:
            dataset=f'{prefix}_{family}'
            groups=identity[identity.dataset==dataset][['architecture','normalization']].drop_duplicates().sort_values(['architecture','normalization'])
            for group_index,(_,record) in enumerate(groups.iterrows()):
                import re
                percent=float(re.search(r'(?:sign_test_og_cel|sign_test_er)([0-9.]+)$',record.architecture).group(1))*100
                experiment='sign' if prefix=='sign' else 'normalization'
                values={}
                for metric in METRICS[:3]:
                    path=BASE/'cache/matrices'/f'{dataset}_{group_index}_{metric}.npz'
                    with np.load(path) as a:
                        y=a['values'].copy();grid=a['grid'].round(12);seeds=a['seeds'].copy()
                    if values:
                        np.testing.assert_array_equal(seeds,seed_ref);np.testing.assert_allclose(grid,grid_ref)
                    seed_ref,grid_ref=seeds,grid
                    g=points[(points.experiment==experiment)&(points.family==family)&(points.normalization==record.normalization)&(points.metric==metric)&np.isclose(points.negative_percent,percent,rtol=0,atol=1e-8)].sort_values(list(PARAMETERS))
                    assert len(g)==96 and np.isfinite(y).all()
                    np.testing.assert_allclose(y.mean(axis=0),g.performance,rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(grid,g[list(PARAMETERS)],rtol=0,atol=1e-10)
                    values[metric]=y.ravel()
                    caches.append(dict(path=str(path.relative_to(BASE)),sha256=sha(path),verified_against_plotted_means=True))
                fields=dict(experiment=experiment,family=family,normalization=record.normalization,negative_percent=percent,seed=np.repeat(seeds,96),**values)
                fields.update({name:np.tile(grid[:,i],len(seeds)) for i,name in enumerate(PARAMETERS)})
                frames.append(pd.DataFrame(fields))
    data=pd.concat(frames,ignore_index=True)
    data['IPC_minus_MC']=data.IPC-data.MC
    # Retain inconsistent archived scores for audit; never clip or hide them.
    data[data.IPC_minus_MC<0].to_csv(OUT/'tables/ipc_below_mc_anomalies.csv',index=False,float_format='%.15g')
    return data,manifest,caches


def summarize(g,meta,rng):
    # One observation per seed, already averaged over its matched configurations.
    assert not g.seed.duplicated().any()
    mean,ci=bootstrap_mean(g[METRICS].to_numpy(),NBOOT,rng)
    rows=[]
    for i,metric in enumerate(METRICS):
        rows.append(dict(**meta,metric=metric,estimate=mean[i],ci_low=ci[0,i],ci_high=ci[1,i],n_seeds=len(g)))
    return rows


def compare_models(data,rng):
    keys=['experiment','normalization','negative_percent','seed',*PARAMETERS]
    pair=data[data.family=='cel_matched'].merge(data[data.family=='matched_er'],on=keys,suffixes=('_ce','_er'),validate='one_to_one')
    for metric in METRICS:pair[metric]=pair[f'{metric}_ce']-pair[f'{metric}_er']
    assert len(pair)>0
    np.testing.assert_allclose(pair.IPC,pair.MC+pair.IPC_minus_MC,atol=1e-10)
    rows=[];cells=[]
    base=['experiment','normalization','negative_percent']
    for parameter in ['ALL',*PARAMETERS]:
        groupkeys=base+([] if parameter=='ALL' else [parameter])
        for key,g in pair.groupby(groupkeys,sort=True):
            if not isinstance(key,tuple):key=(key,)
            meta=dict(zip(groupkeys,key));meta['parameter']=parameter
            meta['level']='ALL' if parameter=='ALL' else meta.pop(parameter)
            seed=g.groupby('seed',as_index=False)[METRICS].mean()
            rows.extend(summarize(seed,meta,rng))
    # Configurations here are points of the finite design, never independent networks.
    for keys2,g in pair.groupby(base+list(PARAMETERS)):
        meta=dict(zip(base+list(PARAMETERS),keys2))
        for metric in METRICS:
            cells.append(dict(**meta,metric=metric,mean_difference=g[metric].mean(),n_seeds=g.seed.nunique()))
    return pd.DataFrame(rows),pd.DataFrame(cells)


def manipulations(data,rng):
    common=data[data.negative_percent.isin(COMMON)]
    rows=[];frequencies=[]
    for parameter,low,high in [('neuron_bias',0.,.1),('input_scale',.1,1.5)]:
        keys=['experiment','family','normalization','negative_percent','seed',*[p for p in PARAMETERS if p!=parameter]]
        pairs=common[common[parameter]==high].merge(common[common[parameter]==low],on=keys,suffixes=('_high','_low'),validate='one_to_one')
        for metric in METRICS:pairs[metric]=pairs[f'{metric}_high']-pairs[f'{metric}_low']
        for grouping in [['experiment','family','normalization'],['experiment','family','normalization','negative_percent']]:
            for key,g in pairs.groupby(grouping):
                meta=dict(zip(grouping,key));meta.update(parameter=parameter,low=low,high=high,scope='across_common_ei' if 'negative_percent' not in grouping else 'per_ei')
                means=g.groupby('seed',as_index=False)[METRICS].mean()
                rows.extend(summarize(means,meta,rng))
        groupkeys=['experiment','family','normalization','negative_percent',*[p for p in PARAMETERS if p!=parameter]]
        avg=pairs.groupby(groupkeys,as_index=False)[METRICS].mean()
        for key,g in avg.groupby(['experiment','family','normalization']):
            for metric in METRICS:
                frequencies.append(dict(zip(['experiment','family','normalization'],key),parameter=parameter,metric=metric,n_matched_configuration_pairs=len(g),n_positive=int((g[metric]>0).sum()),n_negative=int((g[metric]<0).sum()),fraction_positive=np.mean(g[metric]>0)))
    return pd.DataFrame(rows),pd.DataFrame(frequencies)


def polarity(data,rng):
    keys=['experiment','family','normalization','seed',*PARAMETERS]
    pairs=data[data.negative_percent==100].merge(data[data.negative_percent==0],on=keys,suffixes=('_inhib','_exc'),validate='one_to_one')
    for metric in METRICS:pairs[metric]=pairs[f'{metric}_inhib']-pairs[f'{metric}_exc']
    rows=[]
    for key,g in pairs.groupby(['experiment','family','normalization','leak','neuron_bias']):
        meta=dict(zip(['experiment','family','normalization','leak','neuron_bias'],key))
        seed=g.groupby('seed',as_index=False)[METRICS].mean()
        rows.extend(summarize(seed,meta,rng))
    return pd.DataFrame(rows)


def tested_input_optima(out):
    points=pd.read_csv(BASE/'figures/ei_hyperparameter_dots/plotted_points.csv.gz')
    points=points[(points.experiment=='sign')&points.negative_percent.isin(COMMON)]
    profile_keys=['family','negative_percent','metric','rho_target','leak','neuron_bias']
    best=points.loc[points.groupby(profile_keys).performance.idxmax(),profile_keys+['input_scale','performance']]
    best.to_csv(out/'tables/best_tested_input_by_profile.csv',index=False,float_format='%.12g')
    marginal=points.groupby(['family','negative_percent','metric','input_scale'],as_index=False).performance.mean()
    best_mean=marginal.loc[marginal.groupby(['family','negative_percent','metric']).performance.idxmax()]
    best_mean.to_csv(out/'tables/best_tested_input_by_ei_mean.csv',index=False,float_format='%.12g')
    counts=best.groupby(['family','metric','input_scale']).size().rename('n_profiles').reset_index()
    counts.to_csv(out/'tables/best_tested_input_counts.csv',index=False)
    return counts


def plots(conditional,effects,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    styles=[('sign','spectral_radius','Updated sign sweep: own radius (200 seeds)'),('normalization','spectral_radius','Archival controls: own radius (100 seeds)'),('normalization','original_radius','Archival controls: fixed reference (100 seeds)')]
    colors=['#332288','#0077bb','#009988','#ee7733']
    fig,axes=plt.subplots(3,3,figsize=(13.5,10),layout='constrained')
    for i,(experiment,norm,label) in enumerate(styles):
        for j,metric in enumerate(METRICS[:3]):
            ax=axes[i,j]
            g=conditional[(conditional.experiment==experiment)&(conditional.normalization==norm)&(conditional.parameter=='input_scale')&(conditional.metric==metric)]
            for k,level in enumerate([.1,.5,1.,1.5]):
                h=g[np.isclose(pd.to_numeric(g.level),level)].sort_values('negative_percent')
                x=h.negative_percent+(k-1.5)*.7
                ax.errorbar(x,h.estimate,yerr=[h.estimate-h.ci_low,h.ci_high-h.estimate],fmt='o-',ms=3.5,lw=1,color=colors[k],label=f'Input {level:g}')
            ax.axhline(0,color='black',lw=.8)
            ax.set(xlabel='Inhibitory edges (%)',ylabel=f'CE − ER mean {metric}',title=f'{label}\n{metric}',xticks=COMMON)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('The crossover depends on input drive\nOther hyperparameters averaged uniformly; 95% paired seed-bootstrap intervals',fontsize=12)
    fig.savefig(out/'crossover_by_input_scale.png',dpi=300,bbox_inches='tight');plt.close(fig)
    for parameter,stem,title in [('neuron_bias','bias_effects','Adding bias: 0 → 0.1'),('input_scale','input_drive_effects','Increasing input scale: 0.1 → 1.5')]:
        fig,axes=plt.subplots(3,3,figsize=(12,8.5),layout='constrained')
        for i,family in enumerate(FAMILIES):
            for j,metric in enumerate(METRICS[:3]):
                ax=axes[i,j]
                g=effects[(effects.family==family)&(effects.metric==metric)&(effects.parameter==parameter)&(effects.scope=='per_ei')]
                for k,(experiment,norm,label) in enumerate(styles):
                    h=g[(g.experiment==experiment)&(g.normalization==norm)].sort_values('negative_percent')
                    ax.errorbar(h.negative_percent+(k-1)*.7,h.estimate,yerr=[h.estimate-h.ci_low,h.ci_high-h.estimate],fmt='o-',ms=3,lw=1,color=colors[k],label=label)
                ax.axhline(0,color='black',lw=.8)
                ax.set(xlabel='Inhibitory edges (%)',ylabel=f'Change in {metric}',title=f'{LABELS[family]}\n{metric}',xticks=COMMON)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=3,frameon=False,fontsize=8)
        fig.suptitle(title+'\nMatched configurations within each seed; 95% paired seed-bootstrap intervals',fontsize=12)
        fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight');plt.close(fig)
    g=conditional[(conditional.experiment=='sign')&(conditional.parameter=='ALL')]
    table=g.pivot(index='negative_percent',columns='metric',values='estimate')[['MC','IPC','IPC_minus_MC','KR']]
    fig,ax=plt.subplots(figsize=(10.5,4.7));ax.axis('off')
    cells=[[f'{p:.0f}%',*[f'{v:+.3f}' for v in row]] for p,row in table.iterrows()]
    t=ax.table(cellText=cells,colLabels=['Inhibitory edges','ΔMC','ΔIPC','Δ(IPC − MC)','ΔKR'],cellLoc='center',bbox=[0,.16,1,.73],colWidths=[.2]*5)
    t.auto_set_font_size(False);t.set_fontsize(12)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#dce3eb')
        cell.set_facecolor('#223b58' if r==0 else ('#f2f5f8' if r%2==0 else 'white'))
        if r==0:cell.get_text().set_color('white')
    fig.suptitle('CE − ER: separating the shared linear-memory component',fontsize=14)
    fig.text(.11,.055,'IPC − MC is an implementation-dependent residual, not an independently measured task score.',fontsize=9)
    fig.savefig(out/'ipc_memory_component_comparison.png',dpi=300,bbox_inches='tight');plt.close(fig)


def report(conditional,effects,frequencies,polarity_results,data,out):
    lines=['# Exploratory follow-up: candidate findings and counterexamples\n',
      'This is an exploratory search of stored final_results only. No simulations were run. Results are descriptive, conditional on the sampled grid and protocols, and selected after inspecting the data. Bootstrap intervals are marginal, not multiplicity-adjusted confirmation.\n',
      '## 1. The CE–ER crossover is strongly conditional on input scale\n',
      'At an input scale of 0.1, the updated sign sweep gives CE a mean MC and IPC advantage at 0%, 20%, 80% and 100% inhibitory edges. At 40% and 60%, ER is higher. At larger input scales, CE’s MC advantage near 20% and 80% becomes negligible or reverses, and its IPC advantage often reverses even at 0%. A single pooled crossover therefore reflects a chosen distribution over hyperparameters.\n']
    g=conditional[(conditional.experiment=='sign')&(conditional.parameter=='input_scale')&conditional.metric.isin(['MC','IPC'])]
    lines.append('| Metric | Inhibitory edges | Input scale | CE − ER mean | 95% seed interval |\n| --- | ---: | ---: | ---: | --- |')
    for _,r in g[g.negative_percent.isin([0,20,40,80,100]) & pd.to_numeric(g.level).isin([.1,1.5])].iterrows():
        lines.append(f'| {r.metric} | {r.negative_percent:.0f}% | {float(r.level):g} | {r.estimate:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
    lines+=['\n**Potential use:** define the crossover conditional on input drive, or state exactly which grid average it estimates. This can guide a small, targeted follow-up rather than a full multidimensional simulation campaign. Normalization controls and their archived KR remain separate; see the figure and full CSV.\n',
      '## 2. Bias exchanges linear memory for IPC and state-space separation\n',
      'Changing bias range from 0 to 0.1 usually lowers MC, raises IPC, and raises KR. The following estimates first average matched high-minus-low differences within each seed over the other parameters and the six shared E/I levels, then average across seeds.\n',
      '| Model | Metric | Mean change from bias | 95% seed interval | Positive matched grid contrasts |\n| --- | --- | ---: | --- | ---: |']
    chosen=effects[(effects.experiment=='sign')&(effects.parameter=='neuron_bias')&(effects.scope=='across_common_ei')&effects.metric.isin(METRICS[:3])]
    for _,r in chosen.iterrows():
        f=frequencies[(frequencies.experiment=='sign')&(frequencies.parameter=='neuron_bias')&(frequencies.family==r.family)&(frequencies.metric==r.metric)].iloc[0]
        lines.append(f'| {LABELS[r.family]} | {r.metric} | {r.estimate:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {int(f.n_positive)}/{int(f.n_matched_configuration_pairs)} |')
    lines+=['\n**Potential use:** parameter importance hides the direction of its effect. Bias is a small contributor to total MC variance but consistently shifts which kind of computation the model supports. A plausible explanation is symmetry breaking in the tanh reservoir; the stored full-grid results do not decompose IPC by order at both bias settings, so this explanation is not established. The existing order-validation experiment contains bias=0 only.\n',
      '## 3. Input drive creates a clear memory–separation tension\n',
      'For both CE variants, changing input scale from 0.1 to 1.5 lowers MC and IPC and raises KR in every matched seed-averaged configuration contrast over the six shared E/I levels. ER also shows a universally negative MC contrast and positive KR contrast; its IPC response has exceptions. These signs describe 144 finite-design contrasts per model, not 144 independent biological samples.\n',
      '| Model | Metric | Mean change with stronger input | 95% seed interval |\n| --- | --- | ---: | --- |']
    selected=effects[(effects.experiment=='sign')&(effects.parameter=='input_scale')&(effects.scope=='across_common_ei')&effects.metric.isin(METRICS[:3])]
    for _,r in selected.iterrows():
        lines.append(f'| {LABELS[r.family]} | {r.metric} | {r.estimate:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |')
    lines+=['\n**Potential use:** do not describe larger KR as uniformly better computational performance. The strongest tested input drive increases separation while reducing linear memory. The best tested input setting for MC is at the lower boundary of the grid; these data do not locate an unconstrained optimum below 0.1.\n',
      '## 4. MC and IPC are not independent pieces of evidence\n',
      'The inspected implementation computes MC and IPC from the same reservoir states, training/testing signals, ridge readout and delay range (30). IPC includes order 1, whose Legendre target is the original input, alongside higher orders. Consequently its linear contribution overlaps MC. The diagnostic residual IPC − MC is appropriate only conditional on those documented settings applying to the archived run; batched numerical solves can differ slightly. It is not an independently validated nonlinear task score.\n',
      'In the updated own-radius CE-versus-ER sweep, CE’s overall IPC advantage at the two E/I endpoints is smaller than its MC advantage. The IPC−MC residual is lower in CE at every shared E/I level. Thus the endpoint MC and IPC advantages should not be presented as two independent demonstrations of a broad computational advantage. The residual pattern changes in the normalization controls, which further limits a universal interpretation.\n',
      f"**Audit finding:** {int((data.IPC_minus_MC<0).sum())} archived configuration/seed rows have IPC below MC, with a minimum residual of {data.IPC_minus_MC.min():.3f}. Their exact identities are in tables/ipc_below_mc_anomalies.csv. These values are incompatible with an exact nonnegative decomposition if archived MC and IPC use the same linear target/readout protocol and exact arithmetic. Historical protocol differences or numerical solve differences need checking; these data do not identify the cause. No values were removed or clipped. Until resolved, IPC−MC should remain a diagnostic residual rather than a verified nonlinear-capacity measure.\n",
      '**Potential use:** explicitly separate linear-memory contribution from higher-order capacity before interpreting IPC. This is a useful qualification even if the paper remains centered on locating a crossover.\n',
      '## A practical way to reduce the crossover experiment\n',
      'Input scale 0.1 gives the highest grid-averaged MC at every shared E/I level for all three model families. Holding radius/leak/bias fixed, it is also the best tested MC input in all 144 CE profiles, all 144 unknown-sign-removed CE profiles, and 129/144 ER profiles; ER prefers 0.5 in the other 15. These counts are over seed-averaged surfaces. The best input sits at the tested lower boundary, so it is not an estimate of the unconstrained optimum.\n',
      'For an explicitly conditional MC crossover study, fixing input scale at 0.1 would reduce the grid from 96 to 24 settings while retaining all radii, leaks and bias levels. That is a 75% reduction in configurations per E/I point. It would estimate the crossover at a shared, generally favorable tested MC input, not the crossover averaged over the original full grid, and it would not optimize input separately for every ER profile. The existing data support checking this restricted estimand before any new simulations. Relevant per-profile and per-E/I optima are exported.\n',
      '## Additional check: endpoint polarity symmetry\n',
      'The all-negative minus all-positive contrasts, split by leak and bias, are exported in tables/polarity_asymmetry.csv. Their dependence on leak can motivate an explanation involving self-memory and recurrent sign. These were inspected as secondary exploratory diagnostics, not used to claim a new biological mechanism.\n',
      '## Methods and limitations\n',
      'All effects hold the other hyperparameters fixed before averaging. Each seed retains its full matched grid. Mean differences and 95% percentile intervals use 1,000 seed bootstrap draws; configurations and E/I levels are not resampled as biological replicates. Across-model paired numerical seeds preserve the experiment’s common-random-number design; they do not make CE and ER the same network. All summary effects across E/I use exactly the six shared fractions 0, 20, 40, 60, 80 and 100%, avoiding the extra CE-only reference point.\n',
      'Main sign sweeps use updated KR over 200 seeds; archival normalization controls use 100 seeds and are kept separate. CE and ER differ in topology and weight structure, so no contrast isolates topology by itself. The models use assigned edge signs, not a fitted functional E/I balance. An unexplained correlation or crossover cannot independently estimate biological sign fraction.\n',
      'The complete tables retain all examined strata, including counterexamples and weak effects. They are not confirmatory tests of hypotheses specified before seeing these data. No publication novelty or acceptance claim is made.\n',
      '## Files\n\n- crossover_by_input_scale.png\n- bias_effects.png\n- input_drive_effects.png\n- ipc_memory_component_comparison.png\n- tables/conditional_ce_minus_er.csv: all four metrics, all individual-parameter strata and complete-grid averages.\n- tables/configuration_ce_minus_er.csv: paired seed-averaged differences at each full parameter combination.\n- tables/parameter_manipulation_effects.csv: matched bias/input contrasts per E/I level and pooled over shared levels.\n- tables/configuration_sign_consistency.csv: finite-grid counts for positive and negative manipulation effects.\n- tables/polarity_asymmetry.csv: all-inhibitory minus all-excitatory contrasts.\n- manifest.json: exact sources, cached matrices, code hashes, settings and validation.\n']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);(OUT/'tables').mkdir()
    data,source_manifest,caches=load_surfaces()
    print('Verified surfaces loaded',len(data),flush=True)
    rng=np.random.default_rng(20261117)
    conditional,cells=compare_models(data,rng)
    print('Conditional comparisons complete',flush=True)
    effects,frequencies=manipulations(data,rng)
    polarity_results=polarity(data,rng)
    tested_input_optima(OUT)
    print('Paired input/bias/polarity effects complete',flush=True)
    for name,frame in [('conditional_ce_minus_er',conditional),('configuration_ce_minus_er',cells),('parameter_manipulation_effects',effects),('configuration_sign_consistency',frequencies),('polarity_asymmetry',polarity_results)]:
        frame.to_csv(OUT/'tables'/f'{name}.csv',index=False,float_format='%.12g')
    plots(conditional,effects,OUT)
    report(conditional,effects,frequencies,polarity_results,data,OUT)
    assert (conditional.n_seeds.isin([100,200])).all() and (effects.n_seeds.isin([100,200])).all()
    for source in source_manifest['sources']:assert sha(ROOT/source['path'])==source['sha256']
    manifest=dict(sources=source_manifest['sources'],verified_cache=caches,code_sha256=sha(Path(__file__)),bootstrap=NBOOT,seed=20261117,common_ei_percentages=COMMON,metrics=METRICS,filtering='Complete finite 96-configuration surfaces; shared E/I fractions only for across-family/pooled contrasts; no imputation',scopes='Exploratory pointwise/stratum analyses; seed as resampling unit; no multiplicity correction',n_source_score_rows=len(data),output_format='PNG only',validation='PASS: every cached surface matches plotted means; metric identities checked; pairing identities unique; original source hashes unchanged; seed counts 100/200 as expected')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
