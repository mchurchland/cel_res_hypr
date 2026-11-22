#!/usr/bin/env python3
"""Local CPU MC-horizon pilot using frozen archived networks and task streams.

Reuses the production simulator and sklearn readout, never edits original runs.
Stores states for follow-up readout checks and a held-out target-permutation null.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mc-horizon-matplotlib')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from architecture_mc_narma import read_cache,raw_from_cache,file_digest
from network_stats.audited_temporal_tasks import TaskSettings,simulate_batch,delay_targets,validated_ridge,squared_correlations

BASE=ROOT/'results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871'
WS=ROOT/'results/ws_binary_halfnormal/ws_binary_halfnormal_272279'
ER=ROOT/'results/er_weight_distribution/er_weights_274402'
MODELS={
    'CE empirical':(BASE,'ce_empirical'),
    'CE shuffled':(BASE,'ce_shuffled'),
    'ER binary':(BASE,'er_binary'),
    'ER empirical':(BASE,'er_signed_empirical'),
    'ER gamma':(ER,'gamma'),
    'WS binary':(WS,'ws_binary'),
    'WS empirical':(BASE,'ws_signed_empirical'),
    'WS half-normal':(WS,'ws_halfnormal'),
}
CASES={
    'balanced_peak':(.5,.95,1.,.01),
    'imbalanced_peak':(.2,.95,1.,.01),
    'slow_leak':(.5,.95,.6,.01),
    'high_radius':(.5,1.05,1.,.01),
    'high_radius_slow_leak':(.5,1.05,.6,.01),
}
HORIZONS=[30,60,80,100,120,150,200,250,300,350,400]


def permutation_null(target,prediction,count,seed):
    """Shuffle held-out target rows jointly across lags, keeping fits fixed.

    This estimates finite-test chance correlations, not the sampling interval
    of the reservoir's true memory curve and not a refitted null model.
    """
    y=target-target.mean(axis=0);p=prediction-prediction.mean(axis=0)
    yn=np.linalg.norm(y,axis=0);pn=np.linalg.norm(p,axis=0)
    y=np.divide(y,yn,out=np.zeros_like(y),where=yn>0)
    p=np.divide(p,pn,out=np.zeros_like(p),where=pn>0)
    rng=np.random.default_rng(seed)
    return np.asarray([np.einsum('ij,ij->j',y[rng.permutation(len(y))],p)**2 for _ in range(count)])


def horizon_diagnostics(scores,null_scores,horizons,window=25,tolerance=.1):
    scores=np.asarray(scores,float);null_scores=np.asarray(null_scores,float)
    if scores.ndim!=1 or null_scores.ndim!=2 or null_scores.shape[1]!=len(scores):raise ValueError('Invalid curve/null shape')
    if not np.isfinite(scores).all() or not np.isfinite(null_scores).all():raise ValueError('Nonfinite scores')
    result=[]
    for h in horizons:
        if not window<=h<=len(scores):raise ValueError('Invalid horizon')
        late=scores[h-window:h].mean();late_null=null_scores[:,h-window:h].mean(axis=1)
        remaining=scores[h:].sum();remaining_null=null_scores[:,h:].sum(axis=1)
        excess=remaining-remaining_null.mean()
        result.append(dict(horizon=h,MC=float(scores[:h].sum()),remaining_raw=float(remaining),
            remaining_null_mean=float(remaining_null.mean()),remaining_excess=float(excess),
            trailing_window_mean=float(late),trailing_null_95=float(np.quantile(late_null,.95)),
            trailing_near_null=bool(late<=np.quantile(late_null,.95)),
            estimated_remaining_le_01=bool(excess<=tolerance),
            # Practical pilot criterion, not a formal equivalence/confidence test.
            practical_pass=bool(excess<=tolerance and late<=np.quantile(late_null,.95))))
    return result


def one_case(args):
    repeat,model,case,out,max_delay,null_draws=args
    out=Path(out);slug=f'r{repeat:04d}__{case}__{model.lower().replace(" ","_")}'
    done=out/'cache'/f'{slug}.json'
    if done.exists():return json.loads(done.read_text())
    torch.set_num_threads(1);started=time.monotonic()
    run,family=MODELS[model];m=json.loads((run/'manifest.json').read_text());s=TaskSettings(**m['config']['tasks'])
    if max_delay>s.washout:raise ValueError('Pilot delay exceeds available pre-training history')
    fraction,rho,leak,inp=CASES[case];fi=m['config']['fractions'].index(fraction)
    net_path=run/'cache'/f'networks_{repeat:04d}.npz';task_path=run/'cache'/f'tasks_{repeat:04d}.npz'
    cache=read_cache(net_path,m['experiment_id']);data=read_cache(task_path,m['experiment_id'])
    raw,eigen,radius=raw_from_cache(cache,family,fi)
    shard=run/'shards'/f'repeat_{repeat:04d}__{family}__ei_{fi:02d}__own_radius.csv'
    sm=json.loads(Path(str(shard)+'.json').read_text());assert file_digest(shard)==sm['sha256']
    archived=pd.read_csv(shard);r=archived[(archived.rho_target==rho)&(archived.leak==leak)&(archived.input_scale==inp)]
    assert len(r)==1;r=r.iloc[0]
    assert hashlib.sha256(raw.tobytes()).hexdigest()==r.raw_network_sha256
    assert hashlib.sha256(cache['input_vector'].tobytes()).hexdigest()==r.input_vector_sha256
    factor=float(r.scaling_factor);assert np.isclose(factor,rho/radius,rtol=1e-12)
    states,gap=simulate_batch((raw*factor)[None],(cache['input_vector']*inp)[None],[leak],data['mc_input'],s.washout,'cpu',torch.float64)
    x=states[0];a=s.washout;b=a+s.train;c=b+s.validation;y=delay_targets(data['mc_input'],max_delay)
    pred,alphas,loss,scale=validated_ridge(x[a:b],y[a:b],x[b:c],y[b:c],x[c:],s.ridge_alphas)
    scores=squared_correlations(y[c:],pred)
    # Independently check that extending targets leaves old-delay scores intact.
    short_pred,*_=validated_ridge(x[a:b],y[a:b,:60],x[b:c],y[b:c,:60],x[c:],s.ridge_alphas)
    short=squared_correlations(y[c:,:60],short_pred)
    prefix_error=float(abs(short-scores[:60]).max())
    if prefix_error>1e-8:raise ValueError(f'Changing target count affected existing delays: {prefix_error}')
    old=np.asarray([r[f'MC_delay_{i:02d}'] for i in range(1,61)],float)
    null=permutation_null(y[c:],pred,null_draws,20261122+repeat*100003)
    diagnostics=horizon_diagnostics(scores,null,[h for h in HORIZONS if h<=max_delay])
    np.savez_compressed(out/'cache'/f'{slug}.npz',scores=scores,null_scores=null,selected_alphas=alphas,
        validation_mse=loss,states=x,mc_input=data['mc_input'],prediction=pred)
    result=dict(repeat_id=repeat,model=model,case=case,negative_fraction=fraction,rho_target=rho,leak=leak,input_scale=inp,
        experiment_id=m['experiment_id'],source_shard=str(shard.relative_to(ROOT)),source_shard_sha256=sm['sha256'],
        network_cache_sha256=file_digest(net_path),task_cache_sha256=file_digest(task_path),
        source_MC30=float(r.MC),source_MC60=float(r.MC_extended),cpu_MC30=float(scores[:30].sum()),
        cpu_MC60=float(scores[:60].sum()),cpu_MCH=float(scores.sum()),final_lag_score=float(scores[-1]),
        archived_curve_max_abs_error=float(abs(old-short).max()),target_extension_prefix_max_abs_error=prefix_error,
        washout_gap=float(gap[0]),train_state_rms=scale,diagnostics=diagnostics,
        seconds=time.monotonic()-started)
    done.write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize(out,results,max_delay):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];curves=[]
    for r in results:
        slug=f'r{r["repeat_id"]:04d}__{r["case"]}__{r["model"].lower().replace(" ","_")}'
        with np.load(out/'cache'/f'{slug}.npz') as z:score=z['scores'];null=z['null_scores']
        for h in r['diagnostics']:rows.append(dict(repeat_id=r['repeat_id'],model=r['model'],case=r['case'],**h))
        curves.extend(dict(repeat_id=r['repeat_id'],model=r['model'],case=r['case'],delay=i+1,score=float(v),
            null_mean=float(null[:,i].mean()),null_95=float(np.quantile(null[:,i],.95))) for i,v in enumerate(score))
    diag=pd.DataFrame(rows);curve=pd.DataFrame(curves)
    diag.to_csv(out/'horizon_by_realization.csv',index=False);curve.to_csv(out/'lag_curves.csv',index=False)
    summary=diag.groupby(['case','model','horizon']).agg(MC_mean=('MC','mean'),remaining_excess_mean=('remaining_excess','mean'),
        remaining_excess_max=('remaining_excess','max'),trailing_window_mean=('trailing_window_mean','mean'),
        practical_pass_fraction=('practical_pass','mean')).reset_index()
    summary.to_csv(out/'horizon_summary.csv',index=False)
    baseline=pd.DataFrame([{k:v for k,v in r.items() if k!='diagnostics'} for r in results]);baseline.to_csv(out/'parity_and_runtime.csv',index=False)
    palette=dict(zip(MODELS,['#D55E00','#E69F00','#0072B2','#7A3E8E','#007D80','#555555','#332288','#88CCEE']))
    fig,axes=plt.subplots(len(CASES),1,figsize=(11,3*len(CASES)),sharex=True)
    for ax,case in zip(axes,CASES):
        for model in MODELS:
            a=curve[(curve.case==case)&(curve.model==model)].groupby('delay').agg(mean=('score','mean'),null=('null_mean','mean'))
            ax.plot(a.index,a['mean'],color=palette[model],lw=1.4,label=model)
        ax.axhline(.002,color='.4',ls=':',lw=.7);ax.set(title=case,ylabel='Held-out recall r²',ylim=(-.02,1.03))
        for h in [60,100,150,200,300]:
            if h<max_delay:ax.axvline(h,color='.8',lw=.5)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Input delay');axes[0].legend(ncol=4,fontsize=8,loc='upper right')
    fig.tight_layout();fig.savefig(out/'memory_delay_profiles.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for model in MODELS:
        a=summary[summary.model==model].groupby('horizon').remaining_excess_mean.max()
        ax.plot(a.index,a,marker='o',ms=3,color=palette[model],label=model)
    ax.axhline(.1,color='.3',ls='--',lw=.8,label='0.1 MC diagnostic tolerance');ax.axhline(0,color='.7',lw=.5)
    ax.set(xlabel='Summation horizon',ylabel=f'Largest mean excess beyond horizon (through {max_delay})',title='Worst sampled setting per model; test-target null subtracted')
    ax.legend(fontsize=8,ncol=2);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'remaining_memory_by_horizon.png',dpi=250);plt.close(fig)
    final=curve[curve.delay>max_delay-25].groupby(['case','model']).agg(score=('score','mean'),null=('null_mean','mean')).reset_index()
    final.to_csv(out/'final_25_lags.csv',index=False)
    suggested=summary.groupby('horizon').remaining_excess_mean.max()
    smallest=next((int(h) for h,v in suggested.items() if h>=60 and v<=.1),None)
    report=f'''# Local MC-horizon pilot

{len(results)} CPU conditions: {len(MODELS)} model variants, {len(CASES)} low-input settings,
{baseline.repeat_id.nunique()} paired archived realizations; delays 1–{max_delay}.
Original model matrices, input projections, streams, train/validation/test split,
ridge grid and float64 protocol are retained. No original results were modified.
NARMA10 was not rerun. States and predictions are cached for follow-up checks.

## Horizon sensitivity

The smallest tested horizon whose worst case/model **seed-mean** estimated
remaining memory is at most 0.1 MC, through delay {max_delay}, is **{smallest}**.
This is an operational pilot tolerance, not proof of an infinite-horizon capacity.
Inspect final-lag profiles and individual-realization variation before adopting it.

| Horizon | Worst case/model mean excess remaining |
|---|---:|
'''+''.join(f'| {int(h)} | {v:.6f} |\n' for h,v in suggested.items())+f'''
## Interpretation and limitations

- `MC` columns are ordinary sums of squared correlations. Null subtraction is
  a separate horizon diagnostic; it does not silently redefine production MC.
- For each fitted readout, {results[0]['diagnostics'][-1]['horizon']} lags are compared with
  held-out target-row permutations. The null keeps predictions and training fixed.
  It measures chance association, not between-network uncertainty or a refitted null.
- Finite-test positive correlations can keep the raw cumulative sum increasing
  after recoverable memory has decayed. Extra raw area alone is not extra memory.
- The practical per-realization flag additionally requires the preceding 25-lag
  block to lie below its permutation-null 95th percentile. It is a descriptive
  screening rule, not a multiple-comparison-adjusted equivalence test.
- The recommended screen uses averages over the sampled seeds; individual
  conditions can differ. This is not the full 24-setting, 11-fraction, 100-seed grid.
- Max target-extension prefix error: {baseline.target_extension_prefix_max_abs_error.max():.3g}.
  Max CPU/archive prefix error: {baseline.archived_curve_max_abs_error.max():.3g}.
  Above-unit-radius trajectories may diverge numerically between CPU and GPU;
  such discrepancies are recorded rather than hidden.

## Outputs

- [Delay profiles](memory_delay_profiles.png)
- [Remaining-memory diagnostic](remaining_memory_by_horizon.png)
- [Horizon summaries](horizon_summary.csv)
- [Per-realization diagnostics](horizon_by_realization.csv)
- [Late-delay scores](final_25_lags.csv)
- [CPU/archive parity](parity_and_runtime.csv)

Settings: `balanced_peak` and `imbalanced_peak`: radius .95, leak 1, input .01,
E/I .5 and .2. `slow_leak`: E/I .5, radius .95, leak .6, input .01.
`high_radius` and `high_radius_slow_leak`: E/I .5, radius 1.05,
leak 1 and .6 respectively; input .01. These emphasize candidate long-memory regimes.
'''
    (out/'REPORT.md').write_text(report)
    return dict(smallest_tested_horizon_by_mean_excess=smallest,worst_mean_excess_by_horizon={str(h):float(v) for h,v in suggested.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--max-delay',type=int,default=400)
    p.add_argument('--null-draws',type=int,default=199);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.repeats<=100 or not 60<=a.max_delay<=500 or a.null_draws<20 or a.workers<1:raise ValueError('Invalid pilot settings')
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True);(out/'cache').mkdir(exist_ok=True)
    settings=dict(repeats=a.repeats,max_delay=a.max_delay,null_draws=a.null_draws,models=list(MODELS),cases=CASES,
        sources={str(root.relative_to(ROOT)):file_digest(root/'manifest.json') for root in [BASE,WS,ER]},
        source_sha256={name:file_digest(ROOT/name) for name in ['network_stats/audited_temporal_tasks.py','tools/run_mc_horizon_pilot.py']})
    mf=out/'settings.json'
    if mf.exists() and json.loads(mf.read_text())!=settings:raise ValueError('Changed pilot; choose a new output folder')
    if not mf.exists():mf.write_text(json.dumps(settings,indent=2)+'\n')
    jobs=[(r,m,c,str(out),a.max_delay,a.null_draws) for r in range(a.repeats) for c in CASES for m in MODELS]
    results=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one_case,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            r=future.result();results.append(r)
            print(f'{len(results)}/{len(jobs)} {r["model"]} {r["case"]} seed {r["repeat_id"]}: MC60={r["cpu_MC60"]:.3f}, MCH={r["cpu_MCH"]:.3f}, last={r["final_lag_score"]:.4f}, {r["seconds"]:.2f}s',flush=True)
    answer=summarize(out,results,a.max_delay)
    (out/'summary.json').write_text(json.dumps(dict(**answer,conditions=len(results),elapsed_seconds=time.monotonic()-start,settings=settings),indent=2)+'\n')
    print(json.dumps(answer),flush=True)


if __name__=='__main__':main()
