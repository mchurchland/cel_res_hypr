#!/usr/bin/env python3
"""Longer held-out MC horizon check for the hardest cases of the local pilot."""
from pathlib import Path
import argparse,concurrent.futures,json,os,sys,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from tools.run_mc_horizon_pilot import MODELS,CASES,permutation_null,horizon_diagnostics
from architecture_mc_narma import read_cache,raw_from_cache,file_digest
from network_stats.audited_temporal_tasks import TaskSettings,simulate_batch,delay_targets,validated_ridge,squared_correlations

HORIZONS=[30,60,80,100,120,150,200,250,300,350,400,450,500]


def evaluate(args):
    parent,out,case,model,repeat=args;parent=Path(parent);out=Path(out)
    torch.set_num_threads(1);start=time.monotonic();slug=f'r{repeat:04d}__{case}__{model.lower().replace(" ","_")}'
    old=json.loads((parent/'cache'/f'{slug}.json').read_text());run,family=MODELS[model]
    m=json.loads((run/'manifest.json').read_text());s=TaskSettings(**m['config']['tasks'])
    cache=read_cache(run/'cache'/f'networks_{repeat:04d}.npz',m['experiment_id'])
    fraction,rho,leak,inp=CASES[case];fi=m['config']['fractions'].index(fraction)
    raw,_,_=raw_from_cache(cache,family,fi)
    source=ROOT/old['source_shard'];assert file_digest(source)==old['source_shard_sha256']
    d=pd.read_csv(source);r=d[(d.rho_target==rho)&(d.leak==leak)&(d.input_scale==inp)].iloc[0]
    test=5000;horizon=500;total=s.washout+s.train+s.validation+test
    seed=m['config']['seed']+m['config']['seed_stride']*repeat
    signal=np.random.default_rng(seed+31_000_003).uniform(-1,1,(total,1))
    with np.load(parent/'cache'/f'{slug}.npz') as z:
        old_states=z['states'];old_signal=z['mc_input'];old_scores=z['scores']
    np.testing.assert_array_equal(signal[:len(old_signal)],old_signal)
    states,gap=simulate_batch((raw*float(r.scaling_factor))[None],(cache['input_vector']*inp)[None],[leak],signal,s.washout,'cpu',torch.float64)
    x=states[0];state_error=float(abs(x[:len(old_states)]-old_states).max())
    if state_error>1e-12:raise ValueError(f'Extended input changed original state prefix: {state_error}')
    a=s.washout;b=a+s.train;c=b+s.validation;y=delay_targets(signal,horizon)
    pred,alpha,loss,scale=validated_ridge(x[a:b],y[a:b],x[b:c],y[b:c],x[c:],s.ridge_alphas)
    scores=squared_correlations(y[c:],pred)
    original_test=squared_correlations(y[c:c+s.test,:400],pred[:s.test,:400])
    prefix_error=float(abs(original_test-old_scores).max())
    if prefix_error>1e-8:raise ValueError(f'Extended scoring altered original test-prefix results: {prefix_error}')
    null=permutation_null(y[c:],pred,99,20261122+repeat*100003)
    diagnostics=horizon_diagnostics(scores,null,HORIZONS)
    np.savez_compressed(out/'cache'/f'{slug}.npz',scores=scores,null_scores=null,selected_alphas=alpha)
    result=dict(repeat_id=repeat,case=case,model=model,max_delay=horizon,test_steps=test,null_draws=99,
        unchanged_training=True,unchanged_validation=True,state_prefix_max_error=state_error,score_prefix_max_error=prefix_error,
        parent_case=old,diagnostics=diagnostics,seconds=time.monotonic()-start)
    (out/'cache'/f'{slug}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize(out,results,selected):
    rows=[];curves=[]
    for r in results:
        common={k:r[k] for k in ['repeat_id','model','case']}
        rows.extend(dict(**common,**v) for v in r['diagnostics'])
        slug=f'r{r["repeat_id"]:04d}__{r["case"]}__{r["model"].lower().replace(" ","_")}'
        with np.load(out/'cache'/f'{slug}.npz') as z:
            null_mean=z['null_scores'].mean(axis=0)
            curves.extend(dict(**common,delay=i+1,score=v,null_mean=null_mean[i]) for i,v in enumerate(z['scores']))
    d=pd.DataFrame(rows);d.to_csv(out/'horizon_by_realization.csv',index=False)
    s=d.groupby(['case','model','horizon']).agg(MC_mean=('MC','mean'),remaining_excess_mean=('remaining_excess','mean'),
        remaining_excess_max=('remaining_excess','max'),trailing_window_mean=('trailing_window_mean','mean'),
        trailing_null_95_mean=('trailing_null_95','mean')).reset_index();s.to_csv(out/'horizon_summary.csv',index=False)
    curves=pd.DataFrame(curves);curves.to_csv(out/'lag_curves.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['diagnostics','parent_case']} for r in results]).to_csv(out/'parity.csv',index=False)
    worst=s.groupby('horizon').remaining_excess_mean.max()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for model in sorted(s.model.unique()):
        v=s[s.model==model].groupby('horizon').remaining_excess_mean.max();axes[0].plot(v.index,v,marker='o',ms=3,label=model)
    axes[0].axhline(.1,color='.4',ls='--',lw=.8);axes[0].set_ylabel('Largest mean excess remaining');axes[0].legend(fontsize=8,ncol=3)
    for case,model in selected:
        z=curves[(curves.case==case)&(curves.model==model)].groupby('delay').agg(score=('score','mean'),null=('null_mean','mean'))
        # Smooth in disjoint 25-delay bins; retain unsmoothed data in CSV.
        block=z.groupby((z.index-1)//25).mean();axes[1].plot(block.index*25+13,block.score-block.null,lw=1,alpha=.7,label=f'{model} {case}')
    axes[1].axhline(0,color='.4',ls=':',lw=.8);axes[1].set(xlabel='Input delay / summation horizon',ylabel='25-delay mean recall above null')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    axes[1].set_yscale('symlog',linthresh=.001);fig.tight_layout();fig.savefig(out/'long_test_horizon_check.png',dpi=250);plt.close(fig)
    answer=dict(conditions=len(results),test_steps=5000,max_delay=500,worst_mean_remaining_excess={str(int(h)):float(v) for h,v in worst.items()},
        first_horizon_below_point1=next((int(h) for h,v in worst.items() if h>=60 and v<=.1),None),
        state_prefix_max_error=max(r['state_prefix_max_error'] for r in results),score_prefix_max_error=max(r['score_prefix_max_error'] for r in results))
    (out/'summary.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer),flush=True)

    return answer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=False);(out/'cache').mkdir()
    d=pd.read_csv(a.parent/'horizon_summary.csv');hard=d[d.horizon==100].nlargest(12,'remaining_excess_mean')
    selected=list(dict.fromkeys(list(zip(hard['case'],hard['model']))+[('balanced_peak',x) for x in ['WS binary','ER binary','ER empirical','ER gamma','WS empirical']]))
    settings=dict(selection='12 largest pilot model/case seed-mean residuals after delay 100, plus five balanced-peak comparisons',
        selected=selected,repeats=5,test_steps=5000,max_delay=500,null_draws=99,parent=str(a.parent.resolve()),
        source_sha256={name:file_digest(ROOT/name) for name in ['network_stats/audited_temporal_tasks.py','tools/run_mc_horizon_pilot.py','tools/extend_mc_horizon_test.py']})
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    jobs=[(str(a.parent.resolve()),str(out),case,model,r) for case,model in selected for r in range(5)]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(evaluate,j) for j in jobs]
        for f in concurrent.futures.as_completed(fs):
            r=f.result();results.append(r);print(f'{len(results)}/{len(jobs)} {r["model"]} {r["case"]} r{r["repeat_id"]}: {r["seconds"]:.1f}s',flush=True)
    summarize(out,results,selected)


if __name__=='__main__':main()
