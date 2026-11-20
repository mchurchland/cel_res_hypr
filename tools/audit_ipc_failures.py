#!/usr/bin/env python3
"""Reconstruct selected IPC failures; preserve production data and scoring."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/architecture-sweep-xdg-cache')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from scipy.linalg import svd
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from architecture_mc_narma import read_cache,raw_from_cache,file_digest
from network_stats.audited_temporal_tasks import TaskSettings,simulate_batch,validated_ridge
from network_stats.ipc_capacity import target_capacities

CASES=[
 ('CE_fixed_failure',34,'ce_empirical',0.,'fixed_reference',1.05,.6,.5),
 ('CE_same_seed_lower_gain',34,'ce_empirical',0.,'fixed_reference',.6,.6,.5),
 ('WS_own_failure',14,'ws_signed_empirical',.2,'own_radius',1.05,.6,.01),
 ('WS_own_good_validation_failure',36,'ws_signed_empirical',.2,'own_radius',1.05,.6,.1),
 ('ER_halfnormal_failure',77,'er_halfnormal',.4,'own_radius',1.05,.6,.1)]


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
 args=p.parse_args();run=args.run_dir.resolve();out=args.out_dir.resolve()
 if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a fresh audit directory')
 out.mkdir(parents=True,exist_ok=True);m=json.loads((run/'manifest.json').read_text());settings=TaskSettings(**m['config']['tasks']);all_results=[]
 torch.set_num_threads(1)
 for name,repeat,family,fraction,norm,rho,leak,inp in CASES:
  start=time.monotonic();fi=m['config']['fractions'].index(fraction)
  path=run/'shards'/f'repeat_{repeat:04d}__{family}__ei_{fi:02d}__{norm}.csv'
  meta=json.loads(Path(str(path)+'.json').read_text());assert file_digest(path)==meta['sha256']
  d=pd.read_csv(path);q=d[(d.rho_target==rho)&(d.leak==leak)&(d.input_scale==inp)];assert len(q)==1;r=q.iloc[0]
  cache=read_cache(run/'cache'/f'networks_{repeat:04d}.npz',m['experiment_id'])
  data=read_cache(run/'cache'/f'tasks_{repeat:04d}.npz',m['experiment_id'])
  raw,eigenvalues,radius=raw_from_cache(cache,family,fi);assert hashlib.sha256(raw.tobytes()).hexdigest()==r.raw_network_sha256
  state,gap=simulate_batch((raw*float(r.scaling_factor))[None],(cache['input_vector']*inp)[None],[leak],data['mc_input'],settings.washout,'cpu',torch.float64)
  x=state[0].astype(float);a=settings.washout;b=a+settings.train;c=b+settings.validation;y=data['ipc_targets']
  pred,chosen,loss,scale=validated_ridge(x[a:b],y[a:b],x[b:c],y[b:c],x[c:],settings.ridge_alphas)
  capacity=target_capacities(y[c:],pred)
  center=x[a:b].mean(0);xt,xv,xe=[(z-center)/scale for z in (x[a:b],x[b:c],x[c:])]
  _,singular,vt=svd(xt,full_matrices=False)
  modes=[z@vt.T for z in (xt,xv,xe)];rms=[np.sqrt(np.mean(z*z,axis=0)) for z in modes]
  mode_table=pd.DataFrame(dict(mode=np.arange(len(singular)),singular_value=singular,train_rms=rms[0],validation_rms=rms[1],test_rms=rms[2],test_over_train_rms=rms[2]/np.maximum(rms[0],1e-30)))
  mode_table.to_csv(out/f'{name}_state_modes.csv',index=False)
  per_alpha=[];reference_predictions=[];reference_losses=[];models={}
  for alpha in sorted(set(settings.ridge_alphas)|{10.,100.,1000.},reverse=True):
   model=Ridge(alpha=alpha,fit_intercept=True,solver='svd').fit(xt,y[a:b]);models[alpha]=model
   vp=model.predict(xv);tp=model.predict(xe);vl=np.mean((vp-y[b:c])**2,axis=0);co=np.linalg.norm(model.coef_,axis=1)
   per_alpha.append(dict(alpha=alpha,mean_validation_mse=float(vl.mean()),IPC=float(target_capacities(y[c:],tp).sum()),max_abs_test_prediction=float(abs(tp).max()),max_coefficient_norm=float(co.max())))
   if alpha in settings.ridge_alphas:reference_predictions.append(tp);reference_losses.append(vl)
  ordered=sorted(settings.ridge_alphas,reverse=True);choice=np.argmin(reference_losses,axis=0)
  reference=np.stack(reference_predictions,axis=1)[:,choice,np.arange(y.shape[1])]
  independent_error=float(abs(pred-reference).max())
  if not np.allclose(pred,reference,rtol=1e-6,atol=1e-6):raise ValueError(f'Library-reference prediction mismatch: {independent_error}')
  pd.DataFrame(per_alpha).to_csv(out/f'{name}_penalty_sensitivity.csv',index=False)
  policies=[]
  for floor in [1e-8,1e-6,1e-4,.01,1.,10.]:
   penalties=[p for p in sorted(models,reverse=True) if p>=floor and (p in settings.ridge_alphas or p in [10.,100.,1000.])]
   val_losses=np.stack([np.mean((models[p].predict(xv)-y[b:c])**2,axis=0) for p in penalties]);idx=np.argmin(val_losses,axis=0)
   selected=np.stack([models[p].predict(xe) for p in penalties],axis=1)[:,idx,np.arange(y.shape[1])]
   policies.append(dict(minimum_alpha=floor,IPC=float(target_capacities(y[c:],selected).sum()),mean_validation_mse=float(val_losses[idx,np.arange(y.shape[1])].mean())))
  pd.DataFrame(policies).to_csv(out/f'{name}_validation_selected_floor_sensitivity.csv',index=False)
  terms=pd.DataFrame(dict(target=np.arange(y.shape[1]),degree=data['ipc_exponents'].sum(1),alpha=chosen,validation_mse=loss,test_mse=np.mean((pred-y[c:])**2,axis=0),target_second_moment=np.mean(y[c:]**2,axis=0),capacity=capacity,max_abs_prediction=np.max(abs(pred),axis=0)))
  terms.to_csv(out/f'{name}_targets.csv',index=False)
  state_mean=x.mean(axis=1);regions={}
  for label,aa,bb in [('train',a,b),('validation',b,c),('test',c,len(x))]:
   u=state_mean[aa:bb]
   regions[label]=dict(network_mean_average=float(u.mean()),network_mean_min=float(u.min()),network_mean_max=float(u.max()),fraction_network_mean_positive=float((u>0).mean()),mean_state_variance=float(np.var(x[aa:bb],axis=0).mean()))
  per_step=np.mean((pred-y[c:])**2,axis=1)
  pd.DataFrame(dict(time=np.arange(len(x)),network_mean=state_mean)).to_csv(out/f'{name}_network_mean.csv',index=False)
  fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
  axes[0].plot(np.arange(len(x)),state_mean,color='#0072b2',lw=1);axes[0].axhline(0,color='.5',lw=.7)
  axes[0].set_ylabel('Mean neuron state')
  axes[1].plot(np.arange(c,len(x)),per_step,color='#d55e00',lw=1);axes[1].set_yscale('log');axes[1].set_ylabel('Test MSE across targets');axes[1].set_xlabel('Time step')
  for ax in axes:
   ax.axvspan(0,a,color='.6',alpha=.1);ax.axvspan(a,b,color='#009e73',alpha=.07);ax.axvspan(b,c,color='#e69f00',alpha=.08);ax.axvspan(c,len(x),color='#d55e00',alpha=.06)
   for t in [a,b,c]:ax.axvline(t,color='.4',ls=':',lw=.7)
   ax.spines[['top','right']].set_visible(False)
  for aa,bb,label in [(0,a,'Washout'),(a,b,'Train'),(b,c,'Validation'),(c,len(x),'Test')]:axes[0].text((aa+bb)/2,1.01,label,transform=axes[0].get_xaxis_transform(),ha='center',fontsize=8)
  fig.suptitle(f'{name}: CPU reconstruction');fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=250,bbox_inches='tight');plt.close(fig)
  # Exact decomposition of the worst target's centered prediction into modes.
  j=int(np.argmin(capacity));model=models[float(chosen[j])];weights=vt@model.coef_[j]
  contribution=modes[2]*weights[None,:];mode_table['worst_target_coefficient']=weights;mode_table['worst_target_test_contribution_rms']=np.sqrt(np.mean(contribution**2,axis=0));mode_table.to_csv(out/f'{name}_state_modes.csv',index=False)
  result=dict(case=name,repeat=repeat,family=family,negative_fraction=fraction,normalization=norm,rho_target=rho,leak=leak,input_scale=inp,
   production_IPC=float(r.IPC),cpu_IPC=float(capacity.sum()),cpu_to_production_ratio=float(capacity.sum()/r.IPC),
   post_scaling_radius=float(r.post_scaling_radius),cpu_washout_gap=float(gap[0]),train_rms_scale=scale,
   mean_validation_mse=float(loss.mean()),max_abs_test_prediction=float(abs(pred).max()),max_abs_target=float(abs(y[c:]).max()),
   min_target_second_moment=float(np.mean(y[c:]**2,axis=0).min()),max_target_second_moment=float(np.mean(y[c:]**2,axis=0).max()),
   independent_sklearn_prediction_max_difference=independent_error,smallest_singular_value=float(singular.min()),largest_singular_value=float(singular.max()),
   n_modes_test_over_train_gt100=int((mode_table.test_over_train_rms>100).sum()),regions=regions,penalty_sensitivity=per_alpha,validation_selected_floor_sensitivity=policies,seconds=time.monotonic()-start)
  all_results.append(result);print(json.dumps(result),flush=True)
 (out/'audit.json').write_text(json.dumps(dict(run=str(run),experiment_id=m['experiment_id'],cpu_environment=dict(numpy=np.__version__,torch=torch.__version__),original_results_modified=False,script_sha256=file_digest(Path(__file__)),cases=all_results),indent=2)+'\n')
 print(out)

if __name__=='__main__':main()
