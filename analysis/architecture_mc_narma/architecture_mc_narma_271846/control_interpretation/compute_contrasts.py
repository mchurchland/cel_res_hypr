from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd
import argparse
parser=argparse.ArgumentParser(description='Reproduce paired control contrasts for architecture run 271846. Run from the repository root.')
parser.add_argument('--out-dir',type=Path,required=True)
args=parser.parse_args()
base=Path('analysis/architecture_mc_narma/architecture_mc_narma_271846');out=args.out_dir
if out.exists() and any(out.iterdir()):
 raise FileExistsError('Choose a new output directory; existing results are preserved')
out.mkdir(parents=True,exist_ok=True)
source=Path('results/architecture_mc_narma/architecture_mc_narma_271846/architecture_mc_narma.ALL.csv')
start=time.monotonic();h=hashlib.sha256()
with source.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
expected=json.loads((base/'manifest.json').read_text())
assert h.hexdigest()==expected['source_sha256']
for name,sha in expected['outputs'].items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha,name
cols=['experiment_id','repeat_id','family','negative_fraction','normalization','rho_target','leak','input_scale','MC','MC_extended','MC_tail','NARMA10_test_nmse','NARMA10_val_nmse','MC_washout_initialization_rms_gap','NARMA10_washout_initialization_rms_gap','post_scaling_radius','scaling_factor','raw_radius','NARMA10_saturation_test','MC_saturation_test']
d=pd.read_csv(source,usecols=cols);print('Loaded',len(d),'seconds',time.monotonic()-start,flush=True)
keys=['repeat_id','family','negative_fraction','normalization','rho_target','leak','input_scale']
assert len(d)==432000 and not d.duplicated(keys).any()
assert np.isfinite(d.select_dtypes(include='number')).all().all()
assert (d.groupby(['repeat_id','family','negative_fraction','normalization']).size()==24).all()
assert len(d.repeat_id.unique())==100
weights=np.random.default_rng(20261119).multinomial(100,np.full(100,.01),size=2000)/100
contrasts=[]
comparisons=[('ce_empirical',f) for f in sorted(d.family.unique()) if f!='ce_empirical']
comparisons += [('ce_shuffled','er_signed_empirical'),('ce_degree_rewired','er_signed_empirical'),('ce_binary','er_binary'),('ce_halfnormal','er_halfnormal'),('er_signed_empirical','er_binary')]
for norm,gn in d.groupby('normalization'):
 for frac,g in gn.groupby('negative_fraction'):
  for input_group,gp in [('all',g),('0.01',g[g.input_scale==.01]),('0.1',g[g.input_scale==.1])]:
   agg=gp.groupby(['repeat_id','family'])[['MC','NARMA10_test_nmse','MC_extended','MC_tail']].mean()
   for f1,f2 in comparisons:
    for metric,direction in [('MC',1),('NARMA10_test_nmse',-1),('MC_extended',1),('MC_tail',1)]:
     a=agg.xs(f1,level='family')[metric].reindex(range(100));b=agg.xs(f2,level='family')[metric].reindex(range(100))
     delta=direction*(a.to_numpy()-b.to_numpy());ci=np.quantile(weights@delta,[.025,.975])
     contrasts.append(dict(normalization=norm,negative_fraction=frac,input_group=input_group,metric=metric,
       first=f1,second=f2,advantage=delta.mean(),ci_low=ci[0],ci_high=ci[1],n_realizations=100,n_configurations=24 if input_group=='all' else 6))
c=pd.DataFrame(contrasts);c.to_csv(out/'paired_mean_contrasts.csv',index=False)
for task in ['MC','NARMA10']:
 gap=d[f'{task}_washout_initialization_rms_gap']
 d[f'{task}_gap_gt_1e3']=gap>1e-3
 d[f'{task}_gap_gt_1e6']=gap>1e-6
summary=d.groupby(['normalization','family','negative_fraction']).agg(
 observations=('MC','size'),MC_gap_gt_1e3=('MC_gap_gt_1e3','mean'),
 NARMA_gap_gt_1e3=('NARMA10_gap_gt_1e3','mean'),MC_gap_gt_1e6=('MC_gap_gt_1e6','mean'),
 NARMA_gap_gt_1e6=('NARMA10_gap_gt_1e6','mean'),
 max_MC_gap=('MC_washout_initialization_rms_gap','max'),max_NARMA_gap=('NARMA10_washout_initialization_rms_gap','max'),
 fraction_radius_gt1=('post_scaling_radius',lambda x:(x>1).mean()),max_radius=('post_scaling_radius','max'),
 max_NARMA_NMSE=('NARMA10_test_nmse','max'),mean_MC_tail=('MC_tail','mean')).reset_index()
summary.to_csv(out/'initialization_diagnostics.csv',index=False)
worst=d.nlargest(15,'NARMA10_test_nmse');worst.to_csv(out/'largest_narma_errors.csv',index=False)
print('KEY CONTRASTS',flush=True)
q=c[(c.negative_fraction.isin([0,.2,.5]))&(c['first']=='ce_empirical')&(c['second'].isin(['ce_shuffled','ce_degree_rewired','er_signed_empirical','er_binary']))&(c.input_group=='all')&(c.metric.isin(['MC','NARMA10_test_nmse']))]
print(q.round(4).to_string(index=False),flush=True)
print('WEAK20',flush=True)
print(c[(c.negative_fraction==.2)&(c['first']=='ce_empirical')&(c['second'].isin(['ce_shuffled','er_signed_empirical']))&(c.input_group=='0.01')].round(5).to_string(index=False),flush=True)
print('DYNAMICS',flush=True)
print(summary[(summary.negative_fraction.isin([0,.2,.5]))&summary.family.isin(['ce_empirical','ce_shuffled','er_signed_empirical','er_binary'])].round(4).to_string(index=False),flush=True)
print('WORST',flush=True);print(worst.head(5).to_string(index=False),flush=True)
write=dict(source=str(source),source_sha256=h.hexdigest(),rows=len(d),repeats=100,grid_size=24,
 bootstrap_replicates=2000,bootstrap_seed=20261119,bootstrap_unit='whole paired realization; average grid within realization before bootstrap',
 exploratory_initialization_cutoffs=[1e-3,1e-6],filters='None: all finite scores retained; task-stream replacement is inherited from recorded experiment protocol',
 largest_narma_error=float(d.NARMA10_test_nmse.max()),elapsed_seconds=time.monotonic()-start)
(out/'verification.json').write_text(json.dumps(write,indent=2)+'\n')
