"""Pairing, source preservation and two-model-only WS submission tests."""
from dataclasses import asdict
from pathlib import Path
import json,os,subprocess,tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import architecture_mc_narma as r
from network_stats.architecture_sweep import make_assets,raw_networks,spectrum,FAMILIES,rng_for
from network_stats.audited_temporal_tasks import TaskSettings,make_task_data
from network_stats.ipc_capacity import IPCSettings,add_ipc_data
from network_stats.ws_weight_controls import extend_parent_cache,WS_FAMILIES
from tools.analyze_architecture_mc_narma import attach_comparison_parent,summarize,bootstrap_weights


def fixture():
    n=12;known=np.zeros((n,n));unknown=known.copy()
    for i in range(n):
        for j,offset in enumerate((1,2,4,7)):
            if j==3:unknown[i,(i+offset)%n]=i+1
            else:known[i,(i+offset)%n]=(1 if j!=1 else -1)*(i*3+j+1)
    return known,unknown


def parent_cache(seed,fractions):
    a=make_assets(*fixture(),seed,2)
    cache={k:a[k] for k in ['n_nodes','reference_fraction','directed_swaps','rewired_edge_retention','raw_ce_mean_magnitude']}
    cache['input_vector']=rng_for(seed,'shared all-node input projection').uniform(-1,1,12)
    for i,f in enumerate(list(fractions)+[float(a['reference_fraction'])]):
        for family,w in raw_networks(a,f,seed).items():
            e=np.argwhere(w!=0);ev,radius=spectrum(w);p=f'{family}__{i}'
            cache.update({p+'__edges':e,p+'__values':w[e[:,0],e[:,1]],p+'__eigenvalues':ev,p+'__radius':np.array(radius)})
    return cache


class WSControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fractions=[0.,.5,1.];cls.seed=9876;cls.parent=parent_cache(cls.seed,cls.fractions)

    def test_masks_signs_exact_multisets_and_spectra(self):
        original={k:v.copy() for k,v in self.parent.items()}
        child=extend_parent_cache(self.parent,self.fractions,self.fractions,self.seed)
        for i in range(4):
            empirical=r.raw_from_cache(child,'ws_signed_empirical',i)[0]
            for family,paired in [('ws_binary','ce_binary'),('ws_halfnormal','ce_halfnormal')]:
                w,eigen,radius=r.raw_from_cache(child,family,i)
                np.testing.assert_array_equal(np.sign(w),np.sign(empirical))
                self.assertFalse(np.diag(w).any());self.assertEqual(np.count_nonzero(w),48)
                np.testing.assert_array_equal(np.sort(w[w!=0]),np.sort(self.parent[f'{paired}__{i}__values']))
                self.assertAlmostEqual(spectrum(w)[1],radius)
                np.testing.assert_array_equal(child[f'{family}__{i}__edges'],self.parent[f'ws_signed_empirical__{i}__edges'])
        for key,value in original.items():np.testing.assert_array_equal(self.parent[key],value)

    def test_fraction_subset_reference_and_signed_weight_mapping_guard(self):
        child=extend_parent_cache(self.parent,self.fractions,[1.,0.],self.seed)
        for new,old in [(0,2),(1,0),(2,3)]:
            np.testing.assert_array_equal(child[f'ws_signed_empirical__{new}__values'],self.parent[f'ws_signed_empirical__{old}__values'])
        with self.assertRaises(AssertionError):extend_parent_cache(self.parent,self.fractions,self.fractions,self.seed+1)

    def build_parent(self,root):
        root.mkdir();tasks=TaskSettings(washout=10,train=60,validation=20,test=30,narma_burnin=40,mc_max_delay=3,mc_diagnostic_delay=5)
        config=dict(families=list(FAMILIES),fractions=self.fractions,normalizations=['own_radius'],rhos=[.6],leaks=[1.],inputs=[.1],
                    repeats=2,seed=self.seed,seed_stride=100000,tasks=asdict(tasks),state_dtype='float64',swaps_per_edge=2,ws_probability=.1,ipc=IPCSettings(3,2).definition())
        records=[]
        for repeat in range(2):
            for fi,f in enumerate(self.fractions):
                for j,family in enumerate(FAMILIES):
                    records.append(dict(experiment_id='parent',repeat_id=repeat,family=family,negative_fraction=f,normalization='own_radius',rho_target=.6,leak=1.,input_scale=.1,MC=5.+j,IPC=3.+j,NARMA10_test_nmse=1/(j+2)))
        frame=pd.DataFrame(records);frame.to_csv(root/'architecture_mc_narma.ALL.csv',index=False)
        old=dict(experiment_id='parent',protocol=r.IPC_PROTOCOL,config=config,source_sha256={name:r.file_digest(r.ROOT/name) for name in r.SOURCES+r.DATA})
        (root/'manifest.json').write_text(json.dumps(old));r.write_json(str(root/'architecture_mc_narma.ALL.csv')+'.json',dict(experiment_id='parent',sha256=r.file_digest(root/'architecture_mc_narma.ALL.csv')))
        for repeat in range(2):
            seed=self.seed+100000*repeat
            cache=self.parent if repeat==0 else parent_cache(seed,self.fractions)
            r.write_cache(root/'cache'/f'networks_{repeat:04d}.npz',cache,'parent')
            r.write_cache(root/'cache'/f'tasks_{repeat:04d}.npz',add_ipc_data(make_task_data(tasks,seed),tasks,IPCSettings(3,2)),'parent')
        return old,frame

    def args(self,parent,child):
        return r.parser().parse_args(['prepare','--run-dir',str(child),'--comparison-parent',str(parent),
            '--families',*WS_FAMILIES,'--fractions','0','.5','1','--normalizations','own_radius',
            '--rhos','.6','--leaks','1','--inputs','.1','--repeats','2','--seed',str(self.seed),'--swaps-per-edge','2',
            '--washout','10','--train','60','--validation','20','--test','30','--narma-burnin','40',
            '--mc-max-delay','3','--mc-diagnostic-delay','5','--with-ipc','--ipc-max-delay','3','--ipc-max-degree','2'])

    def test_prepare_stream_pairing_and_only_two_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';child=root/'child';old,_=self.build_parent(parent)
            r.prepare(self.args(parent,child));m=r.load_run(child)
            self.assertEqual(m['config']['families'],list(WS_FAMILIES));self.assertEqual(m['expected_evaluations'],12)
            with patch.object(r,'make_task_data',side_effect=AssertionError('Must reuse paired input streams')):
                data=r.tasks_cached(child,m,0)
            original=r.read_cache(parent/'cache/tasks_0000.npz','parent')
            for key in original:np.testing.assert_array_equal(data[key],original[key])
            with patch.object(r,'make_assets',side_effect=AssertionError('Must reuse existing topology')):
                cache=r.networks_cached(child,m,0)
            for family in WS_FAMILIES:
                rows=r.condition_rows(m,cache,data,0,family,1,'own_radius',r.grid(m),'cpu',1)
                self.assertEqual(rows[0]['family'],family);self.assertEqual(rows[0]['MC_NARMA_reused'],0)
                r.validate_rows(rows,m,0,family,1,'own_radius')
            bad=self.args(parent,root/'bad');bad.ipc_max_delay=2
            with self.assertRaisesRegex(ValueError,'different IPC definition'):r.prepare(bad)

    def test_combined_analysis_uses_parent_without_simulation_or_duplicate_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';child=root/'child';old,original=self.build_parent(parent)
            r.prepare(self.args(parent,child));m=r.load_run(child)
            records=[]
            for repeat in range(2):
                for f in self.fractions:
                    for family in WS_FAMILIES:
                        records.append(dict(experiment_id=m['experiment_id'],repeat_id=repeat,family=family,negative_fraction=f,normalization='own_radius',rho_target=.6,leak=1.,input_scale=.1,MC=6.,IPC=4.,NARMA10_test_nmse=.2))
            data=pd.DataFrame(records);combined,manifest,provenance=attach_comparison_parent(data,m)
            self.assertEqual(len(combined),72);self.assertEqual(len(manifest['config']['families']),12)
            self.assertEqual(provenance['parent_rows_used'],60);self.assertEqual(provenance['new_rows'],12)
            self.assertEqual(m['config']['families'],list(WS_FAMILIES))
            means,wins,advantages,components=summarize(combined,manifest,bootstrap_weights(2,20,4))
            self.assertEqual(len(wins),11*3*3)
            with self.assertRaises(ValueError):attach_comparison_parent(data.iloc[:-1],m)
            malformed=data.copy();malformed.iloc[0,malformed.columns.get_loc('family')]='ce_empirical'
            with self.assertRaises(ValueError):attach_comparison_parent(malformed,m)

    def test_sbatch_dispatches_only_requested_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tools').mkdir();log=root/'args.json'
            (root/'tools/submit_architecture_mc_narma.sh').write_text('''#!/bin/bash
set -euo pipefail
python - "$@" <<'CHECK'
import json,os,sys
json.dump(dict(args=sys.argv[1:],out=os.environ['OUTROOT'],repeats=os.environ['TOTAL_REPEATS']),open(os.environ['TEST_LOG'],'w'))
CHECK
''')
            env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_JOB_ID='789',TEST_LOG=str(log),TOTAL_REPEATS='20')
            for key in ['PROJECT_ROOT','RUN_ID','OUTROOT','ANALYSIS_DIR']:env.pop(key,None)
            subprocess.run(['bash',str(r.ROOT/'r_ws_binary_halfnormal.sbatch')],env=env,check=True)
            result=json.loads(log.read_text());args=result['args']
            self.assertEqual(args[1:3],list(WS_FAMILIES));self.assertIn('--comparison-parent',args);self.assertIn('--with-ipc',args)
            self.assertNotIn('--reuse-parent-through',args);self.assertIn('ws_binary_halfnormal',result['out'])
            self.assertEqual(result['repeats'],'20')

if __name__=='__main__':unittest.main()
