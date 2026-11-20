"""Scientific invariants and failure modes for the new unified sweep."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import Ridge
import torch
import architecture_mc_narma as runner
from network_stats.architecture_sweep import (FAMILIES, NORMALIZATIONS, make_assets,
    raw_networks,rng_for,scaling_factor,spectrum,validate_connectomes)
from network_stats.audited_temporal_tasks import (TaskSettings, conventional_narma,
    delay_targets,make_task_data,NarmaDivergenceError,score_states,simulate_batch,
    squared_correlations,validated_ridge,GENERATOR_SHA256)
from third_party.echotorch.NARMADataset import NARMADataset
from tools.analyze_architecture_mc_narma import variance_components,bootstrap_weights


def fixture():
    n=12; known=np.zeros((n,n)); unknown=known.copy()
    for i in range(n):
        for j,offset in enumerate((1,2,4,7)):
            if j==3: unknown[i,(i+offset)%n]=i+1
            else: known[i,(i+offset)%n]=(1 if j!=1 else -1)*(i*3+j+1)
    return known,unknown


def small_settings():
    return TaskSettings(washout=10,train=60,validation=20,test=30,narma_burnin=40,
                        mc_max_delay=3,mc_diagnostic_delay=5,ridge_alphas=(1e-8,1e-4,.1))


class Models(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known,cls.unknown=fixture()
        cls.assets=make_assets(cls.known,cls.unknown,9876,swaps_per_edge=2)

    def test_model_counts_and_exact_signed_matching(self):
        for f in (0,.1,.2,.3,.4,.5,1):
            nets=raw_networks(self.assets,f,9876)
            self.assertEqual(set(nets),set(FAMILIES))
            parent=np.sort(nets['ce_empirical'][nets['ce_empirical']!=0])
            for name,w in nets.items():
                self.assertEqual(w.shape,(12,12)); self.assertTrue(np.isfinite(w).all())
                self.assertFalse(np.diag(w).any())
                expected=36 if name=='ce_unknown_removed' else 48
                self.assertEqual(np.count_nonzero(w),expected)
                self.assertEqual(np.count_nonzero(w<0),int(f*expected))
            for name in ('ce_shuffled','ce_degree_rewired','er_signed_empirical','ws_signed_empirical'):
                np.testing.assert_array_equal(np.sort(nets[name][nets[name]!=0]),parent)
            for a,b in [('ce_binary','er_binary'),('ce_halfnormal','er_halfnormal')]:
                np.testing.assert_array_equal(np.sort(nets[a][nets[a]!=0]),np.sort(nets[b][nets[b]!=0]))

    def test_degrees_source_weights_and_sign_mask(self):
        nets=raw_networks(self.assets,.3,9876); w=nets['ce_empirical']; r=nets['ce_degree_rewired']
        for axis in (0,1): np.testing.assert_array_equal(np.count_nonzero(w,axis=axis),np.count_nonzero(r,axis=axis))
        for i in range(len(w)): np.testing.assert_array_equal(np.sort(w[i,w[i]!=0]),np.sort(r[i,r[i]!=0]))
        for name in ('ce_shuffled','ce_binary','ce_halfnormal'):
            np.testing.assert_array_equal(np.sign(w),np.sign(nets[name]))
        self.assertFalse(np.array_equal(w,r))
        self.assertFalse(np.array_equal(w,nets['ce_shuffled']))

    def test_nested_signs_and_removed_units(self):
        low=raw_networks(self.assets,.2,9876); high=raw_networks(self.assets,.5,9876)
        for name in FAMILIES:
            self.assertTrue(np.all(high[name][low[name]<0]<0))
        positive=raw_networks(self.assets,0,9876)
        mask=self.known!=0
        np.testing.assert_array_equal(positive['ce_unknown_removed'][mask],positive['ce_empirical'][mask])
        self.assertNotAlmostEqual(abs(positive['ce_unknown_removed'][mask]).mean(),1)

    def test_stream_reproducibility(self):
        a=make_assets(self.known,self.unknown,9876,swaps_per_edge=2)
        for k in a: np.testing.assert_array_equal(a[k],self.assets[k])
        self.assertFalse(np.array_equal(rng_for(1,'a').random(10),rng_for(1,'b').random(10)))

    def test_normalization_semantics(self):
        nets=raw_networks(self.assets,.2,9876)
        ce=spectrum(nets['ce_empirical'])[1]; er=spectrum(nets['er_signed_empirical'])[1]
        self.assertAlmostEqual(scaling_factor('own_radius',.95,er,7,ce)*er,.95)
        self.assertEqual(scaling_factor('fixed_reference',.95,3,7,ce),scaling_factor('fixed_reference',.95,8,7,ce))
        a=scaling_factor('shared_ce_scale',.95,ce,7,ce)
        b=scaling_factor('shared_ce_scale',.95,er,7,ce)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(np.sort((nets['ce_empirical']*a)[nets['ce_empirical']!=0]),
                                      np.sort((nets['er_signed_empirical']*b)[nets['er_signed_empirical']!=0]))
        self.assertAlmostEqual(spectrum(nets['ce_empirical']*a)[1],.95)
        for bad in (0,np.nan,-1):
            with self.assertRaises(ValueError): scaling_factor('own_radius',bad,er,7,ce)
        with self.assertRaises(ValueError): spectrum(np.zeros((4,4)))

    def test_bad_connectomes(self):
        for known,unknown in [(np.zeros((3,2)),np.zeros((3,2))),
                (self.known,self.known),(self.known,-self.unknown),
                (np.eye(12),self.unknown),(np.full((12,12),np.nan),self.unknown)]:
            with self.assertRaises(ValueError): validate_connectomes(known,unknown)
        for f in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError): raw_networks(self.assets,f,9876)


class Tasks(unittest.TestCase):
    def test_upstream_hash_and_actual_conventional_alignment(self):
        path=runner.ROOT/'third_party/echotorch/NARMADataset.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),GENERATOR_SHA256)
        u,y,residual=conventional_narma(60,40,123)
        self.assertEqual(u.shape,(60,1)); self.assertLess(residual,1e-12)
        with torch.random.fork_rng(devices=[]):
            old=torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64); torch.random.default_generator.manual_seed(123)
                raw_u,raw_y=NARMADataset(101,1,10)[0]
            finally: torch.set_default_dtype(old)
        raw_u,raw_y=raw_u.numpy(),raw_y.numpy()
        np.testing.assert_array_equal(u,raw_u[40:-1]); np.testing.assert_array_equal(y,raw_y[41:])
        t=59
        expected=.3*raw_y[t]+.05*raw_y[t]*sum(raw_y[t-i] for i in range(10))+1.5*raw_u[t-9]*raw_u[t]+.1
        np.testing.assert_allclose(raw_y[t+1],expected,atol=1e-15)
        # The archived ReservoirPy lag/history expression is a different task.
        archived=.3*raw_y[t]+.05*raw_y[t]*raw_y[t-10:t].sum()+1.5*raw_u[t-10]*raw_u[t]+.1
        self.assertGreater(float(abs(raw_y[t+1,0]-archived[0])),1e-5)

    def test_no_global_rng_or_default_precision_side_effect(self):
        before=torch.random.get_rng_state().clone(); dtype=torch.get_default_dtype()
        before_numpy=np.random.get_state()
        make_task_data(small_settings(),123)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(dtype,torch.get_default_dtype())
        after_numpy=np.random.get_state()
        np.testing.assert_array_equal(before_numpy[1],after_numpy[1]); self.assertEqual(before_numpy[2:],after_numpy[2:])

    def test_prefix_independent_of_future_input_length(self):
        u1,y1,_=conventional_narma(50,40,123)
        u2,y2,_=conventional_narma(100,40,123)
        np.testing.assert_array_equal(u1,u2[:50]); np.testing.assert_array_equal(y1,y2[:50])

    def test_nonfinite_stream_rejection_is_recorded_and_deterministic(self):
        with self.assertRaises(NarmaDivergenceError): conventional_narma(3000,1000,57761122)
        d=make_task_data(TaskSettings(),20761119)
        self.assertEqual(d['narma_rejected_seeds'].tolist()[0],57761122)
        self.assertEqual(int(d['narma_generator_seed']),57761122+1009*len(d['narma_rejected_seeds']))
        self.assertTrue(np.isfinite(d['narma_target']).all())
        with patch('network_stats.audited_temporal_tasks.conventional_narma',side_effect=ValueError('bad recurrence')):
            with self.assertRaisesRegex(ValueError,'bad recurrence'): make_task_data(small_settings(),123)

    def test_delay_alignment_and_perfect_delay_register(self):
        s=small_settings(); data=make_task_data(s,123)
        targets=delay_targets(data['mc_input'],5)
        np.testing.assert_array_equal(targets[10],data['mc_input'][[9,8,7,6,5],0])
        states=np.nan_to_num(targets,nan=0)
        scores=score_states(states,data,s,'MC')
        self.assertAlmostEqual(scores['MC'],3,places=7)
        self.assertAlmostEqual(scores['MC_extended'],5,places=7)
        self.assertAlmostEqual(scores['MC_tail'],2,places=7)
        self.assertEqual(scores['MC_state_variance_test'],float(np.var(states[-30:],axis=0).mean()))

    def test_constant_and_high_variance_states(self):
        s=small_settings(); data=make_task_data(s,123)
        # Constant predictions have zero correlation-based MC; no epsilon floor.
        result=score_states(np.ones((s.length,7)),data,s,'MC')
        self.assertEqual(result['MC'],0); self.assertEqual(result['MC_extended'],0)
        result=score_states(np.ones((s.length,7)),data,s,'NARMA10')
        self.assertAlmostEqual(result['NARMA10_test_nmse'],result['NARMA10_test_mean_baseline_nmse'])
        for scale in (1e-8,1e8):
            result=score_states(np.random.default_rng(4).normal(size=(s.length,5))*scale,data,s,'MC')
            self.assertTrue(0<=result['MC']<=3)
        p=np.array([[2.,1.],[4.,1.],[6.,1.],[8.,1.]])
        np.testing.assert_allclose(squared_correlations(np.arange(8).reshape(4,2),p),[1,0])

    def test_ridge_matches_separate_library_fits(self):
        rng=np.random.default_rng(6); x=rng.normal(size=(100,8)); y=rng.normal(size=(100,3))
        x[:,4]=x[:,2]; x[:,6]=1
        penalties=(1e-8,.001,1)
        pred,alpha,loss,scale=validated_ridge(x[:60],y[:60],x[60:80],y[60:80],x[80:],penalties)
        center=x[:60].mean(0)
        candidates=[]; validation=[]
        for a in sorted(penalties,reverse=True):
            model=Ridge(alpha=a,solver='svd').fit((x[:60]-center)/scale,y[:60])
            candidates.append(model.predict((x[80:]-center)/scale))
            validation.append(np.mean((model.predict((x[60:80]-center)/scale)-y[60:80])**2,axis=0))
        choice=np.argmin(validation,axis=0)
        expected=np.column_stack([candidates[choice[j]][:,j] for j in range(3)])
        np.testing.assert_allclose(pred,expected,rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(alpha,np.asarray(sorted(penalties,reverse=True))[choice])
        for factor in (1e-9,-1e9):
            transformed=validated_ridge(x[:60]*factor,y[:60],x[60:80]*factor,y[60:80],x[80:]*factor,penalties)
            np.testing.assert_allclose(pred,transformed[0],rtol=1e-10,atol=1e-10)
        # Changing test features cannot affect selected penalties or validation loss.
        changed=validated_ridge(x[:60],y[:60],x[60:80],y[60:80],x[80:]*100,penalties)
        np.testing.assert_array_equal(alpha,changed[1]); np.testing.assert_array_equal(loss,changed[2])

    def test_invalid_values(self):
        for kwargs in [dict(washout=1),dict(mc_max_delay=0),dict(train=0),dict(ridge_alphas=(0,)),dict(test=-1)]:
            with self.assertRaises(ValueError): TaskSettings(**kwargs)
        with self.assertRaises(ValueError): delay_targets(np.ones((4,2)),2)
        with self.assertRaises(ValueError): squared_correlations(np.ones((5,1)),np.ones((5,1)))
        with self.assertRaises(ValueError): squared_correlations(np.ones((5,1)),np.full((5,1),np.nan))
        with self.assertRaises(ValueError): validated_ridge(np.ones((4,2)),np.ones((4,1)),np.ones((4,3)),np.ones((4,1)),np.ones((4,2)),[1])
        with self.assertRaises(ValueError): conventional_narma(10,0,1)


class Recurrence(unittest.TestCase):
    def test_orientation_leak_and_batch_isolation(self):
        w=np.zeros((2,3,3)); w[:,0,1]=.5
        b=np.array([[1,0,0],[2,0,0]])
        u=np.array([[1.],[0.],[0.],[1.]])
        states,gap=simulate_batch(w,b,[1,.6],u,2)
        self.assertAlmostEqual(float(states[0,0,0]),np.tanh(1),places=6)
        self.assertEqual(states[0,0,1],0)
        self.assertAlmostEqual(float(states[0,1,1]),np.tanh(.5*np.tanh(1)),places=6)
        for j,leak in enumerate([1,.6]):
            single,singlegap=simulate_batch(w[j:j+1],b[j:j+1],[leak],u,2)
            np.testing.assert_allclose(single[0],states[j],atol=1e-7)
            np.testing.assert_allclose(singlegap[0],gap[j],atol=1e-7)
        runner.parity_check('cpu')

    def test_invalid_states_and_values(self):
        w=np.zeros((1,3,3)); b=np.ones((1,3)); u=np.ones((5,1))
        for leaks in ([0],[1.1],[np.nan]):
            with self.assertRaises(ValueError): simulate_batch(w,b,leaks,u,2)
        with self.assertRaises(ValueError): simulate_batch(w,b,[1],u,5)
        with self.assertRaises(ValueError): simulate_batch(w,b,[1],u,2,echo_initial=np.nan)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError,'refusing CPU fallback'): runner.configure_device('cuda:0')


class PersistenceAndAnalysis(unittest.TestCase):
    def test_frozen_prepare_and_launcher_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); out=tmp/'run'; fake=tmp/'bin'; fake.mkdir()
            log=tmp/'submitted.jsonl'
            sbatch=fake/'sbatch'
            sbatch.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                'with open(os.environ["TEST_SBATCH_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'print(123456)\n')
            sbatch.chmod(0o755)
            env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'],
                PYTHON_BIN=sys.executable,OUTROOT=str(out),ANALYSIS_DIR=str(tmp/'analysis'),
                ARRAY_TASKS='2',TOTAL_REPEATS='2',TEST_SBATCH_LOG=str(log))
            command=['bash',str(runner.ROOT/'tools/submit_architecture_mc_narma.sh'),
                     '--families','ce_empirical','er_binary',
                     '--normalizations','own_radius','--fractions','0','.2',
                     '--rhos','.6','--leaks','1','--inputs','.1']
            result=subprocess.run(command,env=env,cwd=tmp,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            submitted=[json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(submitted),3)
            self.assertTrue(any('preflight' in v for v in submitted[0]))
            self.assertIn('--dependency=afterok:123456',submitted[1])
            self.assertIn('--array=0-1',submitted[1])
            self.assertIn('--dependency=afterok:123456',submitted[2])
            m=runner.load_run(out)
            self.assertEqual(m['expected_evaluations'],8)
            self.assertEqual(m['config']['state_dtype'],'float64')
            # Frozen entry point imports correctly when launched outside repo.
            result=subprocess.run([sys.executable,str(out/'snapshot/architecture_mc_narma.py'),
                'merge','--run-dir',str(out)],cwd=tmp,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('FileNotFoundError',result.stderr)
            with (out/'snapshot/util/util.py').open('a') as f: f.write('\n# changed\n')
            with self.assertRaisesRegex(ValueError,'Frozen source/data changed'): runner.load_run(out)

    def test_immutable_files_cache_hashes_and_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.npz'
            runner.write_cache(p,{'a':np.arange(5)},'test')
            np.testing.assert_array_equal(runner.read_cache(p,'test')['a'],np.arange(5))
            with self.assertRaises(FileExistsError): runner.write_cache(p,{'b':np.arange(5)},'test')
            with self.assertRaises(ValueError): runner.read_cache(p,'wrong')
            p.write_bytes(p.read_bytes()+b'corruption')
            with self.assertRaises(ValueError): runner.read_cache(p,'test')
        partitions=[set(range(rank,13,5)) for rank in range(5)]
        self.assertEqual(set.union(*partitions),set(range(13)))
        self.assertEqual(sum(len(p) for p in partitions),13)

    def test_complete_run_shard_resume_merge_and_reject_tampering(self):
        # Small scientific fixture exercises actual model creation, metrics,
        # serialized identities, complete-grid validation and strict merge.
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'; out.mkdir()
            settings=small_settings()
            m={'protocol':runner.PROTOCOL,'experiment_id':'fixture',
               'config':{'tasks':json.loads(json.dumps(settings.__dict__)),'fractions':[.2],
                         'families':['ce_empirical'],'normalizations':['own_radius'],
                         'rhos':[.6],'leaks':[.6],'inputs':[.1],
                         'repeats':1,'seed':9876,'seed_stride':100000},'expected_evaluations':1}
            known,unknown=fixture(); a=make_assets(known,unknown,9876,2)
            cache={'n_nodes':a['n_nodes'],'reference_fraction':a['reference_fraction'],
                   'input_vector':rng_for(9876,'shared all-node input projection').uniform(-1,1,12),
                   'directed_swaps':a['directed_swaps'],'rewired_edge_retention':a['rewired_edge_retention']}
            for index,fraction in enumerate([.2,float(a['reference_fraction'])]):
                w=raw_networks(a,fraction,9876)['ce_empirical']; e=np.argwhere(w!=0); ev,r=spectrum(w)
                prefix=f'ce_empirical__{index}'
                cache.update({prefix+'__edges':e,prefix+'__values':w[e[:,0],e[:,1]],
                              prefix+'__eigenvalues':ev,prefix+'__radius':np.array(r)})
            data=make_task_data(settings,9876)
            rows=runner.condition_rows(m,cache,data,0,'ce_empirical',0,'own_radius',runner.grid(m),'cpu',1)
            path=runner.shard_path(out,0,'ce_empirical',0,'own_radius')
            runner.save_shard(path,rows,m,0,'ce_empirical',0,'own_radius')
            runner.read_shard(path,m,0,'ce_empirical',0,'own_radius')
            with patch.object(runner,'load_run',return_value=m):
                runner.merge(argparse.Namespace(run_dir=out))
                runner.merge(argparse.Namespace(run_dir=out))
            bad=[dict(rows[0],MC=100)]
            with self.assertRaises(ValueError): runner.validate_rows(bad,m,0,'ce_empirical',0,'own_radius')
            with self.assertRaises(ValueError): runner.validate_rows([],m,0,'ce_empirical',0,'own_radius')
            path.write_text(path.read_text()+'corruption')
            with self.assertRaises(ValueError): runner.read_shard(path,m,0,'ce_empirical',0,'own_radius')
            path.unlink()
            with patch.object(runner,'load_run',return_value=m):
                with self.assertRaises(FileNotFoundError): runner.merge(argparse.Namespace(run_dir=out))

    def test_variance_components_include_interactions_and_constant_floor(self):
        import pandas as pd
        import itertools
        points=list(itertools.product([-.5,.5],[-1.,1.],[-2.,2.]))
        table=pd.DataFrame(points,columns=['rho_target','leak','input_scale'])
        table['score']=10+table.rho_target+table.leak+table.input_scale+table.leak*table.input_scale
        terms,total=variance_components(table)
        self.assertAlmostEqual(total,.25+1+4+4)
        shares={r['component']:r['variance'] for r in terms}
        self.assertAlmostEqual(shares['leak × input_scale'],4)
        table['score']=0
        terms,total=variance_components(table)
        self.assertEqual(total,0); self.assertEqual(sum(r['variance'] for r in terms),0)
        with self.assertRaises(ValueError): variance_components(table.iloc[:-1])
        weights=bootstrap_weights(5,30,123)
        np.testing.assert_allclose(weights.sum(1),1)
        self.assertEqual(weights.shape,(30,5))


if __name__=='__main__': unittest.main()
