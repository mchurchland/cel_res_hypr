"""Scientific invariants, reuse, scaling, persistence and GPU launcher tests."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import architecture_mc_narma as base
import er_weight_distribution as run
from network_stats.er_weight_distributions import FAMILIES,NORMALIZATIONS,make_magnitude_controls,moment_matched,build_cache,scaling_factor
from network_stats.audited_temporal_tasks import TaskSettings,make_task_data
from test_architecture_ws_controls import parent_cache
from tools.analyze_er_weight_distribution import summarize


class Magnitudes(unittest.TestCase):
    def setUp(self):self.w=np.r_[np.ones(25),np.arange(2.,75.)];self.seed=9876

    def test_realized_moment_matching_rank_and_sign_safety(self):
        controls,details=make_magnitude_controls(self.w,self.seed)
        for family in ['gamma','lognormal']:
            w=controls[family];self.assertTrue((w>0).all());self.assertTrue(np.isfinite(w).all())
            np.testing.assert_allclose([w.mean(),w.var()],[self.w.mean(),self.w.var()],rtol=1e-10)
            self.assertGreater(details[family]['fit_shape'],0)
            # Strict empirical inequalities remain ordered, despite broken ties.
            self.assertTrue((np.diff(np.sort(w[self.w==1]))>=0).all())
            self.assertLess(w[self.w==1].max(),w[self.w==2].min())
        again,_=make_magnitude_controls(self.w,self.seed)
        for family in controls:np.testing.assert_array_equal(controls[family],again[family])

    def test_caps_mean_restoration_and_two_heterogeneity_levels(self):
        controls,details=make_magnitude_controls(self.w,self.seed)
        for name,q in [('cap99',.99),('cap95',.95),('cap90',.9)]:
            expected=np.minimum(self.w,np.quantile(self.w,q));expected*=self.w.mean()/expected.mean()
            np.testing.assert_array_equal(controls[name],expected);self.assertAlmostEqual(controls[name].mean(),self.w.mean())
        for name,a in [('hetero50',.5),('hetero75',.75)]:
            self.assertAlmostEqual(controls[name].mean(),self.w.mean())
            self.assertAlmostEqual(controls[name].var(),a*a*self.w.var())

    def test_constant_and_invalid_magnitudes(self):
        for kind in ['gamma','lognormal']:
            w,shape=moment_matched(np.full(10,3.),kind);np.testing.assert_array_equal(w,np.full(10,3.));self.assertEqual(shape,0)
        for w in [[0,1],[-1,2],[1,np.nan],[1,np.inf],[1]]:
            with self.assertRaises(ValueError):moment_matched(w,'gamma')
        with self.assertRaises(ValueError):moment_matched([1,2],'other')

    def test_masks_signs_frozen_projection_and_scaling(self):
        parent=parent_cache(self.seed,[0,.5,1]);before={k:v.copy() for k,v in parent.items()}
        cache=build_cache(parent,[0,.5,1],[0,.5,1],self.seed,FAMILIES)
        for i in range(4):
            signs=np.sign(parent[f'er_signed_empirical__{i}__values'])
            reference_f=np.linalg.norm(parent[f'ce_empirical__{i}__values']);ce_r=float(parent[f'ce_empirical__{i}__radius'])
            for family in FAMILIES:
                w,e,r=base.raw_from_cache(cache,family,i);values=cache[f'{family}__{i}__values'];f=np.linalg.norm(values)
                np.testing.assert_array_equal(np.sign(values),signs)
                np.testing.assert_array_equal(cache[f'{family}__{i}__edges'],parent[f'er_signed_empirical__{i}__edges'])
                self.assertFalse(np.diag(w).any())
                factor=scaling_factor('frobenius_match',.6,r,ce_r,f,reference_f)
                self.assertAlmostEqual(factor*f,.6/ce_r*reference_f,places=9)
                if family in ['empirical','gamma','lognormal']:
                    self.assertEqual(factor,scaling_factor('shared_ce_scale',.6,r,ce_r,f,reference_f))
        for k,v in before.items():np.testing.assert_array_equal(parent[k],v)
        for norm in NORMALIZATIONS:
            with self.assertRaises(ValueError):scaling_factor(norm,.6,0,1,1,1)


class Pipeline(unittest.TestCase):
    def parent(self,root):
        root.mkdir();fractions=[0.,.5,1.];s=TaskSettings(washout=10,train=60,validation=20,test=30,narma_burnin=40,mc_max_delay=3,mc_diagnostic_delay=5)
        c=dict(families=list(run.PARENT_FAMILIES.values()),normalizations=['own_radius','shared_ce_scale'],fractions=fractions,
            rhos=[.6],leaks=[1.],inputs=[.1],repeats=2,seed=9876,seed_stride=100000,state_dtype='float64',tasks=asdict(s),input_projection='fixture')
        identity=dict(protocol=base.PROTOCOL,config=c,source_sha256={name:base.file_digest(base.ROOT/name) for name in base.SOURCES+base.DATA})
        m=dict(**identity,experiment_id=base.digest(base.canonical(identity)),data=dict(nodes=12,reference_fraction=1/3),expected_evaluations=36)
        base.write_json(root/'manifest.json',m);all_rows=[]
        for repeat in range(2):
            seed=9876+100000*repeat;cache=parent_cache(seed,fractions);data=make_task_data(s,seed)
            base.write_cache(root/'cache'/f'networks_{repeat:04d}.npz',cache,m['experiment_id'])
            base.write_cache(root/'cache'/f'tasks_{repeat:04d}.npz',data,m['experiment_id'])
            for family in c['families']:
                for fi in range(3):
                    for norm in c['normalizations']:
                        rows=base.condition_rows(m,cache,data,repeat,family,fi,norm,base.grid(m),'cpu',1)
                        base.save_shard(base.shard_path(root,repeat,family,fi,norm),rows,m,repeat,family,fi,norm);all_rows.extend(rows)
        base.write_csv(root/'architecture_mc_narma.ALL.csv',all_rows)
        base.write_json(str(root/'architecture_mc_narma.ALL.csv')+'.json',dict(experiment_id=m['experiment_id'],sha256=base.file_digest(root/'architecture_mc_narma.ALL.csv')))
        return m

    def prepared(self,root):
        parent=root/'parent';self.parent(parent);child=root/'child'
        a=run.parser().parse_args(['prepare','--run-dir',str(child),'--parent-run',str(parent),'--repeats','2','--fractions','0','.5','1'])
        run.prepare(a);m=run.load_run(child);return child,m

    def test_preparation_cache_and_baseline_score_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            child,m=self.prepared(Path(tmp));self.assertEqual(m['expected_evaluations'],180)
            self.assertEqual(m['expected_parent_reuse_rows'],36);self.assertEqual(m['expected_scaling_reuse_rows'],18)
            cache=run.networks_cached(child,m,0);data=base.tasks_cached(child,m,0)
            with patch.object(base,'simulate_batch',side_effect=AssertionError('Must reuse baseline scores')):
                for family in run.PARENT_FAMILIES:
                    for norm in ['own_radius','shared_ce_scale']:
                        rows=run.condition_rows(child,m,cache,data,0,family,1,norm,base.grid(m),'cpu',1)
                        self.assertEqual(rows[0]['baseline_scores_reused'],1);run.validate_rows(rows,m,0,family,1,norm)
            bad=dict(cache);bad['input_vector']=cache['input_vector']+.01
            with self.assertRaisesRegex(ValueError,'identical matrix'):run.condition_rows(child,m,bad,data,0,'empirical',1,'own_radius',base.grid(m),'cpu',1)

    def test_all_models_norms_schema_and_frobenius_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            child,m=self.prepared(Path(tmp));cache=run.networks_cached(child,m,0);data=base.tasks_cached(child,m,0);fields=None
            for family in FAMILIES:
                for norm in NORMALIZATIONS:
                    rows=run.condition_rows(child,m,cache,data,0,family,1,norm,base.grid(m),'cpu',1)
                    run.validate_rows(rows,m,0,family,1,norm)
                    fields=set(rows[0]) if fields is None else fields;self.assertEqual(set(rows[0]),fields)
                    path=base.shard_path(child,0,family,1,norm);run.save_shard(path,rows,m,0,family,1,norm)
                    run.read_shard(path,m,0,family,1,norm)
                    if family in ['empirical','gamma','lognormal'] and norm=='frobenius_match':self.assertEqual(rows[0]['scaling_scores_reused'],1)
            fresh=run.condition_rows(child,m,cache,data,0,'cap95',1,'frobenius_match',base.grid(m),'cpu',1,True)
            fresh[0]['scaled_frobenius_norm']*=2
            with self.assertRaises(ValueError):run.validate_rows(fresh,m,0,'cap95',1,'frobenius_match')
            with self.assertRaisesRegex(ValueError,'Incomplete'):run.validate_rows([],m,0,'cap95',1,'frobenius_match')

    def test_complete_small_merge_analysis_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            child,m=self.prepared(Path(tmp));records=[]
            for repeat in range(2):
                cache=run.networks_cached(child,m,repeat);data=base.tasks_cached(child,m,repeat)
                for family in FAMILIES:
                    for fi in range(3):
                        for norm in NORMALIZATIONS:
                            rows=run.condition_rows(child,m,cache,data,repeat,family,fi,norm,base.grid(m),'cpu',1)
                            run.save_shard(base.shard_path(child,repeat,family,fi,norm),rows,m,repeat,family,fi,norm);records.extend(rows)
            run.merge(argparse.Namespace(run_dir=child));run.merge(argparse.Namespace(run_dir=child))
            with self.assertRaises(FileExistsError):run.save_shard(base.shard_path(child,0,'empirical',0,'own_radius'),records[:1],m,0,'empirical',0,'own_radius')
            weights=np.random.default_rng(2).multinomial(2,[.5,.5],size=20)/2
            tables=summarize(pd.DataFrame(records),m,weights)
            self.assertEqual(len(tables[0]),10*3*3*2);self.assertEqual(len(tables[1]),9*3*3*2)
            self.assertTrue(np.allclose(tables[4].empirical_win_auc_percent+tables[4].control_win_auc_percent+tables[4].tie_auc_percent,100))

    def test_gpu_launcher_dependencies_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';self.parent(parent);fake=root/'bin';fake.mkdir();log=root/'jobs.jsonl'
            f=fake/'sbatch';f.write_text('#!'+sys.executable+'\nimport os,json,sys\nwith open(os.environ["TEST_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(123456)\n');f.chmod(0o755)
            env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'],PROJECT_ROOT=str(run.ROOT),SLURM_SUBMIT_DIR=str(run.ROOT),SLURM_JOB_ID='789',
                PARENT_RUN_DIR=str(parent),OUTROOT=str(root/'launch'),ANALYSIS_DIR=str(root/'analysis'),PYTHON_BIN=sys.executable,
                TOTAL_REPEATS='2',ARRAY_TASKS='2',TEST_LOG=str(log),RUN_ID='fixture')
            result=subprocess.run(['bash',str(run.ROOT/'r_er_weight_distribution.sbatch'),'--fractions','0','.5','1'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr);jobs=[json.loads(s) for s in log.read_text().splitlines()]
            self.assertEqual(len(jobs),3);self.assertIn('--dependency=afterok:123456',jobs[1]);self.assertIn('--array=0-1',jobs[1])
            self.assertIn('--dependency=afterok:123456',jobs[2]);self.assertIn('preflight',jobs[0][-1])
            prepared=run.load_run(root/'launch');self.assertEqual(prepared['config']['families'],list(FAMILIES))
            self.assertEqual(prepared['config']['normalizations'],list(NORMALIZATIONS))


if __name__=='__main__':unittest.main()
