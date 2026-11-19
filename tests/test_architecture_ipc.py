"""Basis, finite-sample estimator and extension invariants for IPC."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import eval_legendre
import architecture_mc_narma as runner
from network_stats.audited_temporal_tasks import TaskSettings,make_task_data,simulate_batch,score_states
from network_stats.ipc_capacity import IPCSettings,ipc_targets,add_ipc_data,target_capacities,score_ipc
from tools.analyze_architecture_mc_narma import summarize,bootstrap_weights


class IPC(unittest.TestCase):
    def test_prepare_IPC_extension_keeps_parent_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);parent=root/'parent';child=root/'upper'
            common=['--repeats','2','--rhos','.6','--leaks','1','--inputs','.1']
            args=runner.parser().parse_args(['prepare','--run-dir',str(parent),*common])
            runner.prepare(args)
            args=runner.parser().parse_args(['prepare','--run-dir',str(child),*common,
                '--fractions','.5','.6','.7','.8','.9','1','--with-ipc','--overlap-parent',str(parent)])
            runner.prepare(args);m=runner.load_run(child)
            self.assertEqual(m['protocol'],runner.IPC_PROTOCOL)
            self.assertEqual(m['config']['ipc']['target_count'],285)
            self.assertEqual(m['config']['fractions'],[.5,.6,.7,.8,.9,1])
            self.assertEqual(m['config']['overlap_parent']['path'],str(parent))
            args.inputs=[.5]
            with self.assertRaisesRegex(ValueError,'different inputs'):runner.prepare(args)

    def settings(self):
        return TaskSettings(washout=10,train=100,validation=40,test=100,narma_burnin=40,
            mc_max_delay=3,mc_diagnostic_delay=5,ridge_alphas=(1e-8,1e-4,.1))

    def test_complete_basis_counts_and_crossdelay_alignment(self):
        spec=IPCSettings();self.assertEqual(spec.target_count,285)
        u=np.random.default_rng(4).uniform(-1,1,(100,1));y,e=ipc_targets(u,spec)
        self.assertEqual(y.shape,(100,285));self.assertEqual(len(np.unique(e,axis=0)),285)
        for degree,count in [(1,10),(2,55),(3,220)]:self.assertEqual(int((e.sum(1)==degree).sum()),count)
        # Multi-delay product P1(u[t-1])*P1(u[t-3]), not P2 of one lag.
        key=np.zeros(10,dtype=int);key[0]=key[2]=1;j=np.flatnonzero(np.all(e==key,axis=1))[0]
        t=25;self.assertAlmostEqual(y[t,j],3*u[t-1,0]*u[t-3,0])
        key=np.zeros(10,dtype=int);key[1]=3;j=np.flatnonzero(np.all(e==key,axis=1))[0]
        self.assertAlmostEqual(y[t,j],np.sqrt(7)*eval_legendre(3,u[t-2,0]))
        self.assertTrue(np.isnan(y[:10]).all())

    def test_population_orthonormality(self):
        u=np.random.default_rng(23).uniform(-1,1,(80000,1))
        y,e=ipc_targets(u,IPCSettings(3,3));x=y[3:]
        np.testing.assert_allclose(x.mean(0),0,atol=.025)
        np.testing.assert_allclose(x.T@x/len(x),np.eye(len(e)),atol=.035)

    def test_capacity_definition_negative_estimates_and_zero_target(self):
        y=np.array([[-1.,2.],[1.,-2.],[-1.,2.],[1.,-2.]])
        np.testing.assert_allclose(target_capacities(y,y),[1,1])
        np.testing.assert_allclose(target_capacities(y,np.zeros_like(y)),[0,0])
        np.testing.assert_allclose(target_capacities(y,-y),[-3,-3])
        with self.assertRaises(ValueError):target_capacities(np.zeros((4,1)),np.zeros((4,1)))
        with self.assertRaises(ValueError):target_capacities(y,np.full_like(y,np.nan))

    def test_perfect_features_and_component_reconstruction(self):
        t=self.settings();s=IPCSettings(3,2);data=add_ipc_data(make_task_data(t,123),t,s)
        states=np.nan_to_num(data['ipc_targets'])
        r=score_ipc(states,data,t,s)
        self.assertAlmostEqual(r['IPC'],s.target_count,places=7)
        self.assertAlmostEqual(r['IPC_linear'],3,places=7)
        self.assertAlmostEqual(r['IPC_nonlinear'],6,places=7)
        self.assertAlmostEqual(r['IPC'],r['IPC_crossdelay']+r['IPC_single_delay'])
        self.assertAlmostEqual(r['IPC'],r['IPC_degree_1']+r['IPC_degree_2'])

    def test_constant_states_do_not_become_positive_primary_capacity(self):
        t=self.settings();s=IPCSettings(3,2);data=add_ipc_data(make_task_data(t,123),t,s)
        r=score_ipc(np.ones((t.length,8)),data,t,s)
        self.assertAlmostEqual(r['IPC'],r['IPC_mean_predictor_baseline'],places=12)
        self.assertAlmostEqual(r['IPC_excess_over_mean'],0,places=12)
        self.assertGreaterEqual(r['IPC_positive_part'],max(0,r['IPC']))
        self.assertGreater(r['IPC_negative_terms'],0)

    def test_bad_basis_and_input(self):
        for args in [(0,3),(3,0),(1.5,2),(100,5)]:
            with self.assertRaises(ValueError):IPCSettings(*args)
        for u in [np.ones((30,2)),np.full((30,1),2),np.full((30,1),np.nan)]:
            with self.assertRaises(ValueError):ipc_targets(u,IPCSettings(3,2))
        t=self.settings()
        with self.assertRaises(ValueError):add_ipc_data(make_task_data(t,123),t,IPCSettings(11,2))

    def test_adding_ipc_does_not_change_MC_NARMA_or_rng(self):
        t=self.settings();data=make_task_data(t,123);state=np.random.default_rng(2).normal(size=(t.length,7))
        before={k:v.copy() for k,v in data.items()}
        reference=[score_states(state,data,t,task) for task in ['MC','NARMA10']]
        extended=add_ipc_data(data,t,IPCSettings(3,2));score_ipc(state,extended,t,IPCSettings(3,2))
        actual=[score_states(state,extended,t,task) for task in ['MC','NARMA10']]
        self.assertEqual(reference,actual)
        for k,v in before.items():np.testing.assert_array_equal(data[k],v)

    def test_IPC_win_counts_use_signed_higher_is_better(self):
        # Complete tiny grid with negative held-out scores; bigger still wins.
        records=[]
        for repeat in range(2):
            for family,value in [('ce_empirical',-.1),('er_binary',-.2)]:
                records.append(dict(repeat_id=repeat,family=family,normalization='own_radius',negative_fraction=.5,
                     rho_target=.6,leak=1.,input_scale=.1,IPC=value,MC=1.,NARMA10_test_nmse=.5))
        m={'config':{'repeats':2,'families':['ce_empirical','er_binary']}}
        means,wins,advantages,components=summarize(pd.DataFrame(records),m,bootstrap_weights(2,20,4))
        self.assertEqual(float(wins[wins.metric=='IPC'].ce_winning_fraction.iloc[0]),1)
        self.assertAlmostEqual(float(advantages[advantages.metric=='IPC'].ce_advantage.iloc[0]),.1)

    def test_full_sbatch_forwarding(self):
        script=runner.ROOT/'r_architecture_mc_narma_ipc_full.sbatch'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'tools').mkdir();log=root/'args.json'
            (root/'tools/submit_architecture_mc_narma.sh').write_text(
                '#!/bin/bash\nset -euo pipefail\npython - "$@" <<\'CHECK\'\n'
                'import json,os,sys\n'
                'json.dump(dict(args=sys.argv[1:],out=os.environ["OUTROOT"],repeats=os.environ["TOTAL_REPEATS"]),open(os.environ["TEST_LOG"],"w"))\nCHECK\n')
            env=dict(os.environ,SLURM_SUBMIT_DIR=str(root),SLURM_JOB_ID='456',TEST_LOG=str(log),TOTAL_REPEATS='20')
            for k in ['PROJECT_ROOT','RUN_ID','OUTROOT','ANALYSIS_DIR','EI_FRACTIONS']:env.pop(k,None)
            subprocess.run(['bash',str(script)],env=env,check=True)
            result=json.loads(log.read_text());args=result['args']
            self.assertEqual(args[:12],['--fractions','0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1'])
            self.assertEqual(args[args.index('--reuse-parent-through')+1],'.5')
            self.assertIn('--with-ipc',args);self.assertIn('architecture_mc_narma_ipc_full',result['out'])
            self.assertEqual(result['repeats'],'20')

if __name__=='__main__':unittest.main()
