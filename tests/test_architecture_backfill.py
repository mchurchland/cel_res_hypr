"""Parent-score reuse, IPC-only backfill and streaming merge regressions."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import architecture_mc_narma as r
from network_stats.audited_temporal_tasks import TaskSettings,make_task_data
from network_stats.ipc_capacity import IPCSettings,add_ipc_data


def small_fixture():
    n=12;known=np.zeros((n,n));unknown=known.copy()
    for source in range(n):
        for j,offset in enumerate((1,2,4,7)):
            target=(source+offset)%n
            if j==3:unknown[source,target]=source+1
            else:known[source,target]=(1 if j!=1 else -1)*(source*3+j+1)
    return known,unknown


class Backfill(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.parent=self.root/'parent';self.parent.mkdir();self.child=self.root/'child';self.child.mkdir()
        self.tasks=TaskSettings(washout=10,train=60,validation=20,test=30,narma_burnin=40,mc_max_delay=3,mc_diagnostic_delay=5,ridge_alphas=(1e-8,1e-4,.1))
        config=dict(tasks=asdict(self.tasks),fractions=[0,.5],families=['ce_empirical'],normalizations=['own_radius'],
                    rhos=[.6],leaks=[1.],inputs=[.1,.5],repeats=1,seed=9876,seed_stride=100000,state_dtype='float64')
        self.old=dict(protocol=r.PROTOCOL,experiment_id='fixture-parent',config=config,expected_evaluations=4)
        (self.parent/'manifest.json').write_text(json.dumps(self.old))
        c=dict(config,fractions=[0,.5,1.],ipc=IPCSettings(3,2).definition(),reuse_parent_through=.5,
               overlap_parent=dict(path=str(self.parent),manifest_sha256=r.file_digest(self.parent/'manifest.json'),experiment_id=self.old['experiment_id']))
        self.new=dict(protocol=r.IPC_PROTOCOL,experiment_id='fixture-child',config=c,expected_evaluations=6)
        assets=r.make_assets(*small_fixture(),9876,2)
        self.old_cache=self.cache(assets,config['fractions']);self.new_cache=self.cache(assets,c['fractions'])
        self.data=make_task_data(self.tasks,9876);self.extended=add_ipc_data(self.data,self.tasks,IPCSettings(3,2))
        r.write_cache(self.parent/'cache/tasks_0000.npz',self.data,self.old['experiment_id'])
        self.parent_rows={}
        for fi in range(2):
            rows=r.condition_rows(self.old,self.old_cache,self.data,0,'ce_empirical',fi,'own_radius',r.grid(self.old),'cpu',2)
            self.parent_rows[fi]=rows
            r.save_shard(r.shard_path(self.parent,0,'ce_empirical',fi,'own_radius'),rows,self.old,0,'ce_empirical',fi,'own_radius')

    def cache(self,a,fractions):
        cache={k:a[k] for k in ['n_nodes','reference_fraction','directed_swaps','rewired_edge_retention']}
        cache['input_vector']=r.rng_for(9876,'shared all-node input projection').uniform(-1,1,int(a['n_nodes']))
        for index,fraction in enumerate(list(fractions)+[float(a['reference_fraction'])]):
            w=r.raw_networks(a,fraction,9876)['ce_empirical'];edges=np.argwhere(w!=0);ev,rho=r.spectrum(w);p=f'ce_empirical__{index}'
            cache.update({p+'__edges':edges,p+'__values':w[edges[:,0],edges[:,1]],p+'__eigenvalues':ev,p+'__radius':np.array(rho)})
        return cache

    def rows(self,fi,**kw):
        return r.condition_rows(self.new,self.new_cache,self.extended,0,'ce_empirical',fi,'own_radius',r.grid(self.new),'cpu',2,**kw)

    def test_lower_half_simulates_only_MC_and_never_refits_MC_NARMA(self):
        with patch.object(r,'simulate_batch',wraps=r.simulate_batch) as simulation,patch.object(r,'score_states',side_effect=AssertionError('Readout was refitted')):
            rows=self.rows(1)
        self.assertEqual(simulation.call_count,1)
        np.testing.assert_array_equal(simulation.call_args.args[3],self.data['mc_input'])
        for old,new in zip(self.parent_rows[1],rows):
            for key,value in r.inherited_scores(old).items():self.assertEqual(new[key],value)
            self.assertEqual(new['MC_NARMA_reused'],1)
            self.assertEqual(new['MC_NARMA_source_experiment_id'],self.old['experiment_id'])
            self.assertEqual(len(new['MC_NARMA_parent_shard_sha256']),64)
            self.assertTrue(np.isfinite(new['IPC']))
        fresh=self.rows(1,reuse_parent=False)
        np.testing.assert_allclose([x['IPC'] for x in rows],[x['IPC'] for x in fresh],atol=1e-12)

    def test_upper_half_computes_both_trajectories_and_readouts(self):
        with patch.object(r,'simulate_batch',wraps=r.simulate_batch) as simulation,patch.object(r,'score_states',wraps=r.score_states) as readout:
            rows=self.rows(2)
        self.assertEqual(simulation.call_count,2);self.assertEqual(readout.call_count,4)
        self.assertTrue(all(row['MC_NARMA_reused']==0 for row in rows))
        self.assertTrue(all(row['MC_NARMA_source_experiment_id']==self.new['experiment_id'] for row in rows))
        self.assertTrue(all(row['MC_NARMA_parent_shard_sha256']=='' for row in rows))
        r.validate_rows(rows,self.new,0,'ce_empirical',2,'own_radius')

    def test_parent_input_and_matrix_mismatches_stop_reuse(self):
        with patch.object(r,'make_task_data',side_effect=AssertionError('Must use stored parent streams')):
            first=r.tasks_cached(self.child,self.new,0)
        np.testing.assert_array_equal(first['mc_input'],self.data['mc_input'])
        original=r.read_cache
        def changed(path,experiment_id):
            result=original(path,experiment_id)
            if Path(path)==self.parent/'cache/tasks_0000.npz':result['mc_input']=result['mc_input']+.01
            return result
        with patch.object(r,'read_cache',side_effect=changed):
            with self.assertRaises(AssertionError):r.tasks_cached(self.child,self.new,0)
        self.new_cache['ce_empirical__0__values']=self.new_cache['ce_empirical__0__values'].copy()
        self.new_cache['ce_empirical__0__values'][0]+=.01
        with self.assertRaisesRegex(ValueError,'identical raw network'):self.rows(0)

    def test_stream_merge_reused_and_new_rows_with_different_key_order(self):
        import csv
        for fi in range(3):
            rows=self.rows(fi)
            r.save_shard(r.shard_path(self.child,0,'ce_empirical',fi,'own_radius'),rows,self.new,0,'ce_empirical',fi,'own_radius')
        with patch.object(r,'load_run',return_value=self.new):
            r.merge(type('Args',(),{'run_dir':self.child})())
            r.merge(type('Args',(),{'run_dir':self.child})())
        with (self.child/'architecture_mc_narma.ALL.csv').open() as f:merged=list(csv.DictReader(f))
        self.assertEqual(len(merged),6)
        self.assertEqual(sum(int(row['MC_NARMA_reused']) for row in merged),4)
        self.assertTrue(all(np.isfinite(float(row['IPC'])) for row in merged))
        self.assertFalse(list(self.child.glob('*.merge.tmp')))
        bad=dict(self.rows(0)[0],MC_NARMA_reused=0)
        with self.assertRaisesRegex(ValueError,'reuse flag'):r.validate_rows([bad,self.rows(0)[1]],self.new,0,'ce_empirical',0,'own_radius')

    def test_prepare_full_range_counts_and_provenance(self):
        parent=self.root/'prepared-parent';child=self.root/'prepared-full'
        common=['--repeats','2','--rhos','.6','--leaks','1','--inputs','.1']
        r.prepare(r.parser().parse_args(['prepare','--run-dir',str(parent),*common]))
        r.prepare(r.parser().parse_args(['prepare','--run-dir',str(child),*common,'--with-ipc',
            '--fractions','0','.1','.2','.3','.4','.5','.6','.7','.8','.9','1',
            '--overlap-parent',str(parent),'--reuse-parent-through','.5']))
        m=r.load_run(child)
        self.assertEqual(m['expected_evaluations'],660)
        self.assertEqual(m['reused_temporal_evaluations'],360)
        self.assertEqual(m['new_temporal_evaluations'],300)
        self.assertEqual(m['expected_trajectories'],960)
        self.assertEqual(m['config']['reuse_parent_through'],.5)
        self.assertEqual(m['config']['ipc']['target_count'],285)


if __name__=='__main__':unittest.main()
