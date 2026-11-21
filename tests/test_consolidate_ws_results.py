"""Preserve raw numeric strings and reject incomplete/unpaired consolidation."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from architecture_mc_narma import file_digest
from tools.consolidate_ws_results import merge_csvs,validate_raw,validate_cache,ROOT,METRICS


class Consolidation(unittest.TestCase):
    def sources(self,root):
        c=dict(repeats=2,families=['ce_empirical','ws_binary'],fractions=[.2],normalizations=['own_radius'],
               rhos=[.6],leaks=[1.],inputs=[.1],seed=12,seed_stride=100000,tasks=dict(mc_max_delay=30),ipc=dict(target_count=285))
        paths=[];manifests=[]
        for family in c['families']:
            directory=root/family;directory.mkdir();path=directory/'data.csv'
            m=dict(experiment_id=family,protocol='fixture',config=dict(families=[family]),expected_evaluations=2)
            (directory/'manifest.json').write_text(json.dumps(m));rows=[]
            for repeat in range(2):
                rows.append(dict(repeat_id=repeat,family=family,negative_fraction=.2,normalization='own_radius',rho_target=.6,
                    leak=1.,input_scale=.1,MC=3.+repeat,NARMA10_test_nmse=.4,IPC=-1e9 if repeat else 10.,
                    experiment_id=family,protocol='fixture',seed=12+100000*repeat,input_vector_sha256='hash'+str(repeat)))
            pd.DataFrame(rows).to_csv(path,index=False);self.refresh(path,m)
            paths.append(path);manifests.append(m)
        return c,paths,manifests

    def refresh(self,path,m):
        Path(str(path)+'.json').write_text(json.dumps(dict(experiment_id=m['experiment_id'],sha256=file_digest(path))))

    def test_complete_paired_surfaces_retain_negative_ipc(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,paths,manifests=self.sources(Path(tmp));cube,sources=validate_raw(paths,manifests,c)
            self.assertEqual(cube.shape,(2,2,1,1,1,1,1,3));self.assertEqual(sum(s['rows'] for s in sources),4)
            self.assertEqual(cube[1,1,0,0,0,0,0,2],-1e9)

    def test_duplicate_or_missing_observations_rejected(self):
        for duplicate in [True,False]:
            with self.subTest(duplicate=duplicate),tempfile.TemporaryDirectory() as tmp:
                c,paths,manifests=self.sources(Path(tmp));frame=pd.read_csv(paths[1])
                frame=frame.iloc[[0,0]] if duplicate else frame.iloc[:1]
                frame.to_csv(paths[1],index=False);self.refresh(paths[1],manifests[1])
                with self.assertRaises(ValueError):validate_raw(paths,manifests,c)

    def test_unpaired_input_projection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,paths,manifests=self.sources(Path(tmp));frame=pd.read_csv(paths[1]);frame.loc[0,'input_vector_sha256']='different'
            frame.to_csv(paths[1],index=False);self.refresh(paths[1],manifests[1])
            with self.assertRaisesRegex(ValueError,'input projections'):validate_raw(paths,manifests,c)

    def test_column_reordering_preserves_exact_strings_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.csv';b=root/'b.csv';out=root/'combined.csv'
            a.write_text('family,MC,IPC\nce_empirical,0.12345678901234567890,-1000000000.012345678\n')
            b.write_text('IPC,family,MC\n-987654321.987654321,ws_binary,0.98765432109876543210\n')
            before=[file_digest(a),file_digest(b)];sha=merge_csvs([a,b],out)
            with out.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['MC'],'0.12345678901234567890')
            self.assertEqual(rows[1]['IPC'],'-987654321.987654321')
            self.assertEqual(rows[1]['MC'],'0.98765432109876543210')
            self.assertEqual([file_digest(a),file_digest(b)],before);self.assertEqual(sha,file_digest(out))
            with self.assertRaises(FileExistsError):merge_csvs([a,b],out)

    def test_cached_point_estimates_must_reproduce_raw_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,paths,manifests=self.sources(root);cube,sources=validate_raw(paths,manifests,c)
            cache=root/'cache';cache.mkdir();means=[];advantages=[];wins=[]
            for i,metric in enumerate(METRICS):
                for j,family in enumerate(c['families']):
                    means.append(dict(family=family,negative_fraction=.2,normalization='own_radius',metric=metric,mean=cube[:,j,...,i].mean()))
                advantages.append(dict(control='ws_binary',negative_fraction=.2,normalization='own_radius',rho_target=.6,leak=1.,input_scale=.1,metric=metric,ce_advantage=0.))
                wins.append(dict(control='ws_binary',negative_fraction=.2,normalization='own_radius',metric=metric,ce_winning_fraction=0.))
            tables={'mean_performance.csv':pd.DataFrame(means),'configuration_advantages.csv':pd.DataFrame(advantages),'ce_winning_fraction.csv':pd.DataFrame(wins)}
            def write_tables(tables):
                for name,frame in tables.items():frame.to_csv(cache/name,index=False)
                record=dict(experiment_id=sources[1]['experiment_id'],source_sha256=sources[1]['sha256'],
                    analysis_sha256=file_digest(ROOT/'tools/analyze_architecture_mc_narma.py'),
                    comparison_parent=dict(experiment_id=sources[0]['experiment_id'],sha256=sources[0]['sha256'],analysis_rows=4,analysis_families=c['families']),
                    outputs={name:file_digest(cache/name) for name in tables})
                (cache/'manifest.json').write_text(json.dumps(record))
            write_tables(tables);validate_cache(cache,sources,c,cube)
            for name,column in [('mean_performance.csv','mean'),('configuration_advantages.csv','ce_advantage'),('ce_winning_fraction.csv','ce_winning_fraction')]:
                with self.subTest(table=name):
                    changed={key:frame.copy() for key,frame in tables.items()};changed[name].loc[0,column]+=1.
                    write_tables(changed)
                    with self.assertRaises(ValueError):validate_cache(cache,sources,c,cube)


if __name__=='__main__':unittest.main()
