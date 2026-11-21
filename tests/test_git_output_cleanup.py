"""Check ignore policy and data-preserving deployment using disposable Git repos."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class GitOutputCleanup(unittest.TestCase):
    def git(self,root,*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,text=True,capture_output=True).stdout.strip()

    def setup_repo(self,root):
        remote=root/'remote.git';origin=root/'origin';deploy=root/'deploy'
        subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
        subprocess.run(['git','init','-b','main',str(origin)],check=True,capture_output=True)
        self.git(origin,'config','user.name','Test');self.git(origin,'config','user.email','test@example.invalid')
        files={'code.py':'version one\n','Connectome/input.npy':'small source input\n','results/run/data.csv':'original result\n',
            'final_results/main/data.csv':'archived result\n','analysis/run/scores.csv':'scores\n','analysis/run/REPORT.md':'report\n',
            'architecture_variant_figures/picture.png':'figure\n','architecture_variant_figures/make.py':'plot source\n',
            'metric_figures/figure.png':'figure\n','logs/job.out':'log\n','job-submit-1.err':'stderr\n','build/lib/copy.py':'build copy\n'}
        for path,value in files.items():
            p=origin/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
        self.git(origin,'add','.');self.git(origin,'commit','-m','Initial code and tracked output')
        self.git(origin,'remote','add','origin',str(remote));self.git(origin,'push','-u','origin','main')
        subprocess.run(['git','clone','-b','main',str(remote),str(deploy)],check=True,capture_output=True)
        self.git(deploy,'config','user.name','Test');self.git(deploy,'config','user.email','test@example.invalid')
        shutil.copyfile(ROOT/'.gitignore',origin/'.gitignore');(origin/'code.py').write_text('version two\n')
        (origin/'new_run.sbatch').write_text('#!/bin/bash\n')
        self.git(origin,'rm','--cached','-r','results','final_results','logs','build')
        self.git(origin,'rm','--cached','analysis/run/scores.csv','architecture_variant_figures/picture.png','metric_figures/figure.png','job-submit-1.err')
        self.git(origin,'add','.gitignore','code.py','new_run.sbatch');self.git(origin,'commit','-m','Stop tracking output and update code')
        self.git(origin,'push','origin','main')
        return origin,deploy,files

    def invoke(self,deploy):
        return subprocess.run(['bash',str(ROOT/'tools/update_orca_preserving_results.sh')],cwd=deploy,text=True,capture_output=True)

    def test_preserves_generated_data_and_updates_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin,deploy,files=self.setup_repo(Path(tmp))
            (deploy/'results/run/data.csv').write_text('newer cluster result\n');(deploy/'analysis/run/REPORT.md').write_text('local cluster report\n')
            kept=[p for p in files if p.startswith(('results/','final_results/','analysis/','logs/','architecture_variant_figures/','metric_figures/','build/')) or p.endswith('.err')]
            before={p:(deploy/p).read_bytes() for p in kept};result=self.invoke(deploy)
            self.assertEqual(result.returncode,0,result.stderr)
            for path,value in before.items():self.assertEqual((deploy/path).read_bytes(),value,path)
            self.assertEqual((deploy/'code.py').read_text(),'version two\n');self.assertTrue((deploy/'new_run.sbatch').exists())
            self.assertEqual(self.git(deploy,'rev-parse','HEAD'),self.git(origin,'rev-parse','HEAD'))
            self.assertEqual(self.git(deploy,'ls-files','results','final_results','logs','build'),'')
            self.assertTrue(self.git(deploy,'ls-files','Connectome/input.npy'))

    def test_stops_for_local_source_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,deploy,_=self.setup_repo(Path(tmp));old=self.git(deploy,'rev-parse','HEAD')
            (deploy/'code.py').write_text('local edits\n');result=self.invoke(deploy)
            self.assertNotEqual(result.returncode,0);self.assertIn('local source-code edits',result.stderr)
            self.assertEqual(self.git(deploy,'rev-parse','HEAD'),old);self.assertEqual((deploy/'code.py').read_text(),'local edits\n')

    def test_direct_index_removal_preserves_working_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,deploy,_=self.setup_repo(Path(tmp));paths=['results/run/data.csv','final_results/main/data.csv','logs/job.out']
            before={p:(deploy/p).read_bytes() for p in paths}
            subprocess.run(['git','-C',str(deploy),'update-index','--force-remove','-z','--stdin'],
                           input=('\0'.join(paths)+'\0').encode(),check=True,capture_output=True)
            self.assertEqual(self.git(deploy,'ls-files','--',*paths),'')
            for path,value in before.items():self.assertEqual((deploy/path).read_bytes(),value)

    def test_stops_for_local_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,deploy,_=self.setup_repo(Path(tmp));(deploy/'code.py').write_text('local commit\n')
            self.git(deploy,'add','code.py');self.git(deploy,'commit','-m','Local work');old=self.git(deploy,'rev-parse','HEAD')
            result=self.invoke(deploy);self.assertNotEqual(result.returncode,0);self.assertIn('local commits diverge',result.stderr)
            self.assertEqual(self.git(deploy,'rev-parse','HEAD'),old)

    def test_stops_for_untracked_source_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,deploy,_=self.setup_repo(Path(tmp));(deploy/'new_run.sbatch').write_text('local untracked file\n');old=self.git(deploy,'rev-parse','HEAD')
            result=self.invoke(deploy);self.assertNotEqual(result.returncode,0);self.assertIn('already exists locally',result.stderr)
            self.assertEqual((deploy/'new_run.sbatch').read_text(),'local untracked file\n');self.assertEqual(self.git(deploy,'rev-parse','HEAD'),old)

    def test_ignore_policy_keeps_code_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);self.git(repo,'init');shutil.copyfile(ROOT/'.gitignore',repo/'.gitignore')
            ignored=['results/run.csv','final_results/a.csv','analysis/run/scores.csv','analysis/run/plot.png',
                     'analysis/run/snapshot/code.py','analysis/run/code_snapshot/tool.py','analysis/run/default_plan/README.md',
                     'logs/job.out','job-submit-2.err','architecture_variant_figures/picture.png','metric_figures/table.csv','build/lib/code.py']
            kept=['tools/analysis.py','tests/test_new.py','r_er_weight_distribution.sbatch','Connectome/input.npy',
                  'analysis/run/REPORT.md','analysis/run/compute.py','architecture_variant_figures/make.py','metric_figures/make.py']
            for path in ignored+kept:
                p=repo/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
                result=subprocess.run(['git','-C',str(repo),'check-ignore','--quiet',path])
                self.assertEqual(result.returncode,0 if path in ignored else 1,path)


if __name__=='__main__':unittest.main()
