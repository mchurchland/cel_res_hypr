"""Runs on Orca with the verified JSONL inventory on stdin.

Deletes only inventoried entries of ~/Thesis/results, after a complete stat/list
check. Any active Slurm job, novel path, changed inode/size/mtime, or unexpected
file type aborts the entire precheck before deletion. Leaves the root folder.
"""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import datetime,json,os,stat,subprocess,sys

root=Path.home()/'Thesis/results'
assert str(root)=='/home/milesc/Thesis/results' and root.is_dir() and not root.is_symlink()
records=[json.loads(line) for line in sys.stdin if line.strip()]
assert records[0]['kind']=='header' and records[-1]['kind']=='footer'
assert records[0]['root']==str(root) and records[-1]['counts']['error']==0
assert not records[0]['active_jobs'].strip()
for r in records:
    if 'path' in r:
        p=PurePosixPath(r['path']);assert not p.is_absolute() and '..' not in p.parts and p.parts
expected={r['path']:r for r in records if r['kind'] in ('file','directory','symlink')}
assert len(expected)==sum(r['kind'] in ('file','directory','symlink') for r in records)

def idle():
    jobs=subprocess.check_output(['squeue','--me','--noheader'],text=True)
    if jobs.strip():raise RuntimeError('Active Slurm jobs detected; remote results retained')

idle()
actual=set()
for base,dirs,files in os.walk(root,followlinks=False):
    for name in dirs+files:actual.add(str((Path(base)/name).relative_to(root)))
if actual!=set(expected):raise RuntimeError(f'Remote inventory changed: added={len(actual-set(expected))}, missing={len(set(expected)-actual)}')

def check(r):
    p=root/r['path'];s=p.lstat()
    if r['kind']=='directory':
        if not stat.S_ISDIR(s.st_mode):raise RuntimeError(f'Directory type changed: {p}')
        return
    if (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)!=(r['size'],r['mtime_ns'],r['inode'],r['device']):
        raise RuntimeError(f'Remote file changed since its verified SHA-256 snapshot: {p}')
    if r['kind']=='file' and not stat.S_ISREG(s.st_mode):raise RuntimeError(f'Not a regular file: {p}')
    if r['kind']=='symlink' and (not stat.S_ISLNK(s.st_mode) or os.readlink(p)!=r['target']):raise RuntimeError(f'Symlink changed: {p}')

parents=('architecture_mc_narma_ipc_full/architecture_full_ipc_271871',
         'ws_binary_halfnormal/ws_binary_halfnormal_272279',
         'er_weight_distribution/er_weights_274402')
def needed_for_mc200(name):
    return any(name==p+'/manifest.json' or name.startswith(p+'/cache/') or name.startswith(p+'/shards/') for p in parents)
items=[r for r in expected.values() if r['kind']!='directory' and not needed_for_mc200(r['path'])]
retained=[r for r in expected.values() if r['kind']!='directory' and needed_for_mc200(r['path'])]
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(check,expected.values()))
idle()
print(json.dumps(dict(kind='precheck',status='passed',files=len(items),directories=sum(r['kind']=='directory' for r in expected.values()),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)

def remove(r):
    check(r)
    p=root/r['path']
    # The full precheck established that every parent is a real inventoried dir.
    # Symlink entries are unlinked, never followed.
    p.unlink()
    return r['size'] if r['kind']=='file' else 0

with ThreadPoolExecutor(max_workers=8) as pool:
    removed_bytes=sum(pool.map(remove,items))
directories=sorted((r['path'] for r in expected.values() if r['kind']=='directory'),key=lambda p:(p.count('/'),p),reverse=True)
for relative in directories:
    p=root/relative
    if not any(p.iterdir()):p.rmdir()
remaining=list(root.iterdir())
print(json.dumps(dict(kind='completion',status='archived_outputs_deleted_mc200_parents_retained',removed_items=len(items),removed_logical_bytes=removed_bytes,
    retained_mc200_parent_files=len(retained),retained_mc200_parent_bytes=sum(r['size'] for r in retained),
    remaining_entries=[p.name for p in remaining],root_retained=str(root),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
print(subprocess.check_output(['quota','-s'],text=True),flush=True)
