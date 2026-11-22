"""Read-only SHA-256 inventory of ~/Thesis/results; stream JSON Lines to stdout."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,stat,subprocess,sys

root=Path.home()/'Thesis/results'
assert root.is_dir() and not root.is_symlink()
jobs=subprocess.check_output(['squeue','--me','--noheader'],text=True)
if jobs.strip():raise SystemExit('Active jobs detected; do not delete results in use')
print(json.dumps(dict(kind='header',root=str(root),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active_jobs=jobs)),flush=True)
paths=[];directories=[]
for base,dirs,files in os.walk(root,followlinks=False):
    for name in list(dirs):
        p=Path(base)/name
        if p.is_symlink():dirs.remove(name);paths.append(p)
        else:directories.append(p)
    paths.extend(Path(base)/name for name in files)

def inventory(path):
    before=path.lstat();name=str(path.relative_to(root))
    common=dict(path=name,size=before.st_size,mtime_ns=before.st_mtime_ns,inode=before.st_ino,device=before.st_dev)
    if stat.S_ISLNK(before.st_mode):return dict(kind='symlink',target=os.readlink(path),**common)
    if not stat.S_ISREG(before.st_mode):return dict(kind='error',error='Unsupported file type',**common)
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):digest.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)!=(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev):
        return dict(kind='error',error='File changed while hashing',**common)
    return dict(kind='file',sha256=digest.hexdigest(),**common)

counts={'file':0,'symlink':0,'error':0};total=0
with ThreadPoolExecutor(max_workers=8) as pool:
    for start in range(0,len(paths),2048):
        for record in pool.map(inventory,paths[start:start+2048]):
            counts[record['kind']]+=1
            if record['kind']=='file':total+=record['size']
            print(json.dumps(record,separators=(',',':')))
        sys.stdout.flush()
for p in sorted(directories):print(json.dumps(dict(kind='directory',path=str(p.relative_to(root))),separators=(',',':')))
print(json.dumps(dict(kind='footer',counts=counts,total_file_bytes=total,directories=len(directories),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
if counts['error']:raise SystemExit(1)
