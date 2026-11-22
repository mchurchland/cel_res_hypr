"""Compare every remote result with its local copy; never delete anything."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,stat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'results'
records=[json.loads(x) for x in (HERE/'remote_manifest.jsonl').read_text().splitlines()]
assert records[0]['kind']=='header' and records[-1]['kind']=='footer'
assert records[0]['root']=='/home/milesc/Thesis/results'
assert not records[0]['active_jobs'].strip() and records[-1]['counts']['error']==0
items=[r for r in records if r['kind'] in ('file','symlink')]
for r in records:
    if 'path' in r:
        p=PurePosixPath(r['path']);assert not p.is_absolute() and '..' not in p.parts

def check(r):
    p=ROOT/r['path']
    if not p.exists() and not p.is_symlink():return dict(path=r['path'],error='Missing local copy')
    info=p.lstat()
    if r['kind']=='symlink':
        if not stat.S_ISLNK(info.st_mode) or os.readlink(p)!=r['target']:return dict(path=r['path'],error='Symlink differs')
        return dict(path=r['path'],kind=r['kind'],size=info.st_size,mtime_ns=info.st_mtime_ns,inode=info.st_ino,device=info.st_dev)
    if not stat.S_ISREG(info.st_mode) or info.st_size!=r['size']:return dict(path=r['path'],error='Local type/size differs')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    if h.hexdigest()!=r['sha256']:return dict(path=r['path'],error='Local SHA-256 differs')
    after=p.stat()
    if (info.st_size,info.st_mtime_ns,info.st_ino,info.st_dev)!=(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev):
        return dict(path=r['path'],error='Local file changed during verification')
    return dict(path=r['path'],kind=r['kind'],size=info.st_size,mtime_ns=info.st_mtime_ns,inode=info.st_ino,device=info.st_dev)

errors=[];verified=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(check,items):
        if 'error' in result:errors.append(result)
        else:verified.append(result)
with (HERE/'local_verified_stats.jsonl').open('w') as f:
    for record in verified:f.write(json.dumps(record,separators=(',',':'))+'\n')
receipt=dict(status='verified' if not errors else 'mismatch',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    remote_root=records[0]['root'],local_root=str(ROOT),verified_items=len(items)-len(errors),total_items=len(items),
    file_bytes=records[-1]['total_file_bytes'],remote_manifest_sha256=hashlib.sha256((HERE/'remote_manifest.jsonl').read_bytes()).hexdigest(),
    errors=errors,method='SHA-256 for every regular file; exact target for every symlink; no local-only files removed')
(HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (HERE/'repair_files.nul').open('wb') as f:
    for r in errors:f.write(r['path'].encode()+b'\0')
print(json.dumps({k:v for k,v in receipt.items() if k!='errors'},indent=2),flush=True)
if errors:print('Errors',errors[:10],flush=True);raise SystemExit(1)
