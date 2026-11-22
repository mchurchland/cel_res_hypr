"""Execute the user-requested cleanup only after complete local verification."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,shlex,subprocess,sys

here=Path(__file__).resolve().parent
receipt=json.loads((here/'verification.json').read_text())
assert receipt['status']=='verified' and not receipt['errors']
manifest=here/'remote_manifest.jsonl'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==receipt['remote_manifest_sha256']
local=Path(receipt['local_root'])
stats=[json.loads(line) for line in (here/'local_verified_stats.jsonl').read_text().splitlines()]
assert len(stats)==receipt['total_items']==receipt['verified_items']

def unchanged(record):
    p=local/record['path'];s=p.lstat()
    assert (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)==(record['size'],record['mtime_ns'],record['inode'],record['device']),p

with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(unchanged,stats))
program=(here/'delete_verified_remote.py').read_text()
command=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','orca','python3 -c '+shlex.quote(program)]
with manifest.open('rb') as inp,(here/'deletion.log').open('wb') as out,(here/'deletion.stderr.log').open('wb') as err:
    completed=subprocess.run(command,stdin=inp,stdout=out,stderr=err)
print('Remote deletion exit code:',completed.returncode,flush=True)
print((here/'deletion.log').read_text(),flush=True)
if completed.returncode:
    print((here/'deletion.stderr.log').read_text(),file=sys.stderr)
    raise SystemExit(completed.returncode)
