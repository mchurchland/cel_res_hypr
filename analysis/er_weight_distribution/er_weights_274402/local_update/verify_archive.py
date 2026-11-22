"""Verify all transferred checkpoint/cache sidecars and frozen source files."""
from pathlib import Path
import hashlib
import itertools
import json
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / 'results/er_weight_distribution' / HERE.parent.name


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    m = json.loads((RUN / 'manifest.json').read_text())
    c = m['config']
    expected = {f'repeat_{r:04d}__{f}__ei_{i:02d}__{n}.csv'
                for r, f, i, n in itertools.product(range(c['repeats']), c['families'], range(len(c['fractions'])), c['normalizations'])}
    observed = {p.name for p in (RUN / 'shards').glob('*.csv')}
    assert observed == expected, 'Missing/unexpected checkpoint files'
    counts = {}
    for part, pattern in [('shards', '*.csv'), ('cache', '*.npz')]:
        count = 0
        for path in (RUN / part).glob(pattern):
            sidecar = json.loads(Path(str(path) + '.json').read_text())
            assert sidecar == dict(experiment_id=m['experiment_id'], sha256=sha(path)), path
            count += 1
        counts[part] = count
        print(part, count, 'checksums verified', flush=True)
    assert counts['cache'] == 2 * c['repeats'], 'Missing network/task caches'
    for name, digest in m['source_sha256'].items():
        assert sha(RUN / 'snapshot' / name) == digest, name
    gate = json.loads((RUN / 'preflight/PASSED.json').read_text())
    assert gate['experiment_id'] == m['experiment_id']
    assert gate['smoke_sha256'] == sha(RUN / 'preflight/smoke.csv')
    valid = json.loads((HERE / 'validation.json').read_text())
    assert valid['status'] == 'passed' and valid['experiment_id'] == m['experiment_id']
    original = json.loads((HERE.parent / 'manifest.json').read_text())
    for name, digest in original['outputs'].items():
        assert sha(HERE.parent / name) == digest, name
    files = [p for p in RUN.rglob('*') if p.is_file()]
    receipt = dict(status='passed', verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   remote_host='login.orca.pdx.edu', remote_run='/home/milesc/Thesis/results/er_weight_distribution/' + RUN.name,
                   local_run=str(RUN.relative_to(ROOT)), experiment_id=m['experiment_id'],
                   file_count=len(files), total_bytes=sum(p.stat().st_size for p in files),
                   checkpoint_sha256_verified=counts['shards'], cache_sha256_verified=counts['cache'],
                   snapshot_sha256_verified=len(m['source_sha256']), preflight_smoke_sha256_verified=True,
                   original_analysis_sha256_verified=len(original['outputs']), source_sha256=valid['source_sha256'],
                   csv_rows=valid['rows'], scientific_validation='validation.json', verifier_sha256=sha(Path(__file__)))
    (HERE / 'transfer_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
