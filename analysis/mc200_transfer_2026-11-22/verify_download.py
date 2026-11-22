"""Verify a completed MC200 download without changing any scientific outputs."""
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]/'results/mc200_recompute/mc200_274500'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    m = json.loads((RUN/'manifest.json').read_text())
    done = json.loads((RUN/'COMPLETE.json').read_text())
    assert done['experiment_id'] == m['experiment_id']
    assert done['rows'] == m['expected_rows']
    assert done['narma_rerun'] is False and done['ipc_rerun'] is False
    identity = {k: m[k] for k in ('protocol','parents','tasks','source_sha256','logical_specs','unique_specs')}
    assert hashlib.sha256(canonical(identity)).hexdigest() == m['experiment_id']
    merged = RUN/'mc200.ALL.csv.gz'
    assert file_sha(merged) == done['sha256'], 'Merged CSV checksum mismatch'
    for name, sha in m['source_sha256'].items():
        assert file_sha(RUN/'snapshot'/name) == sha, f'Frozen source differs: {name}'
    config = m['parents']['architecture']['config']
    n = len(config['rhos'])*len(config['leaks'])*len(config['inputs'])
    h = m['tasks']['mc_max_delay']
    assert h == 200
    expected = {}
    for repeat in range(config['repeats']):
        for spec in m['unique_specs']:
            for fi in range(len(config['fractions'])):
                name = f'r{repeat:04d}__{"__".join(spec)}__ei{fi:02d}.npz'
                expected[name] = repeat, spec, fi
    assert len(expected) == m['groups']
    actual = {p.name for p in (RUN/'checkpoints').glob('*.npz')}
    assert actual == set(expected), f'Checkpoint set differs: missing={len(set(expected)-actual)}, extra={len(actual-set(expected))}'

    def check(item):
        name, (repeat, spec, fi) = item
        with np.load(RUN/'checkpoints'/name, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            arrays = {k: z[k] for k in z.files if k != 'metadata'}
        assert (meta['experiment_id'], meta['repeat'], meta['spec'], meta['fraction_index']) == (m['experiment_id'], repeat, spec, fi), name
        digest = hashlib.sha256()
        for key, value in sorted(arrays.items()):
            digest.update(canonical(dict(name=key, shape=list(value.shape), dtype=value.dtype.str)))
            digest.update(value.tobytes())
        assert digest.hexdigest() == meta['arrays_sha256'], f'Checkpoint checksum differs: {name}'
        assert arrays['curves'].shape == (n,h) and arrays['alpha_index'].shape == (n,h), name
        assert all(np.isfinite(v).all() for v in arrays.values()), name
        assert np.all((arrays['curves'] >= 0) & (arrays['curves'] <= 1+1e-12)), name
        assert np.all(arrays['alpha_index'] < len(m['tasks']['ridge_alphas'])), name
        return (RUN/'checkpoints'/name).stat().st_size

    verified = 0
    total = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for size in pool.map(check, expected.items()):
            verified += 1
            total += size
            if verified % 10000 == 0:
                print(f'Validated {verified:,}/{len(expected):,} checkpoints', flush=True)
    rows = 0
    with gzip.open(merged, 'rt', newline='') as f:
        for row in csv.DictReader(f):
            assert row['experiment_id'] == m['experiment_id']
            assert float(row['MC']) == float(row['MC_200'])
            assert 0 <= float(row['MC_30']) <= float(row['MC_60'])+1e-12
            assert float(row['MC_60']) <= float(row['MC_100'])+1e-12
            assert float(row['MC_100']) <= float(row['MC_150'])+1e-12
            assert float(row['MC_150']) <= float(row['MC_200'])+1e-12
            rows += 1
    assert rows == m['expected_rows']
    receipt = dict(status='verified', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   remote_source='orca:/home/milesc/Thesis/results/mc200_recompute/mc200_274500/',
                   local_run=str(RUN), experiment_id=m['experiment_id'], merged_rows=rows,
                   merged_sha256=done['sha256'], checkpoints=verified, checkpoint_bytes=total,
                   frozen_source_files=len(m['source_sha256']), delays=h, test_steps=m['tasks']['test'],
                   narma_rerun=False, ipc_rerun=False)
    (HERE/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
