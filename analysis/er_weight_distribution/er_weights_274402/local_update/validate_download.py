"""Check this downloaded experiment and recompute summary point estimates.

Run from the repository root with python and one BLAS thread. No simulations
are run, no source results are changed, and original Orca outputs are retained.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from util.ei_auc import curve_areas

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
RUN = ROOT / 'results/er_weight_distribution' / ANALYSIS.name


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b, label, rtol=1e-10, atol=1e-10):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise ValueError(label)


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    am = json.loads((ANALYSIS / 'manifest.json').read_text())
    identity = {k: manifest[k] for k in ['protocol', 'config', 'source_sha256']}
    canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest['experiment_id'] == am['experiment_id']
    source_locations = {}
    for name, digest in manifest['source_sha256'].items():
        # Raw-data validation can proceed while archived checkpoints download.
        # A matching local source is an explicit fallback, not a snapshot claim.
        location = RUN / 'snapshot' / name
        if not location.exists():
            location = ROOT / name
        assert sha(location) == digest, name
        source_locations[name] = str(location.relative_to(ROOT))
    for name, digest in am['outputs'].items():
        assert sha(ANALYSIS / name) == digest, name
    source = RUN / 'architecture_mc_narma.ALL.csv'
    metadata = json.loads(Path(str(source) + '.json').read_text())
    digest = sha(source)
    assert digest == metadata['sha256'] == am['source_sha256']
    assert metadata['experiment_id'] == manifest['experiment_id']
    print('Source identities and original analysis hashes verified', flush=True)

    c = manifest['config']
    keys = ['repeat_id', 'family', 'negative_fraction', 'normalization', 'rho_target', 'leak', 'input_scale']
    levels = [list(range(c['repeats'])), c['families'], c['fractions'], c['normalizations'], c['rhos'], c['leaks'], c['inputs']]
    shape = tuple(map(len, levels))
    metrics = ['MC', 'NARMA10_test_nmse']
    cube = np.empty((*shape, 2))
    flat = cube.reshape(-1, 2)
    seen = np.zeros(len(flat), dtype=bool)
    strings = {'experiment_id', 'protocol', 'family', 'normalization', 'raw_network_sha256', 'input_vector_sha256',
               'state_dtype', 'readout_dtype', 'backend', 'MC_ridge_alphas', 'MC_NARMA_source_experiment_id', 'MC_NARMA_parent_shard_sha256'}
    reused = []
    count = 0
    for d in pd.read_csv(source, chunksize=50000):
        indices = [pd.Categorical(d[k], categories=v).codes for k, v in zip(keys, levels)]
        assert all((i >= 0).all() for i in indices), 'Unexpected grid coordinates'
        idx = np.ravel_multi_index(indices, shape)
        assert len(np.unique(idx)) == len(idx) and not seen[idx].any(), 'Duplicate settings'
        seen[idx] = True
        flat[idx] = d[metrics].to_numpy()
        numeric = d[[k for k in d if k not in strings]].to_numpy(dtype=float)
        assert np.isfinite(numeric).all(), 'Nonfinite values'
        assert d.experiment_id.eq(manifest['experiment_id']).all() and d.protocol.eq(manifest['protocol']).all()
        assert d.neuron_bias.eq(0).all()
        delays = d[[f'MC_delay_{i:02d}' for i in range(1, c['tasks']['mc_diagnostic_delay'] + 1)]].to_numpy()
        assert (delays >= 0).all() and (delays <= 1 + 1e-12).all()
        equal(d.MC, delays[:, :c['tasks']['mc_max_delay']].sum(axis=1), 'MC delay sum')
        equal(d.MC_extended, delays.sum(axis=1), 'Extended MC delay sum')
        assert (d.NARMA10_test_nmse >= 0).all() and (d.NARMA10_val_nmse >= 0).all()
        equal(d.NARMA10_test_nrmse**2, d.NARMA10_test_nmse, 'NRMSE/NMSE mismatch')
        equal(d.post_scaling_radius, d.raw_radius * d.scaling_factor, 'Radius scaling')
        equal(d.scaled_frobenius_norm, d.raw_frobenius_norm * d.scaling_factor, 'Norm scaling')
        own = d.normalization.eq('own_radius')
        equal(d.loc[own, 'post_scaling_radius'], d.loc[own, 'rho_target'], 'Own-radius target')
        frob = d.normalization.eq('frobenius_match')
        equal(d.loc[frob, 'scaled_frobenius_norm'], (d.rho_target / d.shared_ce_radius * d.reference_raw_frobenius_norm)[frob], 'Frobenius reference')
        matched = d.family.isin(['gamma', 'lognormal'])
        equal(d.loc[matched, 'mean_nonzero_raw_magnitude'], d.loc[matched, 'empirical_magnitude_mean'], 'Matched mean')
        equal(d.loc[matched, 'raw_magnitude_variance'], d.loc[matched, 'empirical_magnitude_variance'], 'Matched variance')
        for family, a in [('hetero50', .5), ('hetero75', .75)]:
            select = d.family.eq(family)
            equal(d.loc[select, 'raw_magnitude_variance'], d.loc[select, 'empirical_magnitude_variance'] * a*a, 'Compressed variance')
        reused.append(d[['baseline_scores_reused', 'scaling_scores_reused']])
        count += len(d)
    assert seen.all() and count == manifest['expected_evaluations'] == len(flat)
    print(f'Checked every numeric row and complete paired grid: {count:,}', flush=True)

    means = cube.mean(axis=(4, 5, 6))  # repeat, family, fraction, normalization, metric
    empirical = c['families'].index('empirical')
    summary = pd.read_csv(ANALYSIS / 'mean_performance.csv')
    wins = pd.read_csv(ANALYSIS / 'empirical_winning_fraction.csv')
    advantages = pd.read_csv(ANALYSIS / 'configuration_advantages.csv')
    auc = pd.read_csv(ANALYSIS / 'performance_auc.csv')
    win_auc = pd.read_csv(ANALYSIS / 'win_fraction_auc.csv')
    x = np.asarray(c['fractions'])
    assert (np.diff(x) > 0).all()
    span = np.ptp(x)
    weights = np.random.default_rng(am['bootstrap_seed']).multinomial(c['repeats'], np.full(c['repeats'], 1/c['repeats']), size=am['bootstrap']) / c['repeats']
    paired = []
    for ni, norm in enumerate(c['normalizations']):
        for mi, metric in enumerate(metrics):
            direction = 1 if metric == 'MC' else -1
            for fi, family in enumerate(c['families']):
                a = auc.query('normalization == @norm and metric == @metric and family == @family').iloc[0]
                seed_auc = curve_areas(means[:, fi, :, ni, mi], x)[0] / span
                equal(seed_auc.mean(), a.performance_auc, 'Performance AUC')
                equal(np.quantile(weights @ seed_auc, [.025, .975]), [a.ci_low, a.ci_high], 'Performance AUC intervals')
                s = summary.query('normalization == @norm and metric == @metric and family == @family').sort_values('negative_fraction')
                equal(means[:, fi, :, ni, mi].mean(axis=0), s['mean'], 'Mean curves')
                if fi == empirical:
                    continue
                delta = direction * (cube[:, empirical, :, ni, :, :, :, mi] - cube[:, fi, :, ni, :, :, :, mi])
                surface = delta.reshape(c['repeats'], len(x), -1)
                point_mean = surface.mean(axis=0)
                w = wins.query('normalization == @norm and metric == @metric and control == @family').sort_values('negative_fraction')
                equal((point_mean > 0).mean(axis=1), w.empirical_winning_fraction, 'Matched-setting win curves')
                wa = win_auc.query('normalization == @norm and metric == @metric and control == @family').iloc[0]
                equal(curve_areas((point_mean > 0).mean(axis=1)*100, x)[0]/span, wa.empirical_win_auc_percent, 'Win-fraction AUC')
                for xi, fraction in enumerate(x):
                    z = advantages.query('normalization == @norm and metric == @metric and control == @family and negative_fraction == @fraction').sort_values(['rho_target', 'leak', 'input_scale'])
                    equal(point_mean[xi], z.empirical_advantage, 'Configuration advantages')
                seed_delta = curve_areas(surface.mean(axis=-1), x)[0]/span
                low, high = np.quantile(weights @ seed_delta, [.025, .975])
                paired.append(dict(normalization=norm, metric=metric, control=family,
                                   empirical_advantage_auc=float(seed_delta.mean()), ci_low=float(low), ci_high=float(high),
                                   n_realizations=c['repeats'], n_configurations=int(np.prod(shape[4:]))))
    for family in ['empirical', 'gamma', 'lognormal']:
        fi = c['families'].index(family)
        np.testing.assert_array_equal(cube[:, fi, :, c['normalizations'].index('shared_ce_scale')], cube[:, fi, :, c['normalizations'].index('frobenius_match')])
    pd.DataFrame(paired).to_csv(HERE / 'paired_performance_auc_advantages.csv', index=False)
    reuse = pd.concat(reused).value_counts().reset_index(name='rows').sort_values(['baseline_scores_reused', 'scaling_scores_reused']).reset_index(drop=True)
    archived = pd.read_csv(ANALYSIS / 'score_reuse_counts.csv').sort_values(['baseline_scores_reused', 'scaling_scores_reused']).reset_index(drop=True)
    pd.testing.assert_frame_equal(reuse, archived, check_dtype=False)
    receipt = dict(status='passed', verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source=str(source.relative_to(ROOT)),
                   source_sha256=digest, experiment_id=manifest['experiment_id'], rows=count, repeats=c['repeats'],
                   configurations=int(np.prod(shape[4:])), families=len(c['families']), fractions=len(x), normalizations=len(c['normalizations']),
                   original_analysis_hashes_verified=len(am['outputs']), scientific_source_hashes_verified=len(manifest['source_sha256']),
                   scientific_source_locations=source_locations,
                   checks=['complete unique factorial grid', 'all numeric fields finite', 'MC delay sums and bounds', 'NARMA NMSE/NRMSE',
                           'radius and Frobenius scaling', 'finite-vector moment matching', 'compression variance', 'mean curves',
                           'win curves', 'configuration advantages', 'both AUC point estimates', 'performance AUC intervals',
                           'identical shared/Frobenius scores where expected', 'score reuse counts'],
                   bootstrap=am['bootstrap'], bootstrap_seed=am['bootstrap_seed'], bootstrap_unit='paired whole realization',
                   validator_sha256=sha(__file__))
    (HERE / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
