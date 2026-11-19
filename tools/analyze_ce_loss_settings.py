#!/usr/bin/env python3
"""Describe exactly which matched configurations favor ER in the two sweeps."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-robustness')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.ei_win_fraction import PARAMETERS

BASE = ROOT/'analysis/ei_hyperparameter_win_fractions_2026-11-18'
OUT = BASE/'loss_settings_2026-11-19'
META = ['dataset', 'family', 'normalization', 'metric']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    final_path = BASE/'configuration_comparisons.csv'
    narma_path = ROOT/'analysis/crossover_mc_narma10_271800/tables/configuration_advantages.csv'
    final_manifest = json.loads((BASE/'manifest.json').read_text())
    assert sha(final_path) == final_manifest['outputs'][final_path.name]
    for source in final_manifest['sources']:
        assert sha(ROOT/source['path']) == source['sha256']
    f = pd.read_csv(final_path).rename(columns={'negative_percent': 'ei_percent', 'mean_ce_minus_er': 'ce_advantage'})
    f['dataset'] = f.experiment.map({'sign': 'final_results_sign', 'normalization': 'final_results_normalization'})
    f['rank_provenance'] = np.where(f.experiment == 'sign', 'updated', 'archival_unverified')
    f['n_realizations'] = f.n_paired_seeds
    n = pd.read_csv(narma_path)
    n['dataset'], n['family'], n['neuron_bias'], n['n_realizations'] = 'narma_271800', 'cel_matched', 0., 100
    n['rank_provenance'] = 'not_applicable'
    # Independently check NARMA-run contrasts against raw mean performance.
    raw_path = ROOT/'results/crossover_mc_narma10/271800/mc_narma_crossover.ALL.csv'
    original = json.loads((ROOT/'analysis/crossover_mc_narma10_271800/manifest.json').read_text())
    assert sha(raw_path) == original['source_files'][str(raw_path.relative_to(ROOT))]
    raw = pd.read_csv(raw_path)
    keys = ['normalization', 'negative_fraction', *PARAMETERS]
    means = raw.groupby(['family', *keys])[['MC', 'NARMA10_test_nmse']].mean()
    direct = means.loc['cel_matched'].join(means.loc['matched_er'], lsuffix='_ce', rsuffix='_er').reset_index()
    direct['ei_percent'] = direct.negative_fraction * 100
    for metric, sign in [('MC', 1), ('NARMA10_test_nmse', -1)]:
        check = n[n.metric == metric].merge(direct, on=['normalization', 'ei_percent', *PARAMETERS], validate='one_to_one')
        assert len(check) == 360
        np.testing.assert_allclose(check.ce_advantage, sign*(check[f'{metric}_ce']-check[f'{metric}_er']), rtol=1e-11, atol=1e-11)
    d = pd.concat([f, n], ignore_index=True)
    assert len(d) == 11088
    assert np.isfinite(d[['ce_advantage', 'ei_percent', *PARAMETERS]].to_numpy(float)).all()
    assert not d.duplicated([*META, 'ei_percent', *PARAMETERS]).any()
    d['ce_loses'] = d.ce_advantage < 0
    d['ce_wins'] = d.ce_advantage > 0
    d['tie'] = d.ce_advantage == 0
    d['ei_region'] = np.where((d.ei_percent <= 20) | (d.ei_percent >= 80), 'imbalanced_0_20_or_80_100', 'interior_20_80_exclusive')
    sources = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in [final_path, narma_path, raw_path]]
    return d, sources


def count_summary(d, keys):
    result = d.groupby(keys, dropna=False).agg(
        n_configurations=('ce_advantage', 'size'), ce_losses=('ce_loses', 'sum'),
        ce_wins=('ce_wins', 'sum'), ties=('tie', 'sum'),
        mean_ce_advantage=('ce_advantage', 'mean'), median_ce_advantage=('ce_advantage', 'median')).reset_index()
    result['ce_loss_percent'] = 100*result.ce_losses/result.n_configurations
    assert (result.ce_losses+result.ce_wins+result.ties == result.n_configurations).all()
    return result


def summaries(d):
    rows = []
    for parameter in PARAMETERS:
        for scope, extra in [('per_ei', ['ei_percent']), ('by_ei_region', ['ei_region']), ('all_fractions', [])]:
            g = count_summary(d, [*META, *extra, parameter]).rename(columns={parameter: 'level'})
            g['parameter'], g['scope'] = parameter, scope
            rows.append(g)
    marginal = pd.concat(rows, ignore_index=True)
    joint = count_summary(d, [*META, 'ei_region', 'leak', 'input_scale'])
    overall = count_summary(d, [*META, 'ei_percent'])
    return marginal, joint, overall


def heatmap(ax, d, row, column, title):
    values = count_summary(d, [row, column])
    rows, columns = sorted(values[row].unique()), sorted(values[column].unique())
    value = values.pivot(index=row, columns=column, values='ce_loss_percent').reindex(index=rows, columns=columns)
    losses = values.pivot(index=row, columns=column, values='ce_losses').reindex(index=rows, columns=columns)
    counts = values.pivot(index=row, columns=column, values='n_configurations').reindex(index=rows, columns=columns)
    image = ax.imshow(value, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, f'{int(losses.iloc[i,j])}/{int(counts.iloc[i,j])}', ha='center', va='center',
                    fontsize=9, color='white' if value.iloc[i,j] >= 60 else 'black')
    ax.set(xticks=range(len(columns)), xticklabels=[f'{v:g}' for v in columns],
           yticks=range(len(rows)), yticklabels=[f'{v:g}' for v in rows],
           xlabel='Input scale', ylabel='Inhibitory edges (%)' if row == 'ei_percent' else 'Leak rate', title=title)
    return image


def plots(d):
    for dataset, metrics, rows, filename in [
        ('final_results_sign', ['MC', 'IPC', 'KR'], ['cel_matched', 'cel_removed'], 'final_results_losses_by_input.png'),
        ('narma_271800', ['MC', 'NARMA10_test_nmse'], ['spectral_radius', 'original_radius'], 'narma_run_losses_by_input.png')]:
        fig, axes = plt.subplots(2, len(metrics), figsize=(4.4*len(metrics), 7.2), layout='constrained')
        for i, value in enumerate(rows):
            for j, metric in enumerate(metrics):
                g = d[(d.dataset == dataset) & (d.metric == metric)]
                if dataset == 'final_results_sign':
                    g = g[g.family == value]
                    label = 'CE unknown signs completed' if value == 'cel_matched' else 'CE unknown-sign edges removed'
                else:
                    g = g[g.normalization == value]
                    label = 'Own-radius scaling' if value == 'spectral_radius' else 'Fixed-reference scaling'
                image = heatmap(axes[i,j], g, 'ei_percent', 'input_scale', f'{label}\n{metric.replace("NARMA10_test_nmse", "NARMA10")}')
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='Settings where ER beats CE (%)')
        fig.suptitle('CE losses at identical hyperparameters — cell labels: losses / tested settings')
        fig.savefig(OUT/filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), layout='constrained')
    for ax, metric in zip(axes, ['MC', 'IPC', 'KR']):
        g = d[(d.dataset == 'final_results_sign') & (d.family == 'cel_matched') & (d.metric == metric)
              & (d.ei_region == 'imbalanced_0_20_or_80_100')]
        image = heatmap(ax, g, 'leak', 'input_scale', metric)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.75, label='Settings where ER beats CE (%)')
    fig.suptitle('Completed-sign CE at 0, 20, 80 and 100% inhibition: losses / tested settings')
    fig.savefig(OUT/'final_results_input_leak_interaction_extremes.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def report(d, marginal):
    lines = ['# Which hyperparameter settings does CE lose on?', '',
             'A loss means the paired seed-averaged advantage is strictly negative: CE−ER for MC/IPC/KR, ER−CE test NMSE for NARMA. These are descriptive counts over fixed grid settings, not significance-filtered losses or independent biological samples. Exact losing tuples, including every E/I fraction and parameter, are in losing_configurations.csv.', '',
             'The archived analysis includes both CE variants, the 200-realization sign sweep and the separate 100-realization normalization controls. The newer MC/NARMA run uses 100 realizations, 30 settings, fixed zero bias and fractions 0–50%. These grids and normalization experiments are never pooled.', '',
             '## Input scale at 20% inhibition', '',
             '| Dataset | CE variant | Scaling | Metric | Input | Losing settings |', '|---|---|---|---|---:|---:|']
    view = marginal[(marginal.scope == 'per_ei') & (marginal.ei_percent == 20) & (marginal.parameter == 'input_scale')
                    & (marginal.family == 'cel_matched') & marginal.dataset.isin(['final_results_sign', 'narma_271800'])]
    for r in view.to_dict('records'):
        lines.append(f"| {r['dataset']} | {r['family']} | {r['normalization']} | {r['metric']} | {r['level']:g} | {r['ce_losses']}/{r['n_configurations']} ({r['ce_loss_percent']:.1f}%) |")
    lines += ['', '## Main interpretation', '',
              '- In the archived own-radius MC sweep, CE has no losses at input 0.1 at the four imbalanced fractions (0, 20, 80, 100%). At those fractions, losses concentrate at larger input and leak 1. IPC has exceptions even at input 0.1 and is not monotonically worse with input.',
              '- At 40% and 60% inhibition in that archived sweep, CE loses on almost all MC/IPC settings regardless of input; all completed-sign CE KR settings lose there. Input alone cannot explain the overall pattern.',
              '- In the newer own-radius run, all MC and NARMA losses at 0/10/20% occur at input 0.5; every tested input 0.001–0.1 wins. At 20%, five of six input-0.5 settings lose on both metrics. The sole winner is radius 0.6, leak 0.6.',
              '- Under fixed-reference scaling, low target radius matters: at 20%, every radius-0.6 setting loses on both metrics, across all five inputs and both leaks. Thus the high-input-only explanation does not transfer across normalization protocols.',
              '- At 40% and 50% in the newer own-radius run, ER wins every MC configuration. NARMA losses remain widespread but include exceptions, so conclusions should remain metric-specific.', '',
              '## Scope and files', '',
              '- hyperparameter_loss_summary.csv: separate loss percentages for each parameter level at each E/I fraction, within explicit E/I regions, and over each sampled sweep. E/I regions are exploratory summaries: imbalanced <=20% or >=80%; interior strictly between 20% and 80%. In the NARMA run these groups contain 0/10/20% and 30/40/50%, respectively.',
              '- input_leak_loss_summary.csv: joint input/leak loss rates within each E/I region. Marginal rates average the other finite grid levels and do not isolate a biological mechanism.',
              '- all_configuration_comparisons.csv and losing_configurations.csv: exact tuples. final_results sign KR is updated effective rank; normalization-control KR has archival/unverified protocol provenance and stays separate.',
              '- Most_negative_advantages.csv lists up to five losses per metric, fraction, normalization and CE family, sorted within each group. Different metric magnitudes are not comparable.',
              '- Binary-weight ER is the control in these sign/crossover experiments, not a CE-weight-matched ER. Do not generalize these contrasts to the weighted-control study.',
              '- All NARMA test errors are retained. A large CE test error at 40%, input 0.01, radius 1.05, leak 0.6 influences the corresponding mean comparison. Existing pointwise paired intervals are retained where available; no new significance threshold is applied.',
              '- No new simulations, source edits, outlier filtering, PDFs or overwritten previous analyses.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    d, sources = load()
    marginal, joint, overall = summaries(d)
    losses = d[d.ce_loses].sort_values([*META, 'ei_percent', 'ce_advantage'])
    worst = losses.groupby([*META, 'ei_percent'], sort=False).head(5)
    OUT.mkdir()
    for name, frame in [('all_configuration_comparisons', d), ('losing_configurations', losses),
                        ('hyperparameter_loss_summary', marginal), ('input_leak_loss_summary', joint),
                        ('overall_loss_summary', overall), ('Most_negative_advantages', worst)]:
        frame.to_csv(OUT/f'{name}.csv', index=False, float_format='%.15g')
    plots(d)
    report(d, marginal)
    provenance = dict(sources=sources, script_sha256=sha(Path(__file__)),
                      definition='strict negative advantage of paired seed-averaged performance; no significance filter',
                      n_configuration_comparisons=len(d), n_losing_configurations=len(losses),
                      original_source_and_derived_tables_verified=True,
                      narma_720_contrasts_verified_against_raw_means=True,
                      bootstrap='No new bootstrap; existing intervals/support retained in exact configuration CSV.',
                      outputs={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    (OUT/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(OUT)
    view = marginal[(marginal.scope == 'per_ei') & (marginal.ei_percent == 20) & (marginal.parameter == 'input_scale')
                    & (marginal.family == 'cel_matched') & marginal.dataset.isin(['final_results_sign', 'narma_271800'])]
    print(view[['dataset', 'normalization', 'metric', 'level', 'ce_losses', 'n_configurations', 'ce_loss_percent']].to_string(index=False))


if __name__ == '__main__':
    main()
