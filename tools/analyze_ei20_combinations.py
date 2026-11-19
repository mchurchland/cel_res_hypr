#!/usr/bin/env python3
"""Inspect full hyperparameter tuples and matched parameter changes at 20% inhibition."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-robustness')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/robustness-xdg-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from util.ei_win_fraction import PARAMETERS

BASE = ROOT/'analysis/ei_hyperparameter_win_fractions_2026-11-18/loss_settings_2026-11-19'
OUT = BASE/'ei20_combinations'
META = ['dataset', 'family', 'normalization', 'metric']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_plot(ax, data, metric, title, limit):
    columns = ['input_scale', 'neuron_bias'] if data.neuron_bias.nunique() > 1 else ['input_scale']
    matrix = data.pivot(index=['rho_target', 'leak'], columns=columns, values='ce_advantage').sort_index()
    assert np.isfinite(matrix.to_numpy()).all()
    image = ax.imshow(matrix, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f'{value:+.3f}' if metric.startswith('NARMA') else f'{value:+.2f}',
                ha='center', va='center', fontsize=8, color='white' if abs(value) > .65*limit else 'black')
    labels = [f'{s:g}\nb={b:g}' for s, b in matrix.columns] if len(columns) == 2 else [f'{v:g}' for v in matrix.columns]
    ax.set(xticks=range(matrix.shape[1]), xticklabels=labels,
           yticks=range(matrix.shape[0]), yticklabels=[f'{r:g} / {l:g}' for r, l in matrix.index],
           xlabel='Input scale / bias' if len(columns) == 2 else 'Input scale', ylabel='Radius target / leak', title=title)
    return image


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    source = BASE/'all_configuration_comparisons.csv'
    old = json.loads((BASE/'manifest.json').read_text())
    assert sha(source) == old['outputs'][source.name]
    d = pd.read_csv(source)
    d = d[d.ei_percent == 20].copy()
    assert len(d) == 1848
    assert not d.duplicated([*META, *PARAMETERS]).any()
    flips, flip_details = [], []
    for meta, group in d.groupby(META):
        for parameter in PARAMETERS:
            low, high = group[parameter].min(), group[parameter].max()
            if low == high:
                continue
            others = [p for p in PARAMETERS if p != parameter]
            pairs = group[group[parameter] == low].merge(group[group[parameter] == high], on=others, suffixes=('_low', '_high'), validate='one_to_one')
            assert len(pairs) == len(group[group[parameter] == low])
            ce_to_er = (pairs.ce_advantage_low > 0) & (pairs.ce_advantage_high < 0)
            er_to_ce = (pairs.ce_advantage_low < 0) & (pairs.ce_advantage_high > 0)
            record = dict(zip(META, meta), parameter=parameter, low_level=low, high_level=high)
            flips.append(dict(**record, n_matched_changes=len(pairs), ce_win_to_er_win=int(ce_to_er.sum()),
                              er_win_to_ce_win=int(er_to_ce.sum()),
                              mean_advantage_change=(pairs.ce_advantage_high-pairs.ce_advantage_low).mean()))
            detail = pairs[others+['ce_advantage_low', 'ce_advantage_high']].copy()
            for key, value in record.items():
                detail[key] = value
            detail['ce_win_to_er_win'], detail['er_win_to_ce_win'] = ce_to_er.to_numpy(), er_to_ce.to_numpy()
            flip_details.append(detail)
    summary = pd.DataFrame(flips)
    main = d[(d.dataset == 'final_results_sign') & (d.family == 'cel_matched')]
    # Exact rule checks for the completed-sign CE architecture at this fraction.
    mc = main[main.metric == 'MC']
    assert not mc[mc.leak == .6].ce_loses.any()
    assert mc[mc.leak == .8].ce_loses.sum() == 1
    assert mc[mc.leak == 1].ce_loses.sum() == 22
    narma = d[d.dataset == 'narma_271800']
    for metric, group in narma.groupby('metric'):
        own = group[group.normalization == 'spectral_radius']
        assert not own[own.input_scale < .5].ce_loses.any()
        assert own[own.input_scale == .5].ce_loses.sum() == 5
        fixed = group[group.normalization == 'original_radius']
        assert fixed[fixed.rho_target == .6].ce_loses.all()
        assert not fixed[fixed.rho_target == 1.05].ce_loses.any()
    OUT.mkdir()
    d.to_csv(OUT/'exact_comparisons_ei20.csv', index=False, float_format='%.15g')
    summary.to_csv(OUT/'matched_parameter_flips_at_ei20.csv', index=False, float_format='%.15g')
    pd.concat(flip_details, ignore_index=True).to_csv(OUT/'matched_parameter_change_details.csv', index=False, float_format='%.15g')
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11})
    for family, label in [('cel_matched', 'Unknown signs completed'), ('cel_removed', 'Unknown-sign edges removed')]:
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 7), layout='constrained')
        for ax, metric in zip(axes, ['MC', 'IPC', 'KR']):
            g = d[(d.dataset == 'final_results_sign') & (d.family == family) & (d.metric == metric)]
            limit = g.ce_advantage.abs().max()
            image = matrix_plot(ax, g, metric, metric, limit)
            fig.colorbar(image, ax=ax, shrink=.6, label='CE − ER mean performance')
        fig.suptitle(f'20% inhibitory edges — {label}\nArchived 96-setting grid, own-radius scaling: positive favors CE; negative favors ER')
        fig.savefig(OUT/f'final_results_{family}_ei20_combinations.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, metric in enumerate(['MC', 'NARMA10_test_nmse']):
        limit = narma[narma.metric == metric].ce_advantage.abs().max()
        for i, (norm, title) in enumerate([('spectral_radius', 'Own-radius'), ('original_radius', 'Fixed-reference')]):
            g = narma[(narma.metric == metric) & (narma.normalization == norm)]
            image = matrix_plot(axes[i,j], g, metric, f'{title}: {metric.replace("NARMA10_test_nmse", "NARMA10")}', limit)
        fig.colorbar(image, ax=axes[:,j].tolist(), shrink=.7, label='CE advantage (MC: CE−ER; NARMA: ER−CE error)')
    fig.suptitle('20% inhibitory edges — MC/NARMA 30-setting grid\nBias = 0; positive favors CE; negative favors ER; mean differences over 100 realizations')
    fig.savefig(OUT/'narma_run_ei20_combinations.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    lines = ['# Exact hyperparameter combinations at 20% inhibitory edges', '',
             'All statements below describe the sign of paired mean differences, not significance-filtered outcomes. Exact differences and existing uncertainty/support fields are in exact_comparisons_ei20.csv. Archived normalization controls and both CE variants are included in the CSV, while the figures emphasize the main archived sweep and newer MC/NARMA experiment.', '',
             '## Archived completed-sign CE: MC', '',
             '- Leak 0.6: wins all 32 combinations of radius, input and bias.',
             '- Leak 0.8: wins 31/32. The only loss is radius 1.05, input 1.5, bias 0.1.',
             '- Leak 1: wins 10/32. Wins every radius at input 0.1, either bias; also wins radius 0.6 at input 0.5, either bias. All other leak-1 settings lose.',
             '- IPC exceptions are shown explicitly in the heatmap: leak 0.6 wins 30/32, leak 0.8 wins 26/32, leak 1 wins 7/32. KR favors CE for all 96 settings at this fraction.', '',
             '## Newer MC/NARMA run', '',
             '- The two metrics have identical win/loss classifications across all 30 settings at 20% within each normalization protocol.',
             '- Own-radius: inputs 0.001, 0.01, 0.05 and 0.1 win for every radius/leak combination. At input 0.5 only radius 0.6 + leak 0.6 wins; the other five combinations lose.',
             '- Fixed-reference: radius 0.6 loses at every input and both leaks; radius 0.95 wins at inputs <=0.1 and loses at 0.5; radius 1.05 wins at every input and both leaks.', '',
             '## Matched changes', '',
             'matched_parameter_flips_at_ei20.csv changes one parameter between its lowest and highest tested levels while holding all other parameters fixed. It records CE-win→ER-win and reverse transitions. This is not an ANOVA attribution, nor does the number of transitions identify a unique dynamical mechanism. The counts depend on the tested ranges; the two experiments have different grids.', '',
             'No new simulations, source changes, test-score optimization or pooling across grids. Binary ER remains the comparison; this is not the newer weight-matched ER study. Larger KR denotes separation rather than general task superiority.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    provenance = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source), script_sha256=sha(Path(__file__)),
                      focus_inhibitory_percent=20, original_comparison_rows=1848,
                      rule_checks='Complete-grid own-radius and fixed-reference rules verified for both metrics',
                      outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT/'manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(OUT)
    print(summary[(summary.family == 'cel_matched') & summary.dataset.isin(['final_results_sign','narma_271800'])].to_string(index=False))


if __name__ == '__main__':
    main()
