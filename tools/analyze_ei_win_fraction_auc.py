#!/usr/bin/env python3
"""Integrate the fraction-of-hyperparameter-configurations-won curves."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.ei_auc import curve_areas, win_fraction_auc
from util.ei_win_fraction import pair_surfaces, bootstrap_counts, surface_matrix
from tools.plot_ei_win_fractions import load_sources, METRICS, FAMILIES, LABELS

OUT = ROOT / "analysis/ei_hyperparameter_win_fractions_2026-11-18"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ["win_fraction_auc_summary.csv", "WIN_FRACTION_AUC.md", "win_fraction_auc_manifest.json", "08_win_fraction_auc.png"]
    if any((OUT / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing win-fraction AUC outputs")
    previous_hashes = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    old = json.loads((OUT / "manifest.json").read_text())
    frames, sources = load_sources()
    assert {s['path']: s['sha256'] for s in sources} == {s['path']: s['sha256'] for s in old['sources']}
    plotted = pd.read_csv(OUT / "win_fraction_summary.csv", dtype={"input_scale": str})
    rng = np.random.default_rng(20261118)
    rows = []
    for experiment in ["sign", "normalization"]:
        er = frames[experiment, "matched_er"]
        seeds = np.sort(er.seed.unique())
        counts = bootstrap_counts(len(seeds), 2000, rng)
        for family in FAMILIES[:2]:
            paired, _ = pair_surfaces(frames[experiment, family], er, METRICS)
            for normalization, group in paired.groupby("normalization"):
                fractions = np.sort(group.negative_fraction.unique())
                assert fractions[0] == 0 and fractions[-1] == 1
                for metric in METRICS:
                    surfaces = []
                    grid = None
                    for fraction in fractions:
                        matrix, this_grid = surface_matrix(group[group.negative_fraction == fraction], metric, seeds)
                        if grid is not None:
                            pd.testing.assert_frame_equal(grid, this_grid)
                        grid = this_grid
                        surfaces.append(matrix)
                    y = np.stack(surfaces, axis=1)
                    subsets = [("ALL", np.ones(len(grid), dtype=bool))]
                    subsets += [(str(level), grid.input_scale.to_numpy() == level) for level in sorted(grid.input_scale.unique())]
                    for level, mask in subsets:
                        result = win_fraction_auc(y[:, :, mask], counts, fractions)
                        assert np.isclose(sum(result[f'{a}_win_auc_percent'] for a in ['ce', 'er', 'tie']), 100)
                        reference = plotted[(plotted.experiment == experiment) & (plotted.family == family)
                                            & (plotted.normalization == normalization) & (plotted.metric == metric)
                                            & (plotted.input_scale == level)].sort_values("negative_percent")
                        assert len(reference) == len(fractions)
                        # Independently integrate the exact data used for the previous PNG.
                        expected = curve_areas(reference.percent_ce_better.to_numpy(), reference.negative_percent.to_numpy()/100)[0]
                        np.testing.assert_allclose(result['ce_win_auc_percent'], expected, atol=1e-10)
                        rows.append(dict(experiment=experiment, family=family, normalization=normalization,
                                         metric=metric, input_scale=level, n_seeds=len(seeds),
                                         n_configurations=int(mask.sum()), **result))
    summary = pd.DataFrame(rows)
    assert len(summary) == 90
    summary.to_csv(OUT / "win_fraction_auc_summary.csv", index=False, float_format="%.15g")
    primary = summary[(summary.experiment == 'sign') & (summary.input_scale == 'ALL')]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.7), sharey=True)
    for ax, metric in zip(axes, METRICS):
        data = primary[primary.metric == metric].set_index('family').loc[FAMILIES[:2]]
        ce = data.ce_win_auc_percent.to_numpy()
        er = data.er_win_auc_percent.to_numpy()
        ax.bar([0, 1], ce, color='#009e73', label='CE wins')
        ax.bar([0, 1], er, bottom=ce, color='#d55e00', alpha=.6, label='ER wins')
        ax.errorbar([0, 1], ce, yerr=[np.maximum(0, ce-data.ce_ci_low), np.maximum(0, data.ce_ci_high-ce)],
                    fmt='none', color='black', capsize=3, lw=1)
        for i, (c, e) in enumerate(zip(ce, er)):
            ax.text(i, c/2, f'{c:.1f}%', ha='center', va='center', color='white')
            ax.text(i, c+e/2, f'{e:.1f}%', ha='center', va='center')
        ax.set(title=metric, ylim=(0, 100), xticks=[0, 1], xticklabels=['CE unknown\ncompleted', 'CE unknown\nremoved'])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Normalized area under win-fraction curve (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.suptitle('Area under the fraction-of-settings-won curve')
    fig.text(.5, .095, 'Uniform E/I weighting over 0–100%; 96 settings; 200 paired realizations; own-radius scaling; binary-weight ER.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .94))
    fig.savefig(OUT / '08_win_fraction_auc.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    lines = ['# AUC of the fraction-of-settings-won curves', '',
             'This integrates the percentage of hyperparameter configurations whose mean performance favors CE, not the mean-performance curve. The ordinate is percent settings won; the inhibitory fraction axis runs 0–1. A value of 50% means CE wins half the integrated configuration/E/I domain under this specified weighting.', '',
             '| CE variant | Metric | CE win AUC [95% interval] | ER win AUC |', '|---|---|---:|---:|']
    for r in primary.to_dict('records'):
        lines.append(f"| {LABELS[r['family']]} | {r['metric']} | {r['ce_win_auc_percent']:.2f}% [{r['ce_ci_low']:.2f}, {r['ce_ci_high']:.2f}] | {r['er_win_auc_percent']:.2f}% |")
    lines += ['', 'The CSV additionally includes each input scale and the separate own-radius/fixed-reference normalization-control runs. Archival KR controls retain their distinct provenance.', '',
              'Each bootstrap resamples whole paired seeds across every fraction and configuration, recomputes mean performance, counts strict wins, then integrates. Intervals are exploratory percentile intervals over 2,000 draws. Exact ties are tracked separately and excluded from both win counts. No main-sweep mean-surface ties occurred.', '',
              'Linear interpolation connects the six sampled points at 0, 20, 40, 60, 80 and 100%. Trapezoidal integration weights endpoints half as much as interior samples. This approximates the displayed curve; it does not observe unmeasured intermediate fractions. The uniform E/I weighting and equally weighted finite hyperparameter grid are analysis choices, not an empirical distribution of biological conditions.', '',
              'All positive differences count equally, including tiny improvements between poor reservoirs. These areas measure breadth of relative wins, not effect size, absolute computational usefulness or a biological robustness probability. The earlier performance-AUC results answer a different question and are preserved.']
    (OUT / 'WIN_FRACTION_AUC.md').write_text('\n'.join(lines)+'\n')
    assert all(sha(OUT / name) == digest for name, digest in previous_hashes.items())
    provenance = dict(sources=sources, n_bootstrap=2000, random_seed=20261118,
                      estimand='AUC of percent mean-surface hyperparameter wins, integrated over inhibitory fraction 0–1',
                      all_90_point_estimates_verified_against_original_plot_table=True,
                      all_existing_analysis_files_unchanged=True,
                      scripts=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in [Path(__file__).resolve(), ROOT/'util/ei_auc.py']],
                      outputs={name: sha(OUT/name) for name in names if name!='win_fraction_auc_manifest.json'})
    (OUT/'win_fraction_auc_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(primary[['family', 'metric', 'ce_win_auc_percent', 'ce_ci_low', 'ce_ci_high', 'er_win_auc_percent', 'tie_win_auc_percent']].to_string(index=False))


if __name__ == '__main__':
    main()
