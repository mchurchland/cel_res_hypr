#!/usr/bin/env python3
"""Add NARMA10 win-fraction curves from the completed 271800 smaller-grid run."""
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
from util.ei_auc import curve_areas
from tools.analyze_mc_narma_crossover import assemble, FAMILIES, NORMS, PARAMS, NORM_NAMES

BASE = ROOT / "analysis/crossover_mc_narma10_271800"
OUT = ROOT / "analysis/ei_hyperparameter_win_fractions_2026-11-18"
COLORS = {"cel_matched": "#d55e00", "matched_er": "#009e73"}
LABELS = {"cel_matched": "CE: unknown signs completed", "matched_er": "ER"}
INPUT_COLORS = ["#332288", "#0077bb", "#009988", "#ee7733", "#cc3311"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_axis(ax, percent=True):
    ax.set(xlabel="Inhibitory edges (%)", xlim=(-1, 51), xticks=np.arange(0, 51, 10))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.16)
    if percent:
        ax.set(ylabel="Hyperparameter settings won (%)", ylim=(0, 100), yticks=np.arange(0, 101, 20))
        ax.axhline(50, color="0.6", ls=":", lw=.8)


def save(fig, name, footer):
    fig.text(.5, .015, footer, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    output_names = ["narma10_win_fraction_sweep.png", "narma10_win_fraction_by_input.png",
                    "narma10_mean_error_sweep.png", "narma10_win_fraction_auc.png",
                    "narma10_win_fraction_summary.csv", "narma10_win_fraction_auc.csv",
                    "narma10_mean_error_summary.csv", "NARMA10_WIN_FRACTIONS.md", "narma10_win_fraction_manifest.json"]
    if any((OUT / name).exists() for name in output_names):
        raise FileExistsError("Refusing to overwrite existing NARMA win-fraction outputs")
    previous_hashes = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    manifest_path = BASE / "manifest.json"
    cache_path = BASE / "cache/bootstrap_surfaces.npz"
    manifest = json.loads(manifest_path.read_text())
    source = ROOT / "results/crossover_mc_narma10/271800/mc_narma_crossover.ALL.csv"
    assert sha(source) == manifest["source_files"][str(source.relative_to(ROOT))]
    raw = pd.read_csv(source)
    assert len(raw) == 72000
    spec = manifest["experiment_spec"]
    panel, _, _, ei, grid, _ = assemble(raw, spec)
    with np.load(cache_path) as z:
        mean, boot = z["mean"], z["bootstrap"]
        np.testing.assert_allclose(mean, panel.mean(axis=0), atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(ei, z["ei_percent"])
        np.testing.assert_allclose(grid.to_numpy(), z["grid"], rtol=0, atol=1e-12)
    assert panel.shape == (100, 2, 2, 2, 6, 30)
    assert boot.shape == (2000, 2, 2, 2, 6, 30)
    reference = pd.read_csv(BASE / "tables/configuration_win_counts.csv")
    rows, areas, means = [], [], []
    for n, normalization in enumerate(NORMS):
        advantage = mean[1, n, 1] - mean[0, n, 1]  # Lower NARMA NMSE is better.
        boot_advantage = boot[:, 1, n, 1] - boot[:, 0, n, 1]
        subsets = [("ALL", np.ones(len(grid), dtype=bool))]
        subsets += [(str(level), np.isclose(grid.input_scale, level)) for level in sorted(grid.input_scale.unique())]
        for level, mask in subsets:
            wins = 100 * (advantage[:, mask] > 0).mean(axis=-1)
            losses = 100 * (advantage[:, mask] < 0).mean(axis=-1)
            ties = 100 - wins - losses
            boot_wins = 100 * (boot_advantage[..., mask] > 0).mean(axis=-1)
            boot_losses = 100 * (boot_advantage[..., mask] < 0).mean(axis=-1)
            ci = np.quantile(boot_wins, [.025, .975], axis=0)
            er_ci = np.quantile(boot_losses, [.025, .975], axis=0)
            for j, fraction in enumerate(ei):
                rows.append(dict(normalization=normalization, ei_percent=fraction, input_scale=level,
                                 ce_win_percent=wins[j], er_win_percent=losses[j], tie_percent=ties[j],
                                 ci_low=ci[0, j], ci_high=ci[1, j], er_ci_low=er_ci[0, j], er_ci_high=er_ci[1, j],
                                 n_configurations=int(mask.sum()), n_realizations=100))
                if level == "ALL":
                    ref = reference[(reference.normalization == normalization) & (reference.metric == "NARMA10_test_nmse")
                                    & np.isclose(reference.ei_percent, fraction)].iloc[0]
                    assert np.isclose(wins[j] / 100 * mask.sum(), ref.ce_positive_mean)
            coordinates = (ei - ei.min()) / (ei.max() - ei.min())
            ce_auc = curve_areas(wins, coordinates)[0]
            er_auc = curve_areas(losses, coordinates)[0]
            lo, hi = np.quantile(curve_areas(boot_wins, coordinates)[0], [.025, .975])
            areas.append(dict(normalization=normalization, input_scale=level, fraction_min_percent=float(ei.min()),
                              fraction_max_percent=float(ei.max()), ce_win_auc_percent=ce_auc, er_win_auc_percent=er_auc,
                              ci_low=lo, ci_high=hi, n_configurations=int(mask.sum()), n_realizations=100))
        for f, family in enumerate(FAMILIES):
            curve = mean[f, n, 1].mean(axis=-1)
            ci = np.quantile(boot[:, f, n, 1].mean(axis=-1), [.025, .975], axis=0)
            for j, fraction in enumerate(ei):
                # Independent average from the raw stored rows.
                direct = raw[(raw.family == family) & (raw.normalization == normalization)
                             & np.isclose(raw.negative_fraction, fraction/100)].NARMA10_test_nmse.mean()
                np.testing.assert_allclose(curve[j], direct, atol=1e-12, rtol=1e-12)
                means.append(dict(normalization=normalization, family=family, ei_percent=fraction,
                                  mean_test_nmse=curve[j], ci_low=ci[0, j], ci_high=ci[1, j],
                                  n_configurations=30, n_realizations=100))
    summary, auc, performance = pd.DataFrame(rows), pd.DataFrame(areas), pd.DataFrame(means)
    summary.to_csv(OUT / "narma10_win_fraction_summary.csv", index=False, float_format="%.15g")
    auc.to_csv(OUT / "narma10_win_fraction_auc.csv", index=False, float_format="%.15g")
    performance.to_csv(OUT / "narma10_mean_error_summary.csv", index=False, float_format="%.15g")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, normalization in zip(axes, NORMS):
        g = summary[(summary.normalization == normalization) & (summary.input_scale == "ALL")]
        for family, value, lo, hi in [("cel_matched", "ce_win_percent", "ci_low", "ci_high"),
                                     ("matched_er", "er_win_percent", "er_ci_low", "er_ci_high")]:
            ax.plot(g.ei_percent, g[value], "o-", color=COLORS[family], label=LABELS[family], lw=1.8, ms=5)
            ax.fill_between(g.ei_percent, g[lo], g[hi], color=COLORS[family], alpha=.13, linewidth=0)
        base_axis(ax)
        ax.set_title(NORM_NAMES[normalization])
    axes[0].legend(frameon=False)
    fig.suptitle("NARMA10: fraction of matched hyperparameter settings won", y=.99)
    save(fig, "narma10_win_fraction_sweep.png", "30 settings; 100 paired realizations; lower mean test NMSE wins. Shading: pointwise 95% whole-seed bootstrap.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, normalization in zip(axes, NORMS):
        for color, level in zip(INPUT_COLORS, sorted(grid.input_scale.unique())):
            g = summary[(summary.normalization == normalization) & (summary.input_scale == str(level))]
            ax.plot(g.ei_percent, g.ce_win_percent, "o-", color=color, label=f"Input scale {level:g}", lw=1.5, ms=4)
            ax.fill_between(g.ei_percent, g.ci_low, g.ci_high, color=color, alpha=.09, linewidth=0)
        base_axis(ax)
        ax.set(ylabel="Hyperparameter settings where CE wins (%)", title=NORM_NAMES[normalization])
    axes[0].legend(frameon=False)
    fig.suptitle("NARMA10: CE wins conditional on input scale", y=.99)
    save(fig, "narma10_win_fraction_by_input.png", "6 settings per input level; 100 paired realizations; lower mean test NMSE wins; bias fixed at zero.")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    for ax, normalization in zip(axes, NORMS):
        for family in FAMILIES:
            g = performance[(performance.normalization == normalization) & (performance.family == family)]
            ax.plot(g.ei_percent, g.mean_test_nmse, "o-", color=COLORS[family], label=LABELS[family], lw=1.8, ms=5)
            ax.fill_between(g.ei_percent, g.ci_low, g.ci_high, color=COLORS[family], alpha=.13, linewidth=0)
        base_axis(ax, percent=False)
        ax.set(ylabel="Mean NARMA10 test NMSE (lower is better)", title=NORM_NAMES[normalization], ylim=(0, performance.ci_high.max()*1.08))
    axes[0].legend(frameon=False)
    fig.suptitle("NARMA10: mean test error across the observed E/I sweep", y=.99)
    save(fig, "narma10_mean_error_sweep.png", "30 equally weighted settings; 100 paired realizations; 95% seed-bootstrap intervals. All test errors retained, including the large outlier.")

    fig, ax = plt.subplots(figsize=(6.7, 4.7))
    all_auc = auc[auc.input_scale == "ALL"].set_index("normalization").loc[NORMS]
    ce, er = all_auc.ce_win_auc_percent.to_numpy(), all_auc.er_win_auc_percent.to_numpy()
    ax.bar([0, 1], ce, color=COLORS["cel_matched"], label="CE wins")
    ax.bar([0, 1], er, bottom=ce, color=COLORS["matched_er"], label="ER wins")
    ax.errorbar([0, 1], ce, yerr=[np.maximum(0, ce-all_auc.ci_low), np.maximum(0, all_auc.ci_high-ce)],
                fmt="none", color="black", capsize=3, lw=1)
    for i, (c, e) in enumerate(zip(ce, er)):
        ax.text(i, c/2, f"{c:.1f}%", ha="center", va="center", color="white")
        ax.text(i, c+e/2, f"{e:.1f}%", ha="center", va="center", color="white")
    ax.set(ylim=(0, 100), xticks=[0, 1], xticklabels=[NORM_NAMES[n] for n in NORMS],
           ylabel="Normalized area under win-fraction curve (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.suptitle("NARMA10: integrated win fraction over 0–50% inhibition", y=.99)
    save(fig, "narma10_win_fraction_auc.png", "Observed 0–50% range only; normalized by that range; 30 settings and 100 realizations. Do not compare directly to the archived 0–100% AUC.")
    main = summary[summary.input_scale == "ALL"]
    lines = ["# NARMA10 win fractions on the existing smaller grid", "",
             "Source run 271800: 100 paired realizations, fractions 0/10/20/30/40/50%, radius targets 0.6/0.95/1.05, leaks 0.6/1.0, inputs 0.001/0.01/0.05/0.1/0.5, bias fixed to zero: 30 settings. Only the completed-unknown-sign CE variant and binary ER were evaluated. Own-radius and fixed-reference protocols remain separate.", "",
             "A configuration is a CE win when mean CE test NMSE across the 100 realizations is strictly lower than mean ER test NMSE at that identical configuration. Exact ties count as neither. It is not a significance-filtered count. Readout ridge is selected on validation data, but no reservoir configuration is selected for these win-fraction or grid-mean plots.", "",
             "| Normalization | Inhibitory edges | CE wins | ER wins |", "|---|---:|---:|---:|"]
    for r in main.to_dict('records'):
        lines.append(f"| {NORM_NAMES[r['normalization']]} | {r['ei_percent']:g}% | {r['ce_win_percent']:.1f}% | {r['er_win_percent']:.1f}% |")
    lines += ["", "## Scope and validation", "",
              "- All mean surfaces and matched grid coordinates agree with the independently reconstructed raw-data panel. All 12 overall win counts match the existing completed-run analysis. Bootstrap surfaces are reused from that analysis (2,000 whole-paired-realization draws, seed 271601), retaining all conditions together.",
              "- Input-conditional curves have six settings each. Normalized AUC integrates the win fraction over the observed 0–50% range, divided by that range. This is not the full 0–100% archived AUC, and the grid differs. No mirroring, extrapolation or pooling with the 96-setting experiments.",
              "- All test errors are retained. A CE own-radius observation at 40% inhibition has NMSE about 604 and strongly influences some mean comparisons. The mean-error figure is untrimmed; see the original run's narma_seed_tail_diagnostics.csv for median and leave-one-seed-out diagnostics.",
              "- Win frequency measures breadth of relative advantage, not its size or absolute usefulness. Seeds are computational realizations, not biological animals. E/I here is an assigned edge-sign fraction.",
              "- Existing source and analysis files are preserved; all new figures are PNGs. No new simulations."]
    (OUT / 'NARMA10_WIN_FRACTIONS.md').write_text('\n'.join(lines)+'\n')
    assert all(sha(OUT/name) == digest for name, digest in previous_hashes.items())
    provenance = dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                      analysis_manifest_sha256=sha(manifest_path), reused_bootstrap_cache_sha256=sha(cache_path),
                      script_sha256=sha(Path(__file__)), experiment_spec=spec,
                      definitions="strict lower mean test NMSE wins; mean across paired seeds before comparing; no test-score reservoir tuning",
                      extrapolation=False, source_rows_filtered=0, existing_analysis_files_unchanged=True,
                      validation="raw and cached means/grid match; 12 overall counts match existing configuration_win_counts.csv",
                      outputs={name: sha(OUT/name) for name in output_names if name!='narma10_win_fraction_manifest.json'})
    (OUT/'narma10_win_fraction_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(main[['normalization', 'ei_percent', 'ce_win_percent', 'er_win_percent']].to_string(index=False))
    print(all_auc[['ce_win_auc_percent', 'er_win_auc_percent', 'ci_low', 'ci_high']].to_string())


if __name__ == '__main__':
    main()
