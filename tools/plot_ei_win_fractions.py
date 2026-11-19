#!/usr/bin/env python3
"""Plot archived CE versus ER matched-hyperparameter win fractions; PNG only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.ei_win_fraction import PARAMETERS, pair_surfaces, bootstrap_counts, summarize_surface, surface_matrix

METRICS = ["MC", "IPC", "KR"]
FAMILIES = ["cel_matched", "cel_removed", "matched_er"]
LABELS = {"cel_matched": "CE: unknown signs completed", "cel_removed": "CE: unknown-sign edges removed"}
COLORS = {"cel_matched": "#d55e00", "cel_removed": "#0072b2"}
INPUT_COLORS = {0.1: "#332288", 0.5: "#009988", 1.0: "#DDCC77", 1.5: "#CC6677"}
NORMS = {"spectral_radius": "Own-radius scaling", "original_radius": "Fixed-reference scaling"}


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_sources():
    frames, sources = {}, []
    for experiment in ["sign", "normalization"]:
        for family in FAMILIES:
            rel = (f"final_results/sign_frac/{family}/combined.ALL.GRKR_erank.rank_updated.csv"
                   if experiment == "sign" else f"final_results/Rho_stuff/sign_norm/{family}/combined.ALL.csv")
            path = ROOT / rel
            frame = pd.read_csv(path)
            fraction = frame["mode"].str.extract(r"^(?:sign_test_og_cel|sign_test_er)([0-9.]+)(?:__norm_.*)?$")[0]
            if fraction.isna().any():
                raise ValueError(f"Unrecognized mode identity in {rel}")
            frame["negative_fraction"] = fraction.astype(float)
            identity = ["normalization", "negative_fraction", "seed"]
            if frame.groupby(identity).shuffle_id.nunique().max() != 1:
                raise ValueError(f"Multiple network realizations per seed in {rel}")
            frames[experiment, family] = frame
            sources.append(dict(path=rel, sha256=sha(path), n_rows=len(frame),
                                experiment=experiment, family=family,
                                grid={p: sorted(frame[p].unique().tolist()) for p in PARAMETERS},
                                rank_provenance="updated effective rank" if experiment == "sign" else "archival rank protocol unverified"))
    return frames, sources


def calculate(frames, n_boot, random_seed):
    rng = np.random.default_rng(random_seed)
    summaries, configurations, audits, per_seed_records = [], [], [], []
    for experiment in ["sign", "normalization"]:
        er = frames[experiment, "matched_er"]
        seeds = np.sort(er.seed.unique())
        counts = bootstrap_counts(len(seeds), n_boot, rng)
        for family in FAMILIES[:2]:
            ce = frames[experiment, family]
            paired, audit = pair_surfaces(ce, er, METRICS)
            audits.append(audit.assign(experiment=experiment, family=family))
            for (norm, fraction), group in paired.groupby(["normalization", "negative_fraction"]):
                meta = dict(experiment=experiment, family=family, normalization=norm, negative_percent=100 * fraction)
                for metric in METRICS:
                    matrix, grid = surface_matrix(group, metric, seeds)
                    subsets = [("ALL", np.ones(len(grid), dtype=bool))]
                    subsets.extend((str(level), grid.input_scale.to_numpy() == level) for level in sorted(grid.input_scale.unique()))
                    for input_label, subset in subsets:
                        summary, means, support, per_seed = summarize_surface(matrix[:, subset], counts)
                        summaries.append(dict(**meta, metric=metric, input_scale=input_label, **summary))
                        if input_label == "ALL":
                            config = grid.copy()
                            for key, value in {**meta, "metric": metric}.items():
                                config[key] = value
                            config["mean_ce_minus_er"] = means
                            config["ce_better"] = means > 0
                            config["bootstrap_positive_fraction"] = support
                            config["n_paired_seeds"] = len(seeds)
                            configurations.append(config)
                            per_seed_records.extend(dict(**meta, metric=metric, seed=int(seed), percent_ce_better=value)
                                                    for seed, value in zip(seeds, per_seed))
    return pd.DataFrame(summaries), pd.concat(configurations, ignore_index=True), pd.concat(audits, ignore_index=True), pd.DataFrame(per_seed_records)


def style_axis(ax):
    ax.set(xlim=(-2, 102), ylim=(0, 100), xticks=np.arange(0, 101, 20), yticks=np.arange(0, 101, 20),
           xlabel="Inhibitory edges (%)", ylabel="Hyperparameter settings where CE > ER (%)")
    ax.axhline(50, color="0.6", ls=":", lw=.8, zorder=0)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)


def plot_line(ax, data, color, label, alternative=False):
    data = data.sort_values("negative_percent")
    value, low, high = (("mean_seed_win_percent", "seed_ci_low", "seed_ci_high") if alternative
                        else ("percent_ce_better", "ci_low", "ci_high"))
    ax.plot(data.negative_percent, data[value], "o-", color=color, label=label, lw=1.8, ms=5)
    ax.fill_between(data.negative_percent, data[low], data[high], color=color, alpha=.12, linewidth=0)


def save(fig, out, name):
    fig.savefig(out / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plots(summary, out):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    main = summary[(summary.experiment == "sign") & (summary.input_scale == "ALL")]
    for alternative, filename in [(False, "01_ce_winning_hyperparameter_fraction.png"),
                                  (True, "05_mean_per_realization_win_fraction.png")]:
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
        for ax, metric in zip(axes, METRICS):
            for family in FAMILIES[:2]:
                data = main[(main.metric == metric) & (main.family == family)]
                plot_line(ax, data, COLORS[family], LABELS[family], alternative)
            style_axis(ax)
            ax.set_title(metric)
            if alternative:
                ax.set_ylabel("Mean within-realization CE win fraction (%)")
        axes[0].legend(loc="best", frameon=False)
        fig.suptitle("CE versus matched ER at identical hyperparameters", y=.99)
        fig.text(.5, .015, ("Average of per-realization win fractions; a distinct estimand from comparing mean surfaces."
                          if alternative else "Compare mean performance over 200 paired realizations; 96 settings per fraction. Shading: pointwise 95% seed bootstrap."),
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .94))
        save(fig, out, filename)

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8), sharex=True, sharey=True)
    d = summary[(summary.experiment == "sign") & (summary.input_scale != "ALL")]
    for row, family in enumerate(FAMILIES[:2]):
        for ax, metric in zip(axes[row], METRICS):
            for level, color in INPUT_COLORS.items():
                g = d[(d.family == family) & (d.metric == metric) & (d.input_scale == str(level))]
                plot_line(ax, g, color, f"Input scale {level:g}")
            style_axis(ax)
            ax.set_title(f"{LABELS[family]}\n{metric}")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Where CE wins, conditional on input scale", y=.995)
    fig.text(.5, .01, "24 matched settings per input scale and fraction; mean surfaces over 200 paired realizations; pointwise 95% seed bootstrap.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, out, "02_ce_winning_fraction_by_input.png")

    norm = summary[(summary.experiment == "normalization") & (summary.input_scale == "ALL")]
    for metrics, filename, subtitle in [(["MC", "IPC"], "03_normalization_controls_mc_ipc.png", "MC and IPC"),
                                        (["KR"], "04_normalization_controls_kr_archival.png", "KR — archival protocol; keep separate from updated KR")]:
        fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 3.9 * len(metrics)), squeeze=False, sharex=True, sharey=True)
        for row, metric in enumerate(metrics):
            for ax, (normalization, title) in zip(axes[row], NORMS.items()):
                for family in FAMILIES[:2]:
                    g = norm[(norm.metric == metric) & (norm.family == family) & (norm.normalization == normalization)]
                    plot_line(ax, g, COLORS[family], LABELS[family])
                style_axis(ax)
                ax.set_title(f"{metric}: {title}")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"Normalization controls: {subtitle}", y=.995)
        fig.text(.5, .01, "96 matched settings; mean surfaces over 100 paired realizations; pointwise 95% seed bootstrap.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05 if len(metrics) == 1 else .03, 1, .94))
        save(fig, out, filename)


def report(summary, audit, out):
    main = summary[(summary.experiment == "sign") & (summary.input_scale == "ALL")]
    lines = ["# CE versus ER: fraction of winning hyperparameter configurations", "",
             "Primary statistic: for each inhibitory-edge fraction and each of the 96 identical hyperparameter settings, average CE−ER performance over paired seeds, then count the settings with a strictly positive mean difference. Higher is treated as better for MC, IPC and KR; greater KR denotes separation, not universally better computation.", "",
             "All plots use a 0–100% y scale. Counts weight the finite grid equally; they are not probabilities under a natural distribution of hyperparameters. Connecting lines guide the eye between sampled fractions, with no inferred crossover precision.", "",
             "## Main sign sweep", "",
             "| CE variant | Inhibitory edges | MC | IPC | KR |", "|---|---:|---:|---:|---:|"]
    for family in FAMILIES[:2]:
        for fraction, g in main[main.family == family].groupby("negative_percent"):
            cells = []
            for metric in METRICS:
                r = g[g.metric == metric].iloc[0]
                cells.append(f"{r.percent_ce_better:.1f}% ({r.n_ce_better}/{r.n_configurations})")
            lines.append(f"| {LABELS[family]} | {fraction:g}% | " + " | ".join(cells) + " |")
    lines.extend(["", "## Definitions, pairing and uncertainty", "",
                  "- Main sign sweep: 200 paired computational realizations; updated effective-rank KR. Normalization control experiment: 100 paired realizations, analyzed separately. Archival KR normalization results have different/unverified historical rank provenance and have their own figure.",
                  "- Match exactly on seed, inhibitory-edge fraction, normalization, spectral-radius target, leak rate, input scale and bias. Each seed carries its full grid throughout bootstrap resampling. These seeds are computational realizations, not independent biological animals.",
                  "- Grid: radius [0.6, 0.8, 0.95, 1.05], leak [0.6, 0.8, 1], input [0.1, 0.5, 1, 1.5], bias [0, 0.1]. The input-stratified plot uses 24 settings per curve point.",
                  "- Shared fractions are 0, 20, 40, 60, 80 and 100%. CE-only reference fractions near 24.25% are excluded because ER has no observation there; no interpolation or nearest-fraction matching. Exact unmatched counts appear in pairing_audit.csv.",
                  "- Strict CE > ER; exact ties count as non-wins. No significance filter, rounding, performance floor threshold, trimming or outlier clipping. A tiny positive difference counts equally to a large one, so win percentages should be read alongside absolute performance/effect sizes.",
                  "- Shading is a pointwise percentile interval from 2,000 paired whole-seed bootstrap resamples of the complete mean-surface calculation. Configurations are fixed, not independently resampled. The hard greater-than-zero decision makes estimates discrete and sensitive to near ties; bootstrap intervals are exploratory, not simultaneous discovery tests.",
                  "- Figure 05 instead computes a win fraction within each seed before averaging. This measures how often individual realizations win, and generally differs from the percentage of settings whose mean performance favors CE.",
                  "- Original-radius controls fix each family's reference multiplier across the sign sweep for a given target radius. This is not the newer shared-CE multiplier protocol. Sign-sweep ER is the archived binary-weight ER control, not a weight-distribution-matched control; topology and weights both differ.",
                  "- Inhibitory percentage refers to assigned edge signs, not an inferred biological inhibitory-neuron fraction. No new simulations were run and no existing result or figure was overwritten.", "",
                  "## Files", "",
                  "- 01_ce_winning_hyperparameter_fraction.png: primary requested plot.",
                  "- 02_ce_winning_fraction_by_input.png: condition on input scale.",
                  "- 03_normalization_controls_mc_ipc.png and 04_normalization_controls_kr_archival.png: separate normalization-control experiment.",
                  "- 05_mean_per_realization_win_fraction.png: alternative order of averaging.",
                  "- win_fraction_summary.csv, configuration_comparisons.csv, per_realization_win_fractions.csv and pairing_audit.csv contain numerical outputs. manifest.json records source hashes and analysis settings."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis/ei_hyperparameter_win_fractions_2026-11-18")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {out}")
    frames, sources = load_sources()
    summary, configurations, audit, seeds = calculate(frames, 2000, 20261118)
    out.mkdir(parents=True)
    for name, frame in [("win_fraction_summary", summary), ("configuration_comparisons", configurations),
                        ("pairing_audit", audit), ("per_realization_win_fractions", seeds)]:
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.15g")
    plots(summary, out)
    report(summary, audit, out)
    manifest = dict(sources=sources, n_bootstrap=2000, random_seed=20261118, metrics=METRICS,
                    win_rule="strict mean paired CE minus ER > 0; ties are not wins", seed_pairing="exact numeric seed",
                    bootstrap_unit="paired whole realization, jointly across its grid and sign fractions",
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
                    scripts=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in [Path(__file__).resolve(), ROOT / "util/ei_win_fraction.py"]],
                    outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(summary[(summary.experiment == "sign") & (summary.input_scale == "ALL")][["family", "metric", "negative_percent", "percent_ce_better", "n_ce_better", "n_configurations"]].to_string(index=False))


if __name__ == "__main__":
    main()
