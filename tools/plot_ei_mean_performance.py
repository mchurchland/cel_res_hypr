#!/usr/bin/env python3
"""Add mean CE/ER performance curves to the archived E/I win-fraction analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.ei_win_fraction import PARAMETERS, bootstrap_counts

METRICS = ["MC", "IPC", "KR"]
FAMILIES = {
    "cel_matched": ("CE: unknown signs completed", "#d55e00", "o"),
    "cel_removed": ("CE: unknown-sign edges removed", "#0072b2", "s"),
    "matched_er": ("ER", "#009e73", "^"),
}
FRACTIONS = [0., .2, .4, .6, .8, 1.]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw(ax, data, metric):
    for family, (label, color, marker) in FAMILIES.items():
        g = data[(data.family == family) & (data.metric == metric)].sort_values("negative_percent")
        ax.plot(g.negative_percent, g["mean"], color=color, marker=marker, label=label, lw=1.8, ms=5)
        ax.fill_between(g.negative_percent, g.ci_low, g.ci_high, color=color, alpha=.13, linewidth=0)
    maximum = data.loc[data.metric == metric, "ci_high"].max()
    ax.set(xlabel="Inhibitory edges (%)", ylabel=f"Mean {metric}", title=metric,
           xlim=(-2, 102), xticks=np.arange(0, 101, 20), ylim=(0, maximum * 1.1))
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis/ei_hyperparameter_win_fractions_2026-11-18")
    out = parser.parse_args().out_dir.resolve()
    old_manifest = json.loads((out / "manifest.json").read_text())
    output_names = ["06_mean_performance_sweep.png", *[f"mean_performance_{m}.png" for m in METRICS],
                    "mean_performance_summary.csv", "mean_performance_manifest.json"]
    if any((out / name).exists() for name in output_names):
        raise FileExistsError("Refusing to overwrite existing mean-performance outputs")
    existing_hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    source_records = [s for s in old_manifest["sources"] if s["experiment"] == "sign"]
    frames = {}
    for source in source_records:
        path = ROOT / source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError(f"Source changed since win-fraction analysis: {path}")
        d = pd.read_csv(path)
        d["negative_fraction"] = d["mode"].str.extract(r"(?:sign_test_og_cel|sign_test_er)([0-9.]+)$")[0].astype(float)
        frames[source["family"]] = d[d.negative_fraction.isin(FRACTIONS)].copy()
    seeds = np.sort(frames["matched_er"].seed.unique())
    counts = bootstrap_counts(len(seeds), 2000, np.random.default_rng(20261118))
    records = []
    for family, d in frames.items():
        keys = ["negative_fraction", "seed", *PARAMETERS]
        assert not d.duplicated(keys).any()
        assert set(d.normalization) == {"spectral_radius"}
        assert np.isfinite(d[METRICS].to_numpy()).all()
        group_sizes = d.groupby(["negative_fraction", "seed"]).size()
        assert len(group_sizes) == len(seeds) * len(FRACTIONS)
        assert (group_sizes == 96).all()
        for fraction, group in d.groupby("negative_fraction"):
            per_seed = group.groupby("seed")[METRICS].mean().reindex(seeds)
            assert np.isfinite(per_seed.to_numpy()).all()
            boot = counts @ per_seed.to_numpy() / len(seeds)
            ci = np.quantile(boot, [.025, .975], axis=0)
            for j, metric in enumerate(METRICS):
                mean = per_seed[metric].mean()
                # Balanced grid: independent direct average must give the same answer.
                np.testing.assert_allclose(mean, group[metric].mean(), rtol=1e-12, atol=1e-12)
                records.append(dict(family=family, normalization="spectral_radius", negative_percent=fraction * 100,
                                    metric=metric, mean=mean, ci_low=ci[0, j], ci_high=ci[1, j],
                                    n_seeds=len(seeds), n_configurations=96))
    summary = pd.DataFrame(records)
    assert len(summary) == 54
    summary.to_csv(out / "mean_performance_summary.csv", index=False, float_format="%.15g")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, metric in zip(axes, METRICS):
        draw(ax, summary, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015), ncol=3, frameon=False)
    fig.suptitle("Mean performance across the inhibitory-edge sweep", y=.99)
    fig.text(.5, .105, "96 hyperparameter settings; 200 realizations; own-radius scaling. Shading: pointwise 95% seed-bootstrap intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(out / "06_mean_performance_sweep.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        draw(ax, summary, metric)
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.text(.5, .02, "96 settings × 200 realizations; own-radius scaling; 95% seed-bootstrap intervals", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .055, 1, 1))
        fig.savefig(out / f"mean_performance_{metric}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    assert all(sha(out / name) == digest for name, digest in existing_hashes.items())
    manifest = dict(sources=source_records, script=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=sha(Path(__file__))),
                    estimand="Mean within each seed over all 96 configurations, then mean across 200 seeds",
                    n_bootstrap=2000, random_seed=20261118, bootstrap_unit="whole seed; same draws across curves and fractions",
                    shared_fractions_percent=[100 * f for f in FRACTIONS], ce_only_reference_points_excluded=True,
                    metric_directions="MC/IPC capacity; KR separation, not generic task performance",
                    normalization="own spectral radius", er_control="archived binary-weight ER",
                    validation=dict(n_summary_rows=54, all_means_match_direct_balanced_raw_averages=True,
                                    all_existing_analysis_files_unchanged=True, n_seeds=200, configurations_per_seed_and_fraction=96),
                    outputs={name: sha(out / name) for name in output_names if name != "mean_performance_manifest.json"})
    (out / "mean_performance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "06_mean_performance_sweep.png")


if __name__ == "__main__":
    main()
