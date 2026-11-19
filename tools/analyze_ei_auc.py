#!/usr/bin/env python3
"""Add integrated mean-performance comparisons to the E/I sweep analysis."""
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
from util.ei_win_fraction import PARAMETERS, bootstrap_counts

OUT = ROOT / "analysis/ei_hyperparameter_win_fractions_2026-11-18"
METRICS = ["MC", "IPC", "KR"]
LABELS = {"cel_matched": "CE: unknown signs completed", "cel_removed": "CE: unknown-sign edges removed"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ["auc_summary.csv", "auc_per_realization.csv", "AUC_REPORT.md", "auc_manifest.json", "07_integrated_performance_advantage.png"]
    if any((OUT / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing AUC outputs")
    old_hashes = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    manifest = json.loads((OUT / "mean_performance_manifest.json").read_text())
    x = np.array([0., .2, .4, .6, .8, 1.])
    arrays = {}
    seed_ref = None
    for source in manifest["sources"]:
        path = ROOT / source["path"]
        assert sha(path) == source["sha256"]
        d = pd.read_csv(path)
        d["fraction"] = d["mode"].str.extract(r"(?:sign_test_og_cel|sign_test_er)([0-9.]+)$")[0].astype(float)
        d = d[d.fraction.isin(x)]
        assert not d.duplicated(["fraction", "seed", *PARAMETERS]).any()
        assert (d.groupby(["fraction", "seed"]).size() == 96).all()
        seeds = np.sort(d.seed.unique())
        if seed_ref is None:
            seed_ref = seeds
        np.testing.assert_array_equal(seed_ref, seeds)
        means = d.groupby(["seed", "fraction"])[METRICS].mean()
        for metric in METRICS:
            a = means[metric].unstack("fraction").reindex(index=seeds, columns=x).to_numpy()
            assert np.isfinite(a).all()
            arrays[source["family"], metric] = a
    n = len(seed_ref)
    counts = bootstrap_counts(n, 5000, np.random.default_rng(20261118))
    rows, per_seed = [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
    for row_index, (family, label) in enumerate(LABELS.items()):
        for j, metric in enumerate(METRICS):
            ce, er = arrays[family, metric], arrays["matched_er", metric]
            ce_auc = curve_areas(ce, x)[0]
            er_auc = curve_areas(er, x)[0]
            difference = ce - er
            curve = difference.mean(axis=0)
            signed, positive, negative = curve_areas(curve, x)
            boot_curves = counts @ difference / n
            boot_signed, boot_positive, boot_negative = curve_areas(boot_curves, x)
            lo, hi = np.quantile(boot_signed, [.025, .975])
            np.testing.assert_allclose(signed, ce_auc.mean() - er_auc.mean(), atol=1e-12)
            record = dict(family=family, metric=metric, ce_auc=ce_auc.mean(), er_auc=er_auc.mean(),
                          ce_minus_er_auc=float(signed), difference_ci_low=lo, difference_ci_high=hi,
                          relative_auc_difference_percent=100 * signed / er_auc.mean(),
                          ce_advantage_area=float(positive), er_advantage_area=float(negative),
                          n_seeds=n, n_configurations=96)
            for name, boot in [("ce_auc", counts @ ce_auc / n), ("er_auc", counts @ er_auc / n),
                               ("ce_advantage_area", boot_positive), ("er_advantage_area", boot_negative)]:
                low, high = np.quantile(boot, [.025, .975])
                record[name + "_ci_low"], record[name + "_ci_high"] = low, high
            rows.append(record)
            per_seed.extend(dict(family=family, metric=metric, seed=int(seed), ce_auc=c, er_auc=e, ce_minus_er_auc=c-e)
                            for seed, c, e in zip(seed_ref, ce_auc, er_auc))
            ax = axes[row_index, j]
            ax.plot(100*x, curve, color="0.2", marker="o", ms=4, lw=1.4)
            ax.fill_between(100*x, curve, 0, where=curve >= 0, interpolate=True, color="#009e73", alpha=.4, label="CE advantage area")
            ax.fill_between(100*x, curve, 0, where=curve <= 0, interpolate=True, color="#d55e00", alpha=.3, label="ER advantage area")
            ax.axhline(0, color="0.3", lw=.8)
            ax.set(title=f"{label}\n{metric}", xlabel="Inhibitory edges (%)", ylabel=f"Mean {metric}: CE − ER",
                   xlim=(0, 100), xticks=100*x)
            ax.text(.04, .06, f"Net area: {signed:+.2f}\n95% CI [{lo:+.2f}, {hi:+.2f}]", transform=ax.transAxes, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "auc_summary.csv", index=False, float_format="%.15g")
    pd.DataFrame(per_seed).to_csv(OUT / "auc_per_realization.csv", index=False, float_format="%.15g")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=2)
    fig.suptitle("Area between CE and ER mean-performance curves", y=.995)
    fig.text(.5, .06, "Areas integrate inhibitory fraction 0–1; straight lines interpolate the six sampled fractions. Own-radius scaling; 96 settings; 200 realizations.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(OUT / "07_integrated_performance_advantage.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    lines = ["# Integrated CE–ER performance across the E/I sweep", "",
             "AUC integrates each mean-performance curve over inhibitory fraction f from 0 to 1. Thus AUC is also the average performance under a uniform distribution of f, with linear interpolation between the six sampled fractions. Integrating a percentage axis 0–100 would multiply every area by 100 without changing comparisons.", "",
             "Mean performance averages all 96 hyperparameter settings within each seed. The paired bootstrap resamples whole seeds (5,000 draws), retaining all settings and fractions. CE-only reference fractions near 24.25% remain excluded. Existing analysis files are unchanged.", "",
             "| CE variant | Metric | CE AUC | ER AUC | CE−ER AUC [95% interval] | Relative difference |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {LABELS[r['family']]} | {r['metric']} | {r['ce_auc']:.3f} | {r['er_auc']:.3f} | {r['ce_minus_er_auc']:+.3f} [{r['difference_ci_low']:+.3f}, {r['difference_ci_high']:+.3f}] | {r['relative_auc_difference_percent']:+.1f}% |")
    lines += ["", "Positive and negative areas split the area between the mean curves at interpolated crossings. Their difference equals the signed AUC difference. They describe the mean curve, not the mean of per-seed positive/negative areas.", "",
              "This is an overall performance summary, not a standalone robustness measure. It weights the interval 0–100% uniformly, which is an explicit modeling choice and not an empirical distribution of biological conditions. Trapezoidal integration gives each endpoint half the weight of an interior sampled fraction. The unmeasured intervals, including 40–60%, are approximated linearly; the true continuous-sweep AUC is not directly observed.", "",
              "The comparison uses archived binary-weight ER, own-radius normalization, and the specified 96-point hyperparameter grid. Do not generalize it to weight-matched controls or different normalization protocols. Compare AUCs within each metric only; MC, IPC and KR have different scales. Larger KR measures separation rather than generic computational superiority.", "",
              "These AUCs concern the mean-performance curves, not the earlier percentage-of-winning-configurations curves."]
    (OUT / "AUC_REPORT.md").write_text("\n".join(lines) + "\n")
    assert all(sha(OUT / name) == digest for name, digest in old_hashes.items())
    provenance = dict(sources=manifest["sources"], axis="inhibitory fraction 0 to 1", integration="piecewise linear / trapezoidal",
                      n_bootstrap=5000, bootstrap_seed=20261118, existing_files_unchanged=True,
                      scripts=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in [Path(__file__).resolve(), ROOT / "util/ei_auc.py"]],
                      outputs={name: sha(OUT / name) for name in names if name != "auc_manifest.json"})
    (OUT / "auc_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary[["family", "metric", "ce_auc", "er_auc", "ce_minus_er_auc", "difference_ci_low", "difference_ci_high", "relative_auc_difference_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
