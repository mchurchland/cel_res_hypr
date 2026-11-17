#!/usr/bin/env python3
"""Make compact PNG figures and a portable HTML copy of the Jordi summary.

Reads existing summary tables only; no simulations or inference are rerun.
"""
from pathlib import Path
import base64
import hashlib
import json
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/robustness_reevaluation_2026-11-16"
OUT = ROOT / "analysis/exploratory_followup_2026-11-17"
FIGURES = OUT / "jordi_summary_figures"
METRICS = ("MC", "IPC")
EI = np.arange(0, 101, 20)
PARAMETERS = ("rho_target", "leak", "input_scale", "neuron_bias")
PARAMETER_COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_axis(ax, metric):
    ax.set_title(metric, weight="semibold", pad=10)
    ax.set_xlabel("Inhibitory edges (%)")
    ax.set_xticks(EI)
    ax.grid(axis="y", color="#e7ebef", linewidth=.7)
    ax.set_axisbelow(True)


def crossover(points):
    data = points[
        (points.experiment == "sign")
        & (points.normalization == "spectral_radius")
        & points.family.isin(["cel_matched", "matched_er"])
        & points.metric.isin(METRICS)
        & points.negative_percent.isin(EI)
    ]
    counts = data.groupby(["family", "metric", "negative_percent"]).size()
    assert len(counts) == 24 and (counts == 96).all()
    means = data.groupby(["family", "metric", "negative_percent"]).performance.mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric in zip(axes, METRICS):
        for family, label, color, marker in [
            ("cel_matched", "C. elegans (sign-matched)", "#332288", "o"),
            ("matched_er", "Matched ER", "#0077bb", "s"),
        ]:
            series = means.loc[(family, metric)].reindex(EI)
            assert series.notna().all()
            ax.plot(EI, series, marker=marker, color=color, lw=1.8, ms=5, label=label)
        setup_axis(ax, metric)
        ax.set_ylabel(f"Mean {metric}")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc="lower center")
    save(fig, "01_performance_crossover.png")


def input_drive(conditional):
    data = conditional[
        (conditional.experiment == "sign")
        & (conditional.normalization == "spectral_radius")
        & (conditional.parameter == "input_scale")
    ].copy()
    data["level_numeric"] = pd.to_numeric(data.level)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, metric in zip(axes, METRICS):
        for level, color, marker in [(.1, "#332288", "o"), (1.5, "#ee7733", "s")]:
            g = data[(data.metric == metric) & np.isclose(data.level_numeric, level)]
            g = g.sort_values("negative_percent")
            assert len(g) == 6 and (g.n_seeds == 200).all()
            assert (g.ci_low <= g.estimate).all() and (g.estimate <= g.ci_high).all()
            ax.errorbar(
                g.negative_percent, g.estimate,
                yerr=[g.estimate - g.ci_low, g.ci_high - g.estimate],
                color=color, marker=marker, lw=1.6, ms=5, capsize=3,
                label=f"Input scale {level:g}",
            )
        setup_axis(ax, metric)
        ax.set_ylabel(f"CE − ER mean {metric}")
        ax.axhline(0, color="#4b5563", lw=1, ls="--")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "02_input_drive.png")


def variance_shares(summary):
    data = summary[(summary.experiment == "sign") & (summary.normalization == "spectral_radius")]
    families = ("cel_matched", "cel_removed", "matched_er")
    labels = ("CE: signs matched", "CE: signs removed", "Matched ER")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    fig.subplots_adjust(left=.16, right=.99, top=.85, bottom=.30, wspace=.16)
    for ax, metric in zip(axes, (*METRICS, "KR")):
        values = data[data.metric == metric].pivot(index="family", columns="parameter", values="allocated_pct")
        values = values.reindex(index=families, columns=PARAMETERS)
        assert values.notna().all().all()
        np.testing.assert_allclose(values.sum(axis=1), 100, atol=1e-8)
        left = np.zeros(3)
        for parameter, color in zip(PARAMETERS, PARAMETER_COLORS):
            width = values[parameter].to_numpy()
            ax.barh(np.arange(3), width, left=left, height=.58, color=color, edgecolor="white", linewidth=.6)
            for row, (start, value) in enumerate(zip(left, width)):
                if value >= 8:
                    ax.text(start + value / 2, row, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)
            left += width
        ax.set(title=metric, xlim=(0, 100), xticks=[0, 50, 100], xlabel="Allocated variance (%)")
        ax.set_yticks(range(3), labels)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles = [Patch(facecolor=color, label=label) for color, label in zip(
        PARAMETER_COLORS, ("Spectral-radius target", "Leak rate", "Input scale", "Bias range")
    )]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .02))
    save(fig, "03_parameter_contributions.png")


def portable_html():
    source = OUT / "SUMMARY_FOR_JORDI.md"
    # The standalone copy contains its three figures; supporting analyses stay separate.
    text = source.read_text().split("## Supporting analyses")[0]
    body = markdown.markdown(text, extensions=["tables"])

    def embed(match):
        path = source.parent / match.group(1)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{encoded}"'

    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    assert body.count("data:image/png;base64,") == 3
    css = """
body {max-width: 960px; margin: 40px auto; padding: 0 24px 48px;
      font: 17px/1.6 system-ui, sans-serif; color: #243248; background: white;}
h1 {font-size: 28px; line-height: 1.25; margin-bottom: 12px;}
h2 {font-size: 22px; margin: 36px 0 12px; color: #172b44;}
p {margin: 14px 0;} img {display: block; width: 100%; height: auto; margin: 24px 0 8px;}
p:has(> em:first-child) {font-size: 14px; color: #526176; line-height: 1.5;}
a {color: #225ea8;} table {border-collapse: collapse; width: 100%;}
th, td {padding: 8px; border-bottom: 1px solid #dce3eb; text-align: left;}
@media print {body {max-width: none; margin: 0;} img {break-inside: avoid;}}
"""
    html = f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Preliminary reanalysis — Miles Churchland</title><style>{css}</style></head><body>{body}</body></html>\n'
    (OUT / "SUMMARY_FOR_JORDI.html").write_text(html)


def main():
    FIGURES.mkdir(exist_ok=True)
    sources = {
        "points": BASE / "figures/ei_hyperparameter_dots/plotted_points.csv.gz",
        "conditional": OUT / "tables/conditional_ce_minus_er.csv",
        "shares": BASE / "hyperparameter_variability/tables/across_ei_summary.csv",
    }
    before = {key: digest(path) for key, path in sources.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    crossover(pd.read_csv(sources["points"]))
    input_drive(pd.read_csv(sources["conditional"]))
    variance_shares(pd.read_csv(sources["shares"]))
    portable_html()
    assert before == {key: digest(path) for key, path in sources.items()}
    manifest = {
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": before[key]} for key, path in sources.items()],
        "script_sha256": digest(Path(__file__)),
        "settings": {"shared_ei_percent": EI.tolist(), "normalization": "spectral_radius", "sign_sweep_seeds": 200,
                     "input_figure_levels": [.1, 1.5], "inference": "reuse existing paired seed-bootstrap intervals",
                     "variance_allocation": "equal split of interactions; original family-specific E/I summaries"},
        "validation": "Complete grid counts, six shared E/I levels, interval ordering, 100% variance sums and unchanged source hashes checked.",
    }
    (FIGURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved three compact PNG figures and SUMMARY_FOR_JORDI.html; source tables unchanged.")


if __name__ == "__main__":
    main()
