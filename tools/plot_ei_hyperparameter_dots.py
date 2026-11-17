#!/usr/bin/env python3
"""Plot stored performance against negative-edge percentage with four dot encodings.

Each point is one hyperparameter combination averaged across all stored seeds.
Only horizontal display coordinates are jittered. Source results remain read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/robustness-xdg-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.robustness import PARAMETERS

METRICS = ("MC", "IPC", "KR")
FAMILIES = {
    "cel_matched": "C. elegans · unknown signs matched",
    "cel_removed": "C. elegans · unknown signs removed",
    "matched_er": "Matched ER",
}
COLORS = {0.6:"#332288", 0.8:"#0077BB", 0.95:"#009988", 1.05:"#EE7733"}
MARKERS = {0.6:"o", 0.8:"s", 1.0:"^"}
SIZES = {0.1:10, 0.5:19, 1.0:32, 1.5:49}
BIAS_WIDTH = {0.0:0.0, 0.1:0.65}


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_points(jitter, seed):
    records, sources = [], []
    required = ["mode", "normalization", "seed", *PARAMETERS, *METRICS]
    for experiment in ("sign", "normalization"):
        for family in FAMILIES:
            relative = (f"final_results/sign_frac/{family}/combined.ALL.GRKR_erank.rank_updated.csv"
                        if experiment == "sign" else f"final_results/Rho_stuff/sign_norm/{family}/combined.ALL.csv")
            path = ROOT/relative
            frame = pd.read_csv(path, usecols=required)
            if frame.duplicated(["mode", "normalization", "seed", *PARAMETERS]).any():
                raise ValueError(f"Duplicate configuration/seed: {path}")
            fraction = frame["mode"].str.extract(r"^sign_test_(?:og_cel|er)([0-9.]+)(?:__norm_.*)?$", expand=False)
            if fraction.isna().any():
                raise ValueError(f"Unrecognized sign fraction: {path}")
            frame["negative_percent"] = pd.to_numeric(fraction)*100
            group_keys = ["normalization", "negative_percent", *PARAMETERS]
            for metric in METRICS:
                selected = frame[np.isfinite(frame[metric])]
                agg = selected.groupby(group_keys, as_index=False).agg(
                    performance=(metric,"mean"), seed_sd=(metric,"std"), n_seeds=("seed","nunique"))
                agg["metric"], agg["family"], agg["experiment"] = metric, family, experiment
                records.append(agg)
            sources.append(dict(path=relative,sha256=digest(path),rows=len(frame),
                                seeds_per_condition=int(frame.groupby(["mode","normalization"]).seed.nunique().min()),
                                rank_vintage="updated_erank" if experiment == "sign" else "archival_control"))
    points = pd.concat(records, ignore_index=True)
    # Match the same configuration's horizontal displacement across metrics,
    # experiments and normalization modes. Spaced slots avoid accidental clumps.
    jitter_keys = ["family", "negative_percent", *PARAMETERS]
    positions = points[jitter_keys].drop_duplicates().sort_values(jitter_keys).reset_index(drop=True)
    positions["jitter_percentage_points"] = 0.0
    for (family, percent), group in positions.groupby(["family", "negative_percent"], sort=True):
        key = f"{seed}:{family}:{percent:.12g}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        slots = np.linspace(-jitter,jitter,len(group))
        positions.loc[group.index,"jitter_percentage_points"] = rng.permutation(slots)
    points = points.merge(positions, on=jitter_keys, validate="many_to_one")
    points["display_negative_percent"] = points.negative_percent+points.jitter_percentage_points
    points["dot_color"] = points.rho_target.map(COLORS)
    points["dot_shape"] = points.leak.map(MARKERS)
    points["dot_area_pt2"] = points.input_scale.map(SIZES)
    points["outline_width_pt"] = points.neuron_bias.map(BIAS_WIDTH)
    if points[["dot_color","dot_shape","dot_area_pt2","outline_width_pt"]].isna().any().any():
        raise ValueError("Unexpected hyperparameter levels; define explicit dot encodings")
    return points, sources


def plot_panel(ax, data, metric):
    g = data[data.metric == metric]
    # Draw large points first so small neighboring points remain visible.
    g = g.sort_values("dot_area_pt2", ascending=False, kind="stable")
    for marker in MARKERS.values():
        h = g[g.dot_shape == marker]
        ax.scatter(h.display_negative_percent, h.performance,
                   s=h.dot_area_pt2, c=h.dot_color, marker=marker,
                   edgecolors="black", linewidths=h.outline_width_pt,
                   alpha=.82, rasterized=False)
    fractions = sorted(g.negative_percent.unique())
    ticks = sorted(set([0,20,40,60,80,100]+[float(v) for v in fractions if not np.isclose(v%20,0)]))
    # Put the closely spaced empirical reference on a second label row.
    ax.set_xticks(ticks, [f"\n{v:.1f}" if not np.isclose(v%20,0) else f"{v:.0f}" for v in ticks])
    ax.tick_params(axis="x", labelsize=8)
    ax.set(xlim=(-2.3,102.3),xlabel="Inhibitory edges (%)",ylabel=f"Mean {metric} across seeds")
    ax.grid(axis="y",color="#ededed",lw=.6)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.margins(y=.08)


def add_legends(fig):
    # Four separate keys make each independent visual property readable.
    legends = [
        ("Color · spectral-radius target",[Line2D([],[],ls="",marker="o",ms=6,color=c,label=f"{v:g}") for v,c in COLORS.items()],.125,4),
        ("Shape · leak rate",[Line2D([],[],ls="",marker=m,ms=6,color="#666666",label=f"{v:g}") for v,m in MARKERS.items()],.39,3),
        ("Area · input scale",[Line2D([],[],ls="",marker="o",ms=np.sqrt(s),color="#777777",label=f"{v:g}") for v,s in SIZES.items()],.64,4),
        ("Outline · bias range",[
            Line2D([],[],ls="",marker="o",ms=7,markerfacecolor="#aaaaaa",markeredgewidth=0,label="0: none"),
            Line2D([],[],ls="",marker="o",ms=7,markerfacecolor="#aaaaaa",markeredgecolor="black",markeredgewidth=.65,label="0.1: black")],.9,2),
    ]
    for title,handles,x,ncol in legends:
        fig.legend(handles=handles,title=title,loc="upper center",bbox_to_anchor=(x,-.035),
                   ncol=ncol,frameon=False,fontsize=8,title_fontsize=9,columnspacing=.9,handletextpad=.35)


def save(fig, output, stem):
    add_legends(fig)
    fig.savefig(output/f"{stem}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def make_figures(points, output, jitter):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42})
    sign = points[points.experiment == "sign"]
    fig,axes = plt.subplots(3,3,figsize=(14.5,10),layout="constrained")
    for i,(family,label) in enumerate(FAMILIES.items()):
        for j,metric in enumerate(METRICS):
            plot_panel(axes[i,j],sign[sign.family == family],metric)
            axes[i,j].set_title(f"{label} · {metric}",fontsize=10)
    # Compare families directly within each metric, retaining its native units.
    for j in range(len(METRICS)):
        low=min(ax.get_ylim()[0] for ax in axes[:,j])
        high=max(ax.get_ylim()[1] for ax in axes[:,j])
        for ax in axes[:,j]:
            ax.set_ylim(low,high)
    fig.suptitle(f"Performance across E/I balance · each dot = one of 96 hyperparameter combinations\n"
                 f"Mean of 200 seeds; horizontal jitter ±{jitter:g} percentage points; own-radius normalization",fontsize=12)
    save(fig,output,"all_families_performance_dots")
    for family,label in FAMILIES.items():
        fig,axes = plt.subplots(1,3,figsize=(14.5,4.2),layout="constrained")
        for ax,metric in zip(axes,METRICS):
            plot_panel(ax,sign[sign.family == family],metric)
            ax.set_title(metric,fontsize=11)
        fig.suptitle(f"{label} · 96 combinations per E/I level\nMean of 200 seeds; horizontal jitter ±{jitter:g} percentage points",fontsize=12)
        save(fig,output,f"{family}_performance_dots")
        fig,axes = plt.subplots(2,3,figsize=(14.5,7.4),layout="constrained")
        for i,norm in enumerate(("spectral_radius","original_radius")):
            data = points[(points.experiment == "normalization")&(points.family == family)&(points.normalization == norm)]
            for j,metric in enumerate(METRICS):
                plot_panel(axes[i,j],data,metric)
                axes[i,j].set_title(f"{'Own radius' if i == 0 else 'Fixed reference'} · {metric}",fontsize=11)
        # Share y scales within a metric so the two normalization modes compare directly.
        for j in range(3):
            low=min(axes[0,j].get_ylim()[0],axes[1,j].get_ylim()[0])
            high=max(axes[0,j].get_ylim()[1],axes[1,j].get_ylim()[1])
            axes[0,j].set_ylim(low,high); axes[1,j].set_ylim(low,high)
        fig.suptitle(f"{label} · normalization controls\nMean of 100 seeds; archival KR kept separate from updated sign sweeps",fontsize=12)
        save(fig,output,f"{family}_normalization_performance_dots")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/robustness_reevaluation_2026-11-16/figures/ei_hyperparameter_dots")
    parser.add_argument("--jitter",type=float,default=1.2,help="Maximum horizontal displacement in percentage points")
    parser.add_argument("--seed",type=int,default=20261116)
    parser.add_argument("--refresh",action="store_true",help="Refresh only a directory previously generated by this script")
    args=parser.parse_args()
    if not 0 <= args.jitter <= 1.5:
        raise ValueError("Use small horizontal jitter between 0 and 1.5 percentage points")
    output=args.out_dir.resolve()
    if output.is_relative_to(ROOT/"final_results"):
        raise ValueError("Output must stay outside final_results")
    if output.exists():
        prior_manifest=output/"manifest.json"
        if not args.refresh or not prior_manifest.exists() or "encodings" not in json.loads(prior_manifest.read_text()):
            raise FileExistsError("Choose a new directory, or use --refresh for this script's existing output")
    points,sources=load_points(args.jitter,args.seed)
    keys=["experiment","family","normalization","metric","negative_percent"]
    counts=points.groupby(keys).size()
    assert (counts == 96).all(),"Expected the complete 96-combination grid at every sign level"
    assert np.isfinite(points.performance).all()
    assert points.jitter_percentage_points.abs().max() <= args.jitter+1e-10
    assert not points.duplicated(keys+list(PARAMETERS)).any()
    expected={"sign":200,"normalization":100}
    for experiment,count in expected.items():
        assert (points.loc[points.experiment == experiment,"n_seeds"] == count).all()
    output.mkdir(parents=True,exist_ok=True)
    points.to_csv(output/"plotted_points.csv.gz",index=False,float_format="%.17g",compression={"method":"gzip","compresslevel":1})
    make_figures(points,output,args.jitter)
    for source in sources:
        assert digest(ROOT/source["path"]) == source["sha256"],"Input file changed"
    settings=dict(sources=sources,script_sha256=digest(Path(__file__)),seed=args.seed,
                  jitter_percentage_points=args.jitter,jitter_axis="horizontal only; true E/I values preserved",
                  aggregation="Arithmetic mean within each sign fraction, normalization and full hyperparameter combination across seeds",
                  encodings=dict(color={str(k):v for k,v in COLORS.items()},shape={str(k):v for k,v in MARKERS.items()},
                                 area_pt2={str(k):v for k,v in SIZES.items()},outline_width_pt={str(k):v for k,v in BIAS_WIDTH.items()}),
                  x_definition="100 × requested fraction of negative edges, not percentage of inhibitory neurons",
                  y_definition="Original performance units; separate axes for MC, IPC and KR; no vertical jitter",
                  n_plotted_records=len(points),n_figures=7,figure_format="PNG only",
                  validation="PASS: 96 configurations per level/metric; 200 or 100 seeds per point; unique identities; finite means; jitter bounded; source hashes unchanged",
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (output/"manifest.json").write_text(json.dumps(settings,indent=2)+"\n")
    (output/"README.md").write_text(
        "# Performance versus E/I balance, with hyperparameter dots\n\n"
        "Each dot is a single hyperparameter combination averaged over seeds, with no CV axis or connecting lines.\n\n"
        "- Color: spectral-radius target (0.6, 0.8, 0.95, 1.05).\n"
        "- Shape: leak rate (circle 0.6, square 0.8, triangle 1).\n"
        "- Area: input scale (0.1, 0.5, 1, 1.5).\n"
        "- Outline: neuron-bias range (none for 0; black outline for 0.1).\n\n"
        f"Horizontal jitter is deterministic and bounded by ±{args.jitter:g} percentage points, matched across metrics and normalization modes. Performance is never jittered. "
        "The x-axis is percentage of negative **edges**, rather than inhibitory neurons. Points just outside 0–100% are display jitter only.\n\n"
        "Updated sign-sweep figures average 200 seeds per combination. Separate normalization-control figures average 100 seeds per combination and retain their archival KR protocol; compare normalization rows within those figures. "
        "Dot colors encode hyperparameters, with architecture families separated into panels. The seed means do not display seed uncertainty.\n\n"
        "Open `cel_matched_performance_dots.png` for the primary connectome view, `all_families_performance_dots.png` for all families, or the normalization figures for own-radius versus fixed-reference scaling. "
        "All figures are 300 dpi PNGs. `plotted_points.csv.gz` preserves exact unjittered coordinates, displayed coordinates, encodings and seed counts.\n\n"
        "Reproduce with `python tools/plot_ei_hyperparameter_dots.py --out-dir analysis/new_dot_figures`. Exact sources, hashes and settings are in `manifest.json`.\n")
    print(f"Saved 7 PNG figures and {len(points)} plotted records to {output}")


if __name__ == "__main__":
    main()
