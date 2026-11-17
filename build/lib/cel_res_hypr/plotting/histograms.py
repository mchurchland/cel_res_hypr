"""Histograms for reservoir research figures."""

from cel_res_hypr.analysis.aggregation import _unique_hparam_rows
from cel_res_hypr.paths import _safe_path
from cel_res_hypr.plotting.style import _short_legend_name
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_mc_vs_gr_all_arch(combined: pd.DataFrame, out_dir: str, alpha: float):
    if not {"MC", "GR", "mode"}.issubset(combined.columns):
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    # light grey background points for all runs
    plt.scatter(
        combined["GR"], combined["IPC"], s=8, c="#bbbbbb", alpha=alpha, label="all"
    )
    # emphasize CE-real and CE-shuffle if present
    for cname, color in [("local_sign+binary", "#1f77b4"), ("real", "#d62728")]:
        sub = combined[combined["mode"] == cname]
        if not sub.empty:
            sub_u = _unique_hparam_rows(sub)
            plt.scatter(
                sub_u["GR"],
                sub_u["MC"],
                s=36,
                alpha=0.5,
                label=_short_legend_name(cname),
                c=color,
            )
    plt.title("MC vs GR across all architectures")
    plt.xlabel("GR (effective rank of Δstate)")
    plt.ylabel("MC (linear memory capacity)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_fig = _safe_path(os.path.join(out_dir, "mc_vs_gr_all_arch.png"))
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[saved] {out_fig}")
