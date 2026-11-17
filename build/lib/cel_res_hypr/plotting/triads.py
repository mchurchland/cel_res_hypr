"""Triads for reservoir research figures."""

from cel_res_hypr.paths import _replace_path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_triad_sign_fraction_summary(
    summary_csv: str,
    out_dir: str,
    *,
    out_path: str = "",
    normalization: str = "spectral_radius",
    triad_scope: str = "closed",
    rho_target: float | None = None,
) -> str | None:
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Triad grouped summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    required = {
        "dataset",
        "normalization",
        "rho_target",
        "sign_frac",
        "triad_scope",
        "triad_avg_abs_w_mean_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Triad summary is missing required columns: {missing}")

    for col in (
        "rho_target",
        "sign_frac",
        "triad_avg_abs_w_mean_mean",
        "triad_avg_abs_w_mean_std",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        np.isfinite(df["sign_frac"]) & np.isfinite(df["triad_avg_abs_w_mean_mean"])
    ].copy()

    if normalization and str(normalization).lower() != "all":
        df = df[df["normalization"].astype(str) == str(normalization)].copy()
    if triad_scope and str(triad_scope).lower() != "all":
        df = df[df["triad_scope"].astype(str) == str(triad_scope)].copy()
    if rho_target is not None:
        df = df[np.isclose(df["rho_target"].astype(float), float(rho_target))].copy()

    if df.empty:
        print("[warn] triad sign-fraction plot: no rows after filters.")
        return None

    datasets = sorted(df["dataset"].astype(str).unique())
    scopes = sorted(df["triad_scope"].astype(str).unique())
    rhos = sorted(df["rho_target"].dropna().unique())
    if not rhos:
        rhos = [np.nan]

    panels = [(scope, rho) for scope in scopes for rho in rhos]
    n_cols = min(2, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig_w = 7.0 if n_cols == 2 else 5.2
    fig_h = 3.45 * n_rows + 1.0
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), dpi=300, squeeze=False
    )

    colors = dict(zip(datasets, plt.cm.tab10(np.linspace(0, 1, max(len(datasets), 1)))))
    handles = []
    labels = []

    for panel_idx, (scope, rho) in enumerate(panels):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        if np.isfinite(rho):
            panel = df[
                (df["triad_scope"].astype(str) == scope)
                & np.isclose(df["rho_target"].astype(float), float(rho))
            ].copy()
        else:
            panel = df[df["triad_scope"].astype(str) == scope].copy()
        if panel.empty:
            ax.set_axis_off()
            continue

        for dataset in datasets:
            sub = panel[panel["dataset"].astype(str) == dataset].sort_values(
                "sign_frac"
            )
            if sub.empty:
                continue
            xs = sub["sign_frac"].to_numpy(float)
            ys = sub["triad_avg_abs_w_mean_mean"].to_numpy(float)
            yerr = None
            if "triad_avg_abs_w_mean_std" in sub.columns:
                yerr_vals = sub["triad_avg_abs_w_mean_std"].to_numpy(float)
                if np.isfinite(yerr_vals).any():
                    yerr = np.nan_to_num(yerr_vals, nan=0.0)
            line = ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                capsize=2.4 if yerr is not None else 0.0,
                color=colors[dataset],
                label=dataset,
            )
            if panel_idx == 0:
                handles.append(line)
                labels.append(dataset)

        title_bits = []
        if len(scopes) > 1:
            title_bits.append(scope)
        if np.isfinite(rho):
            title_bits.append(rf"$\rho={rho:g}$")
        panel_letter = chr(ord("A") + panel_idx)
        panel_title = " | ".join(title_bits) if title_bits else "Triad $|w|$"
        ax.set_title(f"{panel_letter}  {panel_title}", loc="left", fontweight="bold")
        ax.set_xlabel("E/I Edge Balance")
        ax.set_ylabel(r"Mean triad-average $|w|$")
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.24)
        ax.margins(y=0.08)

    for panel_idx in range(len(panels), n_rows * n_cols):
        axes.flat[panel_idx].set_axis_off()

    if handles and len(datasets) > 1:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(handles), 4),
            frameon=False,
        )

    fig.suptitle(r"Triad average $|w|$ by E/I Edge Balance", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], h_pad=2.0, w_pad=1.6)

    os.makedirs(out_dir, exist_ok=True)
    if not out_path:
        out_path = os.path.join(out_dir, "triad_avg_abs_w_by_sign_fraction.png")
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path
