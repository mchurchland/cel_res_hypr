"""Rho for reservoir research figures."""

from cel_res_hypr.analysis.aggregation import (
    _aggregate_over_hparams,
    _assign_group_ids,
    _dispersion_cv as _dispersion,
)
from cel_res_hypr.plotting.rendering import _style_3d_axis
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def plot_rho_cv_other_perf(
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    model: str = "",
    drop_kr_gr: bool = False,
):
    """
    3D line plot at fixed spectral radius, one figure per mode:
      x = spectral radius (rho_target)
      y = CV across non-rho hyperparameters (leak, input_scale, neuron_bias) within each fixed rho
      z = mean performance at that fixed rho

    CV is computed per (mode, src, group_id, rho_target, metric) across leak/input_scale/neuron_bias,
    then averaged across groups for each (mode, rho_target, metric).
    """
    os.makedirs(out_dir, exist_ok=True)

    metric_cols = [m for m in ("MC", "IPC", "KR", "GR") if m in combined.columns]
    if drop_kr_gr:
        metric_cols = [m for m in metric_cols if m not in ("KR", "GR")]
    required_cols = [
        "mode",
        "src",
        "shuffle_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
    ] + metric_cols
    if not metric_cols:
        print("[warn] plot_rho_cv_other_perf: no selected metric columns found.")
        return
    if any(c not in combined.columns for c in required_cols):
        print("[warn] plot_rho_cv_other_perf: missing required columns; skipping.")
        return

    df = combined.copy()
    model = str(model or "").strip()
    if model:
        df = df[df["mode"].astype(str) == model].copy()
        if df.empty:
            print()
            modes = sorted(combined["mode"].astype(str).dropna().unique().tolist())
            print(modes)
            preview = ", ".join(modes[:12]) + (" ..." if len(modes) > 12 else "")
            print(f"[warn] plot_rho_cv_other_perf: no rows for model='{model}'.")
            print(f"[info] available models: {preview}")
            return

    for c in ("rho_target", "leak", "input_scale", "neuron_bias", *metric_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["rho_target"])].copy()
    if df.empty:
        print("[warn] plot_rho_cv_other_perf: no finite rho_target values.")
        return
    df = df[df["rho_target"] > 0].copy()
    if df.empty:
        print(
            "[warn] plot_rho_cv_other_perf: no positive rho_target values for log-scale plot."
        )
        return

    df = _assign_group_ids(df)
    df_agg = _aggregate_over_hparams(df, metric_cols)
    if df_agg.empty:
        print("[warn] plot_rho_cv_other_perf: no aggregated rows after deduplication.")
        return

    keys = [
        "mode",
        "src",
        "group_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
    ]
    df_long = df_agg.melt(
        id_vars=keys, value_vars=metric_cols, var_name="metric", value_name="value"
    )

    per_group_rho = df_long.groupby(
        ["mode", "src", "group_id", "rho_target", "metric"], as_index=False
    ).agg(
        cv_other=("value", lambda x: _dispersion(x.to_numpy(dtype=float))),
        perf_mean=("value", "mean"),
        n_other_hparams=("value", "size"),
    )

    if per_group_rho.empty:
        print("[warn] plot_rho_cv_other_perf: no data to plot.")
        return

    modes_to_plot = sorted(per_group_rho["mode"].astype(str).dropna().unique().tolist())
    if model:
        modes_to_plot = [model]
    if not modes_to_plot:
        print("[warn] plot_rho_cv_other_perf: no modes to plot.")
        return

    saved_any = False
    for mode_name in modes_to_plot:
        mode_rows = per_group_rho[per_group_rho["mode"].astype(str) == mode_name].copy()
        if mode_rows.empty:
            continue

        summary = mode_rows.groupby(["rho_target", "metric"], as_index=False).agg(
            cv_other=("cv_other", "mean"),
            perf_mean=("perf_mean", "mean"),
            n_groups=("group_id", "size"),
        )
        if summary.empty:
            continue

        left_metrics = [m for m in ("MC", "IPC") if m in metric_cols]
        right_metrics = [m for m in ("GR", "KR") if m in metric_cols]
        panel_specs: list[tuple[str, list[str]]] = []
        if left_metrics:
            panel_specs.append(("MC + IPC", left_metrics))
        if right_metrics:
            panel_specs.append(("GR + KR", right_metrics))
        if not panel_specs:
            continue

        fig = plt.figure(figsize=(6.2 * len(panel_specs), 6), dpi=140)
        colors = mpl.colormaps["tab10"]
        metric_color = {
            "MC": colors(0),
            "IPC": colors(1),
            "KR": colors(2),
            "GR": colors(3),
        }
        rho_vals = sorted(
            v
            for v in summary["rho_target"].dropna().unique()
            if np.isfinite(v) and v > 0
        )
        tick_vals = []
        if rho_vals:
            preferred_ticks = [0.5, 1.0, 2.0, 4.0, 10.0]
            tick_vals = [
                v for v in preferred_ticks if any(np.isclose(v, rv) for rv in rho_vals)
            ]
            if len(tick_vals) < min(4, len(rho_vals)):
                k = min(len(rho_vals), 6)
                idx = np.unique(np.linspace(0, len(rho_vals) - 1, k, dtype=int))
                tick_vals = [rho_vals[i] for i in idx]

        plotted_any = False
        for panel_idx, (_panel_title, panel_metrics) in enumerate(panel_specs, start=1):
            ax = fig.add_subplot(1, len(panel_specs), panel_idx, projection="3d")
            panel_plotted = False
            for metric in panel_metrics:
                sub = summary[summary["metric"] == metric].copy()
                if sub.empty:
                    continue
                sub = sub[
                    np.isfinite(sub["rho_target"])
                    & np.isfinite(sub["cv_other"])
                    & np.isfinite(sub["perf_mean"])
                ]
                if sub.empty:
                    continue
                sub = sub.sort_values("rho_target")
                xs = np.log10(sub["rho_target"].to_numpy(float))
                ys = sub["cv_other"].to_numpy(float)
                zs = sub["perf_mean"].to_numpy(float)
                ax.plot(
                    xs,
                    ys,
                    zs,
                    marker="o",
                    color=metric_color.get(metric, colors(0)),
                    label=metric,
                )
                panel_plotted = True
                plotted_any = True

            if not panel_plotted:
                plt.delaxes(ax)
                continue

            if tick_vals:
                ax.set_xticks(np.log10(tick_vals))
                ax.set_xticklabels([f"{v}" for v in tick_vals])
            ax.set_xlabel("rho (log10)", fontsize=13, labelpad=2)
            ax.set_ylabel("Mean CV", fontsize=11, labelpad=2)
            ax.set_zlabel("mean metric", fontsize=13, labelpad=3)
            # Match both panels to the left/front viewpoint.
            ax.view_init(elev=22, azim=-35)
            _style_3d_axis(ax, tick_labelsize=12, tick_pad=-1)
            # Requested: integer std ticks for GR/KR panel only; keep MC/IPC as floats.
            if (
                "GR" in panel_metrics
                and "MC" not in panel_metrics
                and "IPC" not in panel_metrics
            ):
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
            else:
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))

        if not plotted_any:
            plt.close(fig)
            continue

        fig.subplots_adjust(left=0.00, right=0.99, bottom=0.04, top=0.99, wspace=0.02)

        safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", mode_name)
        out_name = f"rho_cv_other_perf_3d.{safe_mode}.png"
        out_fig = os.path.join(out_dir, out_name)
        fig.savefig(
            out_fig,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.005,
            facecolor="white",
            edgecolor="white",
        )
        if show:
            plt.show()
        plt.close(fig)
        print(f"[saved] {out_fig}")
        saved_any = True

    if not saved_any:
        print("[warn] plot_rho_cv_other_perf: no finite data to plot.")


def _rho_delta_by_mode(
    combined: pd.DataFrame,
    baseline_mode: str | None,
    *,
    rho_col: str = "raw_rho",
) -> dict[str, float]:
    """Mean raw-rho difference by mode relative to the requested baseline mode."""
    if (
        not baseline_mode
        or rho_col not in combined.columns
        or "mode" not in combined.columns
    ):
        return {}
    tmp = combined[["mode", rho_col]].copy()
    tmp["mode"] = tmp["mode"].astype(str)
    tmp[rho_col] = pd.to_numeric(tmp[rho_col], errors="coerce")
    mean_rho = tmp.groupby("mode", sort=False)[rho_col].mean()
    if baseline_mode not in mean_rho.index or not np.isfinite(
        mean_rho.loc[baseline_mode]
    ):
        return {}
    baseline_rho = float(mean_rho.loc[baseline_mode])
    return {
        str(mode): float(value - baseline_rho)
        for mode, value in mean_rho.items()
        if np.isfinite(value)
    }
