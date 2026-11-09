"""Weight controls for reservoir research figures."""

from cel_res_hypr.analysis.aggregation import _compute_mean_table
from cel_res_hypr.plotting.io import _load_local_sign_binary_overlay
from cel_res_hypr.plotting.rendering import (
    _clear_3d_axis_background,
    _save_3d_front_back,
    _style_3d_axis,
    _tight_layout_quiet,
)
from cel_res_hypr.plotting.style import _mode_numeric_value
from cel_res_hypr.plotting.tables import _write_weight_metric_difference_table
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_weight_gauss_mean_cv(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    3D line plot for weight_test (and other numeric modes):
      x = Gaussian magnitude (alpha) on log scale (log10 transform)
      y = mean CV (dispersion) per metric (MC/IPC/KR/GR)
      z = mean of an invariance column (prefers `kl_to_gaussian`)
    CV is not computed for the invariance column; we reuse the dispersion table for MC/IPC/KR/GR only.
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_cv: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    pos_logs = [np.log10(v) for _, v in mode_vals if v > 0]
    if not pos_logs:
        print(
            "[warn] plot_weight_gauss_mean_cv: need at least one positive noise magnitude."
        )
        return
    has_zero = any(v == 0 for _, v in mode_vals)
    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)
    x_zero = (min(pos_logs) - 0.60) if has_zero else np.nan

    def _x_plot(v: float) -> float:
        if v > 0:
            return float(np.log10(v))
        if v == 0 and has_zero:
            return float(x_zero)
        return float("nan")

    mean_tbl = _compute_mean_table(combined)
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_mean_cv: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include kl_to_gaussian."
        )
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    mean_mean_lookup = (
        inv_tbl[inv_tbl["metric"] == inv_metric].groupby("mode")["mean"].mean()
    )
    if mean_mean_lookup.empty:
        print(
            f"[warn] plot_weight_gauss_mean_cv: no values for invariance metric '{inv_metric}'."
        )
        return
    z_label_base = "KL to Gaussian"
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()

    # Metrics that have both mean and cv
    metrics = [
        m
        for m in ("MC", "IPC", "KR", "GR")
        if (m in mean_tbl["metric"].unique()) and (m in disp["metric"].unique())
    ]
    if not metrics:
        print("[warn] plot_weight_gauss_mean_cv: no overlapping metrics with mean+CV.")
        return

    # Choose a scaling for the invariance column so z-values stay in a readable range.
    max_mean = np.nanmax(mean_mean_lookup.values) if len(mean_mean_lookup) else np.nan
    if np.isfinite(max_mean) and max_mean > 0:
        z_power = int(np.floor(np.log10(max_mean)))
        z_scale = 10.0**z_power
    else:
        z_power = 0
        z_scale = 1.0

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    colors = mpl.colormaps["tab10"]
    plotted = False
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = mean_mean_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_cv = cv_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            x_plot = _x_plot(x_raw)
            if not (np.isfinite(y_cv) and np.isfinite(z_mean) and np.isfinite(x_plot)):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((x_plot, y_cv, inv_scaled))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        # y-axis is metric value, z-axis is KL.
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

        z_lsb = (
            cv_lookup.get((zero_mode, metric), np.nan)
            if zero_mode is not None
            else np.nan
        )
        if not np.isfinite(z_lsb) and overlay is not None:
            z_lsb = overlay["cv_lookup"].get(("real", metric), np.nan)
        if np.isfinite(z_lsb):
            x_plane = np.array(
                [[x_min_line, x_max_line], [x_min_line, x_max_line]], dtype=float
            )
            y_plane = np.full_like(x_plane, float(z_lsb))
            z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
            ax.plot_surface(
                x_plane,
                y_plane,
                z_plane,
                color=colors(idx % 10),
                alpha=0.12,
                shade=False,
                linewidth=0.0,
                antialiased=False,
            )
            ax.plot(
                [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                [z_lsb, z_lsb, z_lsb, z_lsb, z_lsb],
                [z_min, z_min, z_max, z_max, z_min],
                color=colors(idx % 10),
                linewidth=1.0,
                alpha=0.7,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_cv: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_ylabel("Mean CV", fontsize=18, labelpad=12)
    ax.set_zlabel(z_label_base, fontsize=18, labelpad=10)
    # helpful x ticks at common magnitudes if they are within range
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    xtick_pos = [np.log10(v) for v in xticks]
    xtick_lbl = [str(v) for v in xticks]
    if has_zero:
        xtick_pos = [x_zero] + xtick_pos
        xtick_lbl = ["0"] + xtick_lbl
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl)
    x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
    ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
    # ax.legend()
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_cv_log3d.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(25, -15),
        back_view=(25, 15),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
        pane_fill=True,
        pane_facecolor=(0.95, 0.95, 0.95, 0.90),
        pane_edgecolor=(0.78, 0.78, 0.78, 1.0),
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_mean_perf(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    3D line plot like plot_weight_gauss_mean_cv but using mean performance instead of mean CV:
      x = Gaussian magnitude (alpha) on log scale
      y = mean performance (per metric MC/IPC/KR/GR)
      z = invariance column (prefers `kl_to_gaussian`)
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_perf: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    pos_logs = [np.log10(v) for _, v in mode_vals if v > 0]
    if not pos_logs:
        print(
            "[warn] plot_weight_gauss_mean_perf: need at least one positive noise magnitude."
        )
        return
    has_zero = any(v == 0 for _, v in mode_vals)
    x_zero = (min(pos_logs) - 0.60) if has_zero else np.nan

    def _x_plot(v: float) -> float:
        if v > 0:
            return float(np.log10(v))
        if v == 0 and has_zero:
            return float(x_zero)
        return float("nan")

    mean_tbl = _compute_mean_table(combined)
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_mean_perf: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include kl_to_gaussian."
        )
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    mean_mean_lookup = (
        inv_tbl[inv_tbl["metric"] == inv_metric].groupby("mode")["mean"].mean()
    )
    if mean_mean_lookup.empty:
        print(
            f"[warn] plot_weight_gauss_mean_perf: no values for invariance metric '{inv_metric}'."
        )
        return
    z_label_base = "KL to Gaussian"

    # Determine z scaling for nicer scientific-label axis
    max_mean = np.nanmax(mean_mean_lookup.values) if len(mean_mean_lookup) else np.nan
    if np.isfinite(max_mean) and max_mean > 0:
        z_power = int(np.floor(np.log10(max_mean)))
        z_scale = 10.0**z_power
    else:
        z_power = 0
        z_scale = 1.0

    # Metrics present in mean table (performance means)
    metrics = [m for m in ("MC", "IPC", "KR", "GR") if m in mean_tbl["metric"].unique()]
    if not metrics:
        print("[warn] plot_weight_gauss_mean_perf: no metrics with mean values found.")
        return

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    colors = mpl.colormaps["tab10"]
    plotted = False
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = mean_mean_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_mean = mean_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            x_plot = _x_plot(x_raw)
            if not (
                np.isfinite(y_mean) and np.isfinite(z_mean) and np.isfinite(x_plot)
            ):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((x_plot, y_mean, inv_scaled))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        # y-axis is metric value, z-axis is KL.
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

        if overlay is not None:
            z_lsb = overlay["mean_lookup"].get(("real", metric), np.nan)
            if has_zero:
                y0 = [r[1] for r in rows if abs(r[0] - x_zero) < 1e-12]
                if y0:
                    z_lsb = float(y0[0])
            if np.isfinite(z_lsb):
                x_plane = np.array(
                    [[x_min_line, x_max_line], [x_min_line, x_max_line]], dtype=float
                )
                y_plane = np.full_like(x_plane, float(z_lsb))
                z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
                ax.plot_surface(
                    x_plane,
                    y_plane,
                    z_plane,
                    color=colors(idx % 10),
                    alpha=0.12,
                    shade=False,
                    linewidth=0.0,
                    antialiased=False,
                )
                ax.plot(
                    [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                    [z_lsb, z_lsb, z_lsb, z_lsb, z_lsb],
                    [z_min, z_min, z_max, z_max, z_min],
                    color=colors(idx % 10),
                    linewidth=1.0,
                    alpha=0.7,
                )
                plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_perf: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_ylabel("Mean Metric Value", fontsize=18, labelpad=10)
    ax.set_zlabel(z_label_base, fontsize=18, labelpad=18)
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    xtick_pos = [np.log10(v) for v in xticks]
    xtick_lbl = [str(v) for v in xticks]
    if has_zero:
        xtick_pos = [x_zero] + xtick_pos
        xtick_lbl = ["0"] + xtick_lbl
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl)
    x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
    ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_perf_log3d.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(22, -15),
        back_view=(22, 35),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_cv_metric_2d(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
):
    """
    2D weight-test plot for interpolation sweeps:
      x = interpolation fraction parsed from mode
      left y = mean CV across repeat groups
      right y = mean metric value across repeat groups
    """
    os.makedirs(out_dir, exist_ok=True)

    if disp.empty:
        print("[warn] plot_weight_gauss_cv_metric_2d: dispersion table is empty.")
        return

    mode_vals = []
    for mode in disp["mode"].dropna().unique():
        value = _mode_numeric_value(mode)
        if np.isfinite(value):
            mode_vals.append((mode, value))
    if not mode_vals:
        print(
            "[warn] plot_weight_gauss_cv_metric_2d: no modes with numeric values found."
        )
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])

    mean_tbl = _compute_mean_table(combined)
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    disp_metrics = set(disp["metric"].dropna().unique())
    mean_metrics = set(mean_tbl["metric"].dropna().unique())
    metrics = [
        m for m in ("MC", "IPC", "KR", "GR") if m in disp_metrics and m in mean_metrics
    ]
    if not metrics:
        print(
            "[warn] plot_weight_gauss_cv_metric_2d: no overlapping MC/IPC/KR/GR metrics found."
        )
        return

    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)
    _write_weight_metric_difference_table(
        cv_lookup,
        "Mean CV",
        out_dir,
        tex_name="weight_gauss_diff_mean_cv_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        log_prefix="plot_weight_gauss_cv_metric_2d",
        mode_value_symbol="f",
    )
    _write_weight_metric_difference_table(
        mean_lookup,
        "Mean Metric Value",
        out_dir,
        tex_name="weight_gauss_diff_mean_perf_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        log_prefix="plot_weight_gauss_cv_metric_2d",
        mode_value_symbol="f",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), dpi=300, sharex=True)
    colors = mpl.colormaps["tab10"]

    def _draw_panel(ax, lookup, y_label: str) -> bool:
        plotted = False
        for idx, metric in enumerate(metrics):
            rows = []
            for mode, x_val in mode_vals:
                y_val = lookup.get((mode, metric), np.nan)
                if np.isfinite(x_val) and np.isfinite(y_val):
                    rows.append((float(x_val), float(y_val)))
            if not rows:
                continue
            rows = sorted(rows, key=lambda t: t[0])
            xs, ys = map(np.asarray, zip(*rows))
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                color=colors(idx % 10),
                label=metric,
            )
            plotted = True
        ax.set_ylabel(y_label, fontsize=14)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        return plotted

    plotted_cv = _draw_panel(axes[0], cv_lookup, "Mean CV")
    plotted_mean = _draw_panel(axes[1], mean_lookup, "Mean Metric Value")

    if not (plotted_cv or plotted_mean):
        plt.close(fig)
        print("[warn] plot_weight_gauss_cv_metric_2d: no finite data to plot.")
        return

    x_vals = [value for _, value in mode_vals if np.isfinite(value)]
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    x_pad = 0.06 * (x_max - x_min if x_max > x_min else 1.0)
    x_label = "Interpolation fraction"
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=min(6, max(2, len(set(x_vals)))))
        )
        ax.set_xlabel(x_label, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.92, wspace=0.14)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=len(handles),
            frameon=False,
            fontsize=10,
        )
    else:
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.97, wspace=0.14)

    out_png = os.path.join(out_dir, "weight_mean_cv_metric_2d.png")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_weight_gauss_perf_cv_grid(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    Direct combined 2x2 figure (no intermediate perf/cv files):
      top row: mean performance (front/back)
      bottom:  mean CV (front/back)
      x-axis: raw Gaussian noise magnitude alpha
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print(
            "[warn] plot_weight_gauss_perf_cv_grid: no modes with numeric values found."
        )
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    has_zero = any(v == 0 for _, v in mode_vals)
    x_zero = 0.0 if has_zero else np.nan

    def _x_plot(v: float) -> float:
        return float(v) if np.isfinite(v) else float("nan")

    mean_tbl = _compute_mean_table(combined)
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_perf_cv_grid: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include KL weight diagnostics."
        )
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    kl_lookup = inv_tbl[inv_tbl["metric"] == inv_metric].groupby("mode")["mean"].mean()
    z_label = "KL to Gaussian"
    print(
        f"[info] plot_weight_gauss_perf_cv_grid: using {inv_metric} for z-axis ({z_label})."
    )
    if kl_lookup.empty:
        print(
            f"[warn] plot_weight_gauss_perf_cv_grid: no values for invariance metric '{inv_metric}'."
        )
        return

    metrics = [
        m
        for m in ("MC", "IPC", "KR", "GR")
        if (m in mean_tbl["metric"].unique()) and (m in disp["metric"].unique())
    ]
    if not metrics:
        print(
            "[warn] plot_weight_gauss_perf_cv_grid: no overlapping metrics with mean+CV."
        )
        return

    max_kl = np.nanmax(kl_lookup.values) if len(kl_lookup) else np.nan
    if np.isfinite(max_kl) and max_kl > 0:
        z_power = int(np.floor(np.log10(max_kl)))
        z_scale = 10.0**z_power
    else:
        z_scale = 1.0

    colors = mpl.colormaps["tab10"]
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    if not x_line_vals:
        print("[warn] plot_weight_gauss_perf_cv_grid: no finite x values to plot.")
        return
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = kl_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)

    def _draw_panel(ax, y_lookup, y_label: str):
        plotted = False
        y_lo_panel = np.inf
        y_hi_panel = -np.inf
        for idx, metric in enumerate(metrics):
            rows = []
            for mode, x_raw in mode_vals:
                y_val = y_lookup.get((mode, metric), np.nan)
                z_val = kl_lookup.get(mode, np.nan)
                x_val = _x_plot(x_raw)
                if not (
                    np.isfinite(y_val) and np.isfinite(z_val) and np.isfinite(x_val)
                ):
                    continue
                rows.append((x_val, y_val, z_val / z_scale))
            if not rows:
                continue
            rows = sorted(rows, key=lambda t: t[0])
            xs, ys, zs = map(np.asarray, zip(*rows))
            ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10))
            plotted = True
            y_lo_panel = min(y_lo_panel, float(np.nanmin(ys)))
            y_hi_panel = max(y_hi_panel, float(np.nanmax(ys)))

            if overlay is not None:
                if y_label == "Mean Metric Value":
                    y_ref = overlay["mean_lookup"].get(("real", metric), np.nan)
                else:
                    y_ref = overlay["cv_lookup"].get(("real", metric), np.nan)
                if has_zero:
                    y0 = [r[1] for r in rows if abs(r[0] - x_zero) < 1e-12]
                    if y0:
                        y_ref = float(y0[0])
                if np.isfinite(y_ref):
                    x_plane = np.array(
                        [[x_min_line, x_max_line], [x_min_line, x_max_line]],
                        dtype=float,
                    )
                    y_plane = np.full_like(x_plane, float(y_ref))
                    z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
                    ax.plot_surface(
                        x_plane,
                        y_plane,
                        z_plane,
                        color=colors(idx % 10),
                        alpha=0.05,
                        shade=False,
                        linewidth=0.0,
                        antialiased=False,
                    )
                    ax.plot(
                        [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                        [y_ref, y_ref, y_ref, y_ref, y_ref],
                        [z_min, z_min, z_max, z_max, z_min],
                        color=colors(idx % 10),
                        linewidth=1.0,
                        alpha=0.7,
                    )

        if not plotted:
            ax.set_axis_off()
            return False

        ax.set_xlabel("Noise Mag.", fontsize=18, labelpad=6)
        ax.set_ylabel(y_label, fontsize=18, labelpad=7)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(z_label, fontsize=18, labelpad=10, rotation=90)
        ax.zaxis.set_label_coords(1.06, 0.50)
        ax.xaxis.label.set_clip_on(False)
        ax.yaxis.label.set_clip_on(False)
        ax.zaxis.label.set_clip_on(False)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
        ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
        if np.isfinite(y_lo_panel) and np.isfinite(y_hi_panel):
            y_pad = 0.08 * (y_hi_panel - y_lo_panel if y_hi_panel > y_lo_panel else 1.0)
            ax.set_ylim(y_lo_panel - y_pad, y_hi_panel + y_pad)
        ax.set_zlim(z_min, z_max)
        _style_3d_axis(ax, tick_labelsize=18, tick_pad=-1)
        _clear_3d_axis_background(ax)
        return True

    _write_weight_metric_difference_table(
        cv_lookup,
        "Mean CV",
        out_dir,
        tex_name="weight_gauss_diff_mean_cv_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        overlay_lookup=overlay["cv_lookup"] if overlay is not None else None,
        log_prefix="plot_weight_gauss_perf_cv_grid",
    )
    _write_weight_metric_difference_table(
        mean_lookup,
        "Mean Metric Value",
        out_dir,
        tex_name="weight_gauss_diff_mean_perf_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        overlay_lookup=overlay["mean_lookup"] if overlay is not None else None,
        log_prefix="plot_weight_gauss_perf_cv_grid",
    )

    fig = plt.figure(figsize=(14.0, 10.0), dpi=300)
    ax00 = fig.add_subplot(2, 2, 1, projection="3d")
    ax01 = fig.add_subplot(2, 2, 2, projection="3d")
    ax10 = fig.add_subplot(2, 2, 3, projection="3d")
    ax11 = fig.add_subplot(2, 2, 4, projection="3d")

    any_plotted = False

    any_plotted |= _draw_panel(ax10, cv_lookup, "Mean CV")
    any_plotted |= _draw_panel(ax11, cv_lookup, "Mean CV")
    any_plotted |= _draw_panel(ax00, mean_lookup, "Mean Metric Value")
    any_plotted |= _draw_panel(ax01, mean_lookup, "Mean Metric Value")
    if not any_plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_perf_cv_grid: no finite data to plot.")
        return

    front_view = (22, -20)
    back_view = (22, 20)
    for ax in (ax00, ax10):
        ax.view_init(elev=front_view[0], azim=front_view[1])
        ax.set_zorder(100)
    for ax in (ax01, ax11):
        ax.view_init(elev=back_view[0], azim=back_view[1])
        ax.set_zorder(2)

    # Bottom-row y endpoints can crowd x-corner ticks; prune only y endpoints.
    for ax in (ax10, ax11, ax00, ax01):
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="upper"))

    fig.subplots_adjust(
        left=-0.00, right=0.955, bottom=-0.0, top=1.05, wspace=0.2, hspace=-0.05
    )
    fig.tight_layout()
    out_png = os.path.join(out_dir, "weight_mean_perf_cv_grid_3d.png")
    fig.savefig(
        out_png,
        dpi=600,
        facecolor="white",
        edgecolor="white",
    )
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_png}")
