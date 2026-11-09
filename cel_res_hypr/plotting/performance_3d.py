"""Performance 3d for reservoir research figures."""

from cel_res_hypr.paths import _replace_path
from cel_res_hypr.plotting.rendering import _style_3d_axis
from cel_res_hypr.plotting.style import (
    _add_sign_balance_colorbar,
    _mode_marker_map,
    _mode_styles_for_cv_performance,
    _short_legend_name,
    _short_thesis_name,
)
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def _mean_performance_differences(
    mean_tbl: pd.DataFrame,
    baseline_mode: str,
) -> pd.DataFrame:
    """Calculate control-minus-baseline mean-performance differences by metric."""
    required = {"mode", "metric", "mean"}
    if mean_tbl.empty or not required.issubset(mean_tbl.columns):
        return pd.DataFrame()

    rows: list[dict] = []
    pairing_keys = [key for key in ("src", "group_id") if key in mean_tbl.columns]
    for metric in ("MC", "IPC", "KR", "GR"):
        metric_df = mean_tbl[mean_tbl["metric"] == metric]
        baseline = metric_df[metric_df["mode"] == baseline_mode].dropna(subset=["mean"])
        if baseline.empty:
            continue
        for control_mode in sorted(set(metric_df["mode"].dropna()) - {baseline_mode}):
            control = metric_df[metric_df["mode"] == control_mode].dropna(
                subset=["mean"]
            )
            if control.empty:
                continue

            paired = pd.DataFrame()
            if pairing_keys:
                base_pair = baseline[pairing_keys + ["mean"]].rename(
                    columns={"mean": "baseline_performance"}
                )
                control_pair = control[pairing_keys + ["mean"]].rename(
                    columns={"mean": "control_performance"}
                )
                paired = base_pair.merge(control_pair, on=pairing_keys, how="inner")

            if not paired.empty:
                baseline_values = paired["baseline_performance"].to_numpy(float)
                control_values = paired["control_performance"].to_numpy(float)
            else:
                baseline_values = baseline["mean"].to_numpy(float)
                control_values = control["mean"].to_numpy(float)

            baseline_mean = float(np.mean(baseline_values))
            control_mean = float(np.mean(control_values))
            delta = control_mean - baseline_mean
            pct_delta = (
                100.0 * delta / abs(baseline_mean)
                if abs(baseline_mean) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "metric": metric,
                    "control_mode": control_mode,
                    "baseline_mean_performance": baseline_mean,
                    "control_mean_performance": control_mean,
                    "delta_performance": delta,
                    "pct_delta_performance": pct_delta,
                }
            )
    return pd.DataFrame(rows)


def plot_cv_performance_differences_3d(
    cv_differences: pd.DataFrame,
    mean_tbl: pd.DataFrame,
    out_dir: str,
    baseline_mode: str = "real",
    show: bool = True,
) -> str | None:
    """Plot isolated CV bumps whose colors encode mean-performance change."""
    if cv_differences.empty:
        print("[warn] interactive 3D plot: no CV differences available.")
        return None
    if "baseline_mode" in cv_differences.columns:
        baseline_modes = cv_differences["baseline_mode"].dropna().astype(str).unique()
        if len(baseline_modes):
            baseline_mode = baseline_modes[0]

    performance = _mean_performance_differences(mean_tbl, baseline_mode)
    if performance.empty:
        print("[warn] interactive 3D plot: no mean-performance differences available.")
        return None

    plot_df = cv_differences.merge(
        performance,
        on=["control_mode", "metric"],
        how="inner",
    )
    plot_df = plot_df[
        np.isfinite(plot_df["pct_delta_cv"])
        & np.isfinite(plot_df["pct_delta_performance"])
    ].copy()
    if plot_df.empty:
        print("[warn] interactive 3D plot: CV and performance rows did not overlap.")
        return None

    # Order controls from most CE-like to least CE-like in joint percent-difference space.
    control_order = (
        plot_df.assign(
            distance=lambda frame: np.hypot(
                frame["pct_delta_cv"], frame["pct_delta_performance"]
            )
        )
        .groupby("control_mode")["distance"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    x_lookup = {mode: idx for idx, mode in enumerate(control_order)}
    label_lookup = (
        plot_df.drop_duplicates("control_mode")
        .set_index("control_mode")["control_label"]
        .to_dict()
    )
    metric_order = [
        metric
        for metric in ("MC", "IPC", "KR", "GR")
        if metric in set(plot_df["metric"])
    ]
    metric_lookup = {metric: idx for idx, metric in enumerate(metric_order)}

    perf_limit = float(np.nanmax(np.abs(plot_df["pct_delta_performance"])))
    perf_limit = max(perf_limit, 1e-9)
    color_norm = mpl.colors.TwoSlopeNorm(
        vmin=-perf_limit,
        vcenter=0.0,
        vmax=perf_limit,
    )
    color_map = mpl.colormaps["coolwarm"]

    fig = plt.figure(figsize=(14.0, 8.6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    # Each category/metric pair gets its own compact Gaussian cap. The caps do not
    # touch, so the surface does not imply continuity between categorical controls.
    local_axis = np.linspace(-0.42, 0.42, 31)
    local_x, local_y = np.meshgrid(local_axis, local_axis)
    radial = np.exp(-0.5 * ((local_x / 0.17) ** 2 + (local_y / 0.17) ** 2))
    radial[radial < 0.025] = np.nan
    for _, row in plot_df.iterrows():
        x_center = float(x_lookup[row["control_mode"]])
        y_center = float(metric_lookup[row["metric"]])
        cv_height = float(row["pct_delta_cv"])
        performance_color = float(row["pct_delta_performance"])
        rgba = color_map(color_norm(performance_color))
        ax.plot_surface(
            x_center + local_x,
            y_center + local_y,
            cv_height * radial,
            color=rgba,
            alpha=0.92,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.scatter(
            [x_center],
            [y_center],
            [cv_height],
            color=[rgba],
            edgecolors="#222222",
            linewidths=0.45,
            s=18,
            depthshade=False,
        )

    x_plane, y_plane = np.meshgrid(
        np.linspace(-0.5, len(control_order) - 0.5, 2),
        np.linspace(-0.5, len(metric_order) - 0.5, 2),
    )
    ax.plot_surface(
        x_plane,
        y_plane,
        np.zeros_like(x_plane),
        color="#777777",
        alpha=0.08,
        linewidth=0,
        shade=False,
    )

    x_values = np.arange(len(control_order), dtype=float)
    y_values = np.arange(len(metric_order), dtype=float)
    ax.set_xlabel("Control architecture", labelpad=17)
    ax.set_ylabel("Metric", labelpad=10)
    ax.set_zlabel(r"Mean CV difference from CE (\%)", labelpad=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [label_lookup.get(mode, _short_thesis_name(mode)) for mode in control_order],
        fontsize=8,
        rotation=18,
        ha="right",
    )
    ax.set_yticks(y_values)
    ax.set_yticklabels(metric_order, fontsize=10)
    ax.set_title("CV landscape colored by performance difference", pad=22)
    ax.view_init(elev=29, azim=-55)
    _style_3d_axis(ax, tick_labelsize=9, tick_pad=1)
    color_scalar = mpl.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    color_scalar.set_array([])
    colorbar = fig.colorbar(color_scalar, ax=ax, shrink=0.64, pad=0.08, aspect=24)
    colorbar.set_label(r"Mean performance difference from CE (\%)")
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.08, top=0.92)

    os.makedirs(out_dir, exist_ok=True)
    out_png = _replace_path(os.path.join(out_dir, "cv_performance_differences_3d.png"))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out_png}")
    if show:
        print(
            "[info] 3D bump window opened; drag to rotate and use the toolbar to zoom."
        )
        plt.show()
    plt.close(fig)
    return out_png


def plot_cv_performance_hills_3d(
    disp: pd.DataFrame,
    mean_tbl: pd.DataFrame,
    out_dir: str,
    bins: int = 18,
    show: bool = True,
) -> str | None:
    """Add performance to the CV histograms as smooth 3D joint-distribution hills."""
    required_disp = {"mode", "metric", "dispersion"}
    required_mean = {"mode", "metric", "mean"}
    if not required_disp.issubset(disp.columns) or not required_mean.issubset(
        mean_tbl.columns
    ):
        print(
            "[warn] 3D hills require mode, metric, CV dispersion, and mean performance."
        )
        return None

    join_keys = ["mode", "metric"] + [
        key
        for key in ("src", "group_id")
        if key in disp.columns and key in mean_tbl.columns
    ]
    left = disp[join_keys + ["dispersion"]].copy()
    right = mean_tbl[join_keys + ["mean"]].copy()
    for key in ("mode", "metric", "src", "group_id"):
        if key in join_keys:
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)
    joint = left.merge(right, on=join_keys, how="inner")
    joint = joint[
        np.isfinite(pd.to_numeric(joint["dispersion"], errors="coerce"))
        & np.isfinite(pd.to_numeric(joint["mean"], errors="coerce"))
    ].copy()
    if joint.empty:
        print("[warn] 3D hills: CV and performance trials did not match.")
        return None

    present_modes = list(dict.fromkeys(joint["mode"].astype(str)))
    modes, color_map, label_map, style_meta = _mode_styles_for_cv_performance(
        present_modes
    )
    marker_map = style_meta.get("marker_map", _mode_marker_map(modes))

    metrics = [
        metric for metric in ("GR", "IPC", "KR", "MC") if metric in set(joint["metric"])
    ]
    if not metrics:
        print("[warn] 3D hills: none of GR/IPC/KR/MC were available.")
        return None

    bins = max(10, min(int(bins), 28))
    # Use explicit sizes here because the 2D histogram routine intentionally
    # enlarges global rcParams for its own publication layout.
    fig = plt.figure(figsize=(13.0, 10.0), dpi=140)
    panel_positions = (
        (0.025, 0.560, 0.440, 0.340),
        (0.515, 0.560, 0.440, 0.340),
        (0.025, 0.145, 0.440, 0.340),
        (0.515, 0.145, 0.440, 0.340),
    )
    axes = []
    plotted_any = False
    for panel_idx, metric in enumerate(metrics):
        ax = fig.add_axes(panel_positions[panel_idx], projection="3d")
        axes.append(ax)
        metric_df = joint[joint["metric"] == metric].copy()
        cv_values = metric_df["dispersion"].to_numpy(float)
        perf_values = metric_df["mean"].to_numpy(float)
        if not len(cv_values):
            ax.set_axis_off()
            continue

        cv_lo, cv_hi = float(np.nanmin(cv_values)), float(np.nanmax(cv_values))
        perf_lo, perf_hi = float(np.nanmin(perf_values)), float(np.nanmax(perf_values))
        if np.isclose(cv_lo, cv_hi):
            cv_lo, cv_hi = cv_lo - 0.5, cv_hi + 0.5
        if np.isclose(perf_lo, perf_hi):
            perf_lo, perf_hi = perf_lo - 0.5, perf_hi + 0.5
        cv_pad = 0.03 * (cv_hi - cv_lo)
        perf_pad = 0.03 * (perf_hi - perf_lo)
        cv_edges = np.linspace(cv_lo - cv_pad, cv_hi + cv_pad, bins + 1)
        perf_edges = np.linspace(perf_lo - perf_pad, perf_hi + perf_pad, bins + 1)
        cv_centers = 0.5 * (cv_edges[:-1] + cv_edges[1:])
        perf_centers = 0.5 * (perf_edges[:-1] + perf_edges[1:])
        grid_cv, grid_perf = np.meshgrid(cv_centers, perf_centers, indexing="ij")

        panel_peak = 0.0
        for mode in modes:
            sub = metric_df[metric_df["mode"] == mode]
            if sub.empty:
                continue
            hist, _, _ = np.histogram2d(
                sub["dispersion"].to_numpy(float),
                sub["mean"].to_numpy(float),
                bins=(cv_edges, perf_edges),
            )
            height = gaussian_filter(
                hist / max(len(sub), 1), sigma=1.0, mode="constant"
            )
            if height.sum() > 0:
                height *= (hist.sum() / max(len(sub), 1)) / height.sum()
            peak = float(np.nanmax(height)) if height.size else 0.0
            if peak <= 0:
                continue
            panel_peak = max(panel_peak, peak)
            height_plot = height.copy()
            height_plot[height_plot < peak * 0.025] = np.nan
            ax.plot_surface(
                grid_cv,
                grid_perf,
                height_plot,
                color=color_map[mode],
                alpha=0.42,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
            plotted_any = True

        ax.set_title(metric, fontsize=18, fontweight="semibold", pad=5)
        ax.set_xlabel("Mean CV", fontsize=12.5, labelpad=6)
        ax.set_ylabel("Mean performance", fontsize=12.5, labelpad=7)
        ax.set_zlabel("")
        if panel_peak > 0:
            ax.set_zlim(0, panel_peak * 1.12)
        ax.set_proj_type("ortho")
        ax.view_init(elev=31, azim=-55)
        ax.set_box_aspect((1.18, 1.0, 0.72), zoom=1.12)
        _style_3d_axis(ax, tick_labelsize=10, tick_pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"].update(
                color=(0.55, 0.55, 0.55, 0.28),
                linewidth=0.55,
            )

    if not plotted_any:
        plt.close(fig)
        print("[warn] 3D hills: no finite architecture surfaces could be drawn.")
        return None

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[mode],
            linewidth=4,
            marker=marker_map.get(mode, "o"),
            markersize=8,
            markeredgecolor="#222222",
            label=label_map.get(mode, _short_legend_name(mode)),
        )
        for mode in modes
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, max(1, len(legend_handles))),
        frameon=True,
        fontsize=10.5,
        columnspacing=1.25,
        handlelength=2.2,
    )
    fig.text(
        0.012,
        0.515,
        "Normalized trial fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    if style_meta.get("sign_balance_colorbar"):
        _add_sign_balance_colorbar(fig, style_meta.get("ce_frac"), bottom=0.074)
    os.makedirs(out_dir, exist_ok=True)
    out_png = _replace_path(os.path.join(out_dir, "cv_performance_hills_3d.png"))
    fig.savefig(out_png, dpi=300, facecolor="white")
    print(f"[saved] {out_png}")
    if show:
        print("[info] 3D hill window opened; drag any panel to rotate it.")
        plt.show()
    plt.close(fig)
    return out_png
