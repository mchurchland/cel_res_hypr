"""Contours for reservoir research figures."""

from cel_res_hypr.paths import _replace_path
from cel_res_hypr.plotting.style import (
    PERFORMANCE_METRICS,
    _add_sign_balance_colorbar,
    _mode_marker_map,
    _mode_styles_for_cv_performance,
    _rho_delta_colormap,
    _short_legend_name,
    _validate_color_scheme,
)
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_cv_performance_contours_2d(
    disp: pd.DataFrame,
    mean_tbl: pd.DataFrame,
    out_dir: str,
    bins: int = 36,
    show: bool = True,
    contour_percent: float = 50.0,
    rho_delta_by_mode: dict[str, float] | None = None,
    rho_baseline_label: str | None = None,
    baseline_mode: str | None = None,
    color_scheme: str = "thesis",
) -> str | None:
    """Plot top-down highest-density contours of joint CV/performance trials."""
    contour_percent = float(contour_percent)
    _validate_color_scheme(color_scheme)
    if not (0.0 < contour_percent < 100.0):
        raise ValueError("contour_percent must be greater than 0 and less than 100.")
    contour_mass = contour_percent / 100.0
    print(f"[info] CV/performance contour coverage: {contour_percent:g}%")
    required_disp = {"mode", "metric", "dispersion"}
    required_mean = {"mode", "metric", "mean"}
    if not required_disp.issubset(disp.columns) or not required_mean.issubset(
        mean_tbl.columns
    ):
        print(
            "[warn] 2D contours require mode, metric, CV dispersion, and mean performance."
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
    joint["dispersion"] = pd.to_numeric(joint["dispersion"], errors="coerce")
    joint["mean"] = pd.to_numeric(joint["mean"], errors="coerce")
    joint = joint[np.isfinite(joint["dispersion"]) & np.isfinite(joint["mean"])].copy()
    if joint.empty:
        print("[warn] 2D contours: CV and performance trials did not match.")
        return None

    present_modes = list(dict.fromkeys(joint["mode"].astype(str)))
    modes, color_map, label_map, style_meta = _mode_styles_for_cv_performance(
        present_modes, color_scheme=color_scheme
    )
    # Omit the shuffled-sign signed-unit control from the main architecture figure.
    modes = [mode for mode in modes if mode != "global_sign_pres"]
    marker_map = style_meta.get("marker_map", _mode_marker_map(modes))
    plot_baseline_mode = baseline_mode if baseline_mode in modes else None
    if plot_baseline_mode is None:
        plot_baseline_mode = (
            "real" if "real" in modes else ("CE-real" if "CE-real" in modes else None)
        )

    rho_delta_by_mode = rho_delta_by_mode or {}
    finite_rho_delta = np.array(
        [
            rho_delta_by_mode[mode]
            for mode in modes
            if np.isfinite(rho_delta_by_mode.get(mode, np.nan))
        ],
        dtype=float,
    )
    rho_color_scalar = None
    use_rho_colors = finite_rho_delta.size > 0
    if use_rho_colors:
        max_abs = float(np.nanmax(np.abs(finite_rho_delta)))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        rho_norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        rho_cmap = _rho_delta_colormap(color_scheme)
        rho_color_scalar = mpl.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
        rho_color_scalar.set_array([])
        for mode in modes:
            delta = rho_delta_by_mode.get(mode, np.nan)
            if np.isfinite(delta):
                color_map[mode] = mpl.colors.to_hex(rho_cmap(rho_norm(delta)))

    metrics = [
        metric for metric in PERFORMANCE_METRICS if metric in set(joint["metric"])
    ]
    if not metrics:
        print("[warn] 2D contours: none of IPC/KR/MC were available.")
        return None

    bins = max(24, min(int(bins), 52))
    fig = plt.figure(figsize=(8.25, 4.35), dpi=300)
    grid = fig.add_gridspec(
        1,
        len(metrics),
        left=0.075,
        right=0.865,
        bottom=0.355,
        top=0.910,
        wspace=0.24,
    )
    flat_axes = np.array(
        [fig.add_subplot(grid[0, idx]) for idx in range(len(metrics))],
        dtype=object,
    )
    plotted_any = False

    for panel_idx, metric in enumerate(metrics):
        ax = flat_axes[panel_idx]
        metric_df = joint[joint["metric"] == metric]
        cv_values = metric_df["dispersion"].to_numpy(float)
        perf_values = metric_df["mean"].to_numpy(float)
        cv_lo, cv_hi = float(np.nanmin(cv_values)), float(np.nanmax(cv_values))
        perf_lo, perf_hi = float(np.nanmin(perf_values)), float(np.nanmax(perf_values))
        cv_pad = max(0.035 * (cv_hi - cv_lo), 1e-9)
        perf_pad = max(0.035 * (perf_hi - perf_lo), 1e-9)
        cv_edges = np.linspace(cv_lo - cv_pad, cv_hi + cv_pad, bins + 1)
        perf_edges = np.linspace(perf_lo - perf_pad, perf_hi + perf_pad, bins + 1)
        cv_centers = 0.5 * (cv_edges[:-1] + cv_edges[1:])
        perf_centers = 0.5 * (perf_edges[:-1] + perf_edges[1:])
        grid_cv, grid_perf = np.meshgrid(cv_centers, perf_centers, indexing="ij")

        draw_modes = [mode for mode in modes if mode != plot_baseline_mode]
        if plot_baseline_mode is not None:
            draw_modes.append(plot_baseline_mode)

        for mode in draw_modes:
            sub = metric_df[metric_df["mode"] == mode]
            if len(sub) < 2:
                continue
            x = sub["dispersion"].to_numpy(float)
            y = sub["mean"].to_numpy(float)
            hist, _, _ = np.histogram2d(x, y, bins=(cv_edges, perf_edges))
            density = gaussian_filter(hist, sigma=1.35, mode="constant")
            total = float(density.sum())
            if total <= 0:
                continue

            ordered = np.sort(density.ravel())[::-1]
            cumulative = np.cumsum(ordered) / total
            idx = min(int(np.searchsorted(cumulative, contour_mass)), len(ordered) - 1)
            contour_level = float(ordered[idx])
            levels = [contour_level] if contour_level > 0 else []
            if levels:
                if mode == plot_baseline_mode:
                    baseline_levels = [levels[-1]]
                    baseline_widths = [2.25]
                    ax.contour(
                        grid_cv,
                        grid_perf,
                        density,
                        levels=baseline_levels,
                        colors=["white"],
                        linewidths=[width + 1.25 for width in baseline_widths],
                        alpha=0.96,
                        zorder=8,
                    )
                    ax.contour(
                        grid_cv,
                        grid_perf,
                        density,
                        levels=baseline_levels,
                        colors=[color_map[mode]],
                        linewidths=baseline_widths,
                        alpha=1.0,
                        zorder=9,
                    )
                else:
                    ax.contour(
                        grid_cv,
                        grid_perf,
                        density,
                        levels=[levels[-1]],
                        colors=[color_map[mode]],
                        linewidths=1.25,
                        alpha=0.86,
                        zorder=4,
                    )
            is_baseline = mode == plot_baseline_mode
            mean_marker = marker_map.get(mode, "o")
            ax.scatter(
                [float(np.mean(x))],
                [float(np.mean(y))],
                s=82 if mean_marker == "*" else 42,
                color=color_map[mode],
                marker=mean_marker,
                edgecolor="#222222",
                linewidth=0.70,
                zorder=10 if is_baseline else 6,
            )
            plotted_any = True

        ax.set_title(metric, fontsize=16.0, fontweight="semibold", pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=11.0, length=3.6, width=0.8, pad=2.0)
        ax.grid(True, color="#cfcfcf", alpha=0.22, linewidth=0.55)
        ax.margins(x=0.015, y=0.025)
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_color("#303030")

    for idx in range(len(metrics), len(flat_axes)):
        flat_axes[idx].set_axis_off()
    if not plotted_any:
        plt.close(fig)
        print("[warn] 2D contours: no finite architecture contours could be drawn.")
        return None

    legend_handles = []
    for mode in modes:
        is_baseline = mode == plot_baseline_mode
        marker = marker_map.get(mode, "o")
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[mode],
                linewidth=2.2 if is_baseline else 1.6,
                marker=marker,
                markersize=8.2 if marker == "*" else 6.0,
                markeredgecolor="#222222",
                markeredgewidth=0.70,
                label=label_map.get(mode, _short_legend_name(mode)),
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=min(3, max(1, len(legend_handles))),
        frameon=False,
        fontsize=13.0,
        columnspacing=0.70,
        handlelength=1.40,
        handletextpad=0.50,
        borderaxespad=0.0,
    )
    fig.text(0.470, 0.245, "Mean CV", fontsize=13.5, ha="center")
    fig.text(0.018, 0.635, "Mean performance", fontsize=13.5, va="center", rotation=90)
    if use_rho_colors and rho_color_scalar is not None:
        cax = fig.add_axes([0.895, 0.195, 0.016, 0.70])
        cbar = fig.colorbar(rho_color_scalar, cax=cax, orientation="vertical")
        baseline_text = rho_baseline_label or "baseline"
        cbar.set_label(
            r"$\Delta \rho_{\mathrm{raw}}$ from " + baseline_text,
            fontsize=11.0,
            rotation=270,
            labelpad=10,
        )
        cbar.ax.tick_params(labelsize=10.5, length=3.0, width=0.7, pad=1.8)
        cbar.outline.set_linewidth(0.45)
        cbar.outline.set_edgecolor("#333333")
    elif style_meta.get("sign_balance_colorbar"):
        _add_sign_balance_colorbar(
            fig,
            style_meta.get("ce_frac"),
            bottom=0.325,
            left=0.890,
            width=0.016,
            height=0.585,
            labelsize=11,
            orientation="vertical",
            color_scheme=color_scheme,
        )

    os.makedirs(out_dir, exist_ok=True)
    out_png = _replace_path(
        os.path.join(out_dir, "cv_performance_density_contours.png")
    )
    fig.savefig(out_png, dpi=300, facecolor="white")
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def _cv_performance_joint_table(
    disp: pd.DataFrame, mean_tbl: pd.DataFrame
) -> pd.DataFrame | None:
    required_disp = {"mode", "metric", "dispersion"}
    required_mean = {"mode", "metric", "mean"}
    if not required_disp.issubset(disp.columns) or not required_mean.issubset(
        mean_tbl.columns
    ):
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
    joint["dispersion"] = pd.to_numeric(joint["dispersion"], errors="coerce")
    joint["mean"] = pd.to_numeric(joint["mean"], errors="coerce")
    joint = joint[np.isfinite(joint["dispersion"]) & np.isfinite(joint["mean"])].copy()
    return joint


def plot_cv_performance_contour_triptych(
    disp: pd.DataFrame,
    mean_tbl: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    bins: int = 36,
    show: bool = True,
    contour_percent: float = 50.0,
    axis_disp: pd.DataFrame | None = None,
    axis_mean_tbl: pd.DataFrame | None = None,
    axis_label: str | None = None,
    color_scheme: str = "thesis",
) -> str | None:
    """Plot the three shuffle-control comparisons for IPC, KR, and MC."""
    contour_percent = float(contour_percent)
    _validate_color_scheme(color_scheme)
    if not (0.0 < contour_percent < 100.0):
        raise ValueError("contour_percent must be greater than 0 and less than 100.")
    contour_mass = contour_percent / 100.0
    print(f"[info] CV/performance triptych contour coverage: {contour_percent:g}%")

    joint = _cv_performance_joint_table(disp, mean_tbl)
    if joint is None:
        print(
            "[warn] contour triptych requires mode, metric, CV dispersion, and mean performance."
        )
        return None
    if joint.empty:
        print("[warn] contour triptych: CV and performance trials did not match.")
        return None

    axis_joint = joint
    if axis_disp is not None and axis_mean_tbl is not None:
        ref_joint = _cv_performance_joint_table(axis_disp, axis_mean_tbl)
        if ref_joint is None or ref_joint.empty:
            print(
                "[warn] contour triptych: axis reference did not produce CV/performance rows; using plotted data limits."
            )
        else:
            axis_joint = ref_joint
            if axis_label:
                print(f"[info] contour triptych axis limits from: {axis_label}")

    comparison_specs = [
        {
            "row_label": "C. elegans",
            "baseline": "real",
            "modes": ["real", "shuffle", "celW+connShuf", "conn_shuf_only"],
        },
        {
            "row_label": "PM1",
            "baseline": "local_sign+binary",
            "modes": [
                "local_sign+binary",
                "binary+shuffle",
                "global_sign_pres",
                "binary+conshuffle+wshuffle",
            ],
        },
        {
            "row_label": "Binary",
            "baseline": "binary_base",
            "modes": ["binary_base", "binary_base_topology_shuffle"],
        },
    ]
    available_modes = set(joint["mode"].astype(str))
    row_specs = []
    for spec in comparison_specs:
        modes = [mode for mode in spec["modes"] if mode in available_modes]
        baseline = spec["baseline"] if spec["baseline"] in modes else None
        if len(modes) < 2 or baseline is None:
            print(
                f"[warn] contour triptych: skipping {spec['row_label']} because required modes are absent."
            )
            continue
        row_specs.append({**spec, "modes": modes, "baseline": baseline})
    if not row_specs:
        print("[warn] contour triptych: no complete comparison rows were available.")
        return None

    metrics = [
        metric for metric in PERFORMANCE_METRICS if metric in set(joint["metric"])
    ]
    if not metrics:
        print("[warn] contour triptych: none of IPC/KR/MC were available.")
        return None

    rho_delta_lookup: dict[tuple[str, str], float] = {}
    finite_delta = []
    if "raw_rho" in combined.columns:
        rho_src = combined[["mode", "raw_rho"]].copy()
        rho_src["mode"] = rho_src["mode"].astype(str)
        rho_src["raw_rho"] = pd.to_numeric(rho_src["raw_rho"], errors="coerce")
        mean_rho = rho_src.groupby("mode", sort=False)["raw_rho"].mean()
        for spec in row_specs:
            baseline = spec["baseline"]
            if baseline not in mean_rho.index or not np.isfinite(
                mean_rho.loc[baseline]
            ):
                continue
            baseline_rho = float(mean_rho.loc[baseline])
            denom = max(abs(baseline_rho), 1e-12)
            for mode in spec["modes"]:
                if mode in mean_rho.index and np.isfinite(mean_rho.loc[mode]):
                    delta = float((mean_rho.loc[mode] - baseline_rho) / denom * 100.0)
                    rho_delta_lookup[(baseline, mode)] = delta
                    finite_delta.append(delta)

    use_rho_colors = len(finite_delta) > 0
    if use_rho_colors:
        max_abs = float(np.nanmax(np.abs(np.asarray(finite_delta, dtype=float))))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        color_norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        color_cmap = _rho_delta_colormap(color_scheme)
        color_scalar = mpl.cm.ScalarMappable(norm=color_norm, cmap=color_cmap)
        color_scalar.set_array([])
    else:
        color_norm = None
        color_cmap = None
        color_scalar = None

    marker_map = _mode_marker_map(
        [mode for spec in row_specs for mode in spec["modes"]]
    )
    fallback_palette = mpl.colormaps[
        "Dark2" if color_scheme == "extended-abstract" else "tab20"
    ]
    fallback_colors = {
        mode: mpl.colors.to_hex(fallback_palette(idx % fallback_palette.N))
        for idx, mode in enumerate(
            dict.fromkeys(mode for spec in row_specs for mode in spec["modes"])
        )
    }

    all_triptych_modes = {mode for spec in row_specs for mode in spec["modes"]}
    metric_limits: dict[str, tuple[float, float, float, float]] = {}
    axis_quantiles = (0.02, 0.98)

    def _robust_axis_limits(
        values: np.ndarray, pad_frac: float = 0.06
    ) -> tuple[float, float] | None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if values.size >= 12:
            lo, hi = np.nanquantile(values, axis_quantiles)
        else:
            lo, hi = np.nanmin(values), np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi <= lo:
            center = float(lo)
            spread = max(abs(center) * 0.02, 1e-3)
            return center - spread, center + spread
        pad = max(pad_frac * float(hi - lo), 1e-9)
        return float(lo - pad), float(hi + pad)

    def _clean_contour_components(contour_set, min_area_frac: float = 0.10) -> None:
        if not getattr(contour_set, "allsegs", None):
            return
        segments = [
            np.asarray(seg, dtype=float)
            for seg in contour_set.allsegs[0]
            if len(seg) >= 4 and np.all(np.isfinite(seg))
        ]
        if len(segments) <= 1:
            return
        areas = []
        for seg in segments:
            x = seg[:, 0]
            y = seg[:, 1]
            areas.append(
                abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            )
        max_area = max(areas) if areas else 0.0
        if max_area <= 0:
            return
        kept = [
            seg
            for seg, area in zip(segments, areas)
            if area >= min_area_frac * max_area
        ]
        if not kept:
            kept = [segments[int(np.argmax(areas))]]
        contour_set.set_paths(
            [mpl.path.Path.make_compound_path(*(mpl.path.Path(seg) for seg in kept))]
        )

    for metric in metrics:
        metric_all = axis_joint[
            (axis_joint["metric"] == metric)
            & (axis_joint["mode"].isin(all_triptych_modes))
        ]
        if metric_all.empty:
            metric_all = joint[
                (joint["metric"] == metric) & (joint["mode"].isin(all_triptych_modes))
            ]
        if metric_all.empty:
            continue
        cv_limits = _robust_axis_limits(metric_all["dispersion"].to_numpy(float))
        perf_limits = _robust_axis_limits(metric_all["mean"].to_numpy(float))
        if cv_limits is None or perf_limits is None:
            continue
        metric_limits[metric] = (*cv_limits, *perf_limits)

    bins = max(24, min(int(bins), 52))
    fig = plt.figure(figsize=(8.25, 6.55), dpi=300)
    grid = fig.add_gridspec(
        len(row_specs),
        len(metrics),
        left=0.125,
        right=0.790,
        bottom=0.255,
        top=0.940,
        wspace=0.28,
        hspace=0.36,
    )

    plotted_any = False
    legend_handles_by_mode: dict[str, Line2D] = {}

    for row_idx, spec in enumerate(row_specs):
        row_modes = spec["modes"]
        baseline = spec["baseline"]
        row_df = joint[joint["mode"].isin(row_modes)].copy()
        for col_idx, metric in enumerate(metrics):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            metric_df = row_df[row_df["metric"] == metric]
            if metric_df.empty:
                ax.set_axis_off()
                continue

            if metric not in metric_limits:
                ax.set_axis_off()
                continue
            cv_lo, cv_hi, perf_lo, perf_hi = metric_limits[metric]
            cv_edges = np.linspace(cv_lo, cv_hi, bins + 1)
            perf_edges = np.linspace(perf_lo, perf_hi, bins + 1)
            cv_centers = 0.5 * (cv_edges[:-1] + cv_edges[1:])
            perf_centers = 0.5 * (perf_edges[:-1] + perf_edges[1:])
            grid_cv, grid_perf = np.meshgrid(cv_centers, perf_centers, indexing="ij")

            draw_modes = [mode for mode in row_modes if mode != baseline] + [baseline]
            for mode in draw_modes:
                sub = metric_df[metric_df["mode"] == mode]
                if len(sub) < 2:
                    continue
                x = sub["dispersion"].to_numpy(float)
                y = sub["mean"].to_numpy(float)
                hist, _, _ = np.histogram2d(x, y, bins=(cv_edges, perf_edges))
                density = gaussian_filter(hist, sigma=1.35, mode="constant")
                total = float(density.sum())
                if total <= 0:
                    continue
                ordered = np.sort(density.ravel())[::-1]
                cumulative = np.cumsum(ordered) / total
                idx = min(
                    int(np.searchsorted(cumulative, contour_mass)), len(ordered) - 1
                )
                contour_level = float(ordered[idx])
                if contour_level <= 0:
                    continue

                delta = rho_delta_lookup.get((baseline, mode), np.nan)
                if use_rho_colors and np.isfinite(delta):
                    color = mpl.colors.to_hex(color_cmap(color_norm(delta)))
                else:
                    color = fallback_colors[mode]
                is_baseline = mode == baseline
                if is_baseline:
                    baseline_contour = ax.contour(
                        grid_cv,
                        grid_perf,
                        density,
                        levels=[contour_level],
                        colors=["white"],
                        linewidths=[3.35],
                        alpha=0.96,
                        zorder=8,
                    )
                    _clean_contour_components(baseline_contour)
                main_contour = ax.contour(
                    grid_cv,
                    grid_perf,
                    density,
                    levels=[contour_level],
                    colors=[color],
                    linewidths=[1.85 if is_baseline else 1.05],
                    alpha=1.0 if is_baseline else 0.86,
                    zorder=9 if is_baseline else 4,
                )
                _clean_contour_components(main_contour)
                marker = marker_map.get(mode, "o")
                ax.scatter(
                    [float(np.mean(x))],
                    [float(np.mean(y))],
                    s=62 if marker == "*" else 30,
                    color=color,
                    marker=marker,
                    edgecolor="#222222",
                    linewidth=0.50,
                    zorder=10 if is_baseline else 6,
                )
                plotted_any = True
                if mode not in legend_handles_by_mode:
                    legend_label = (
                        "PM1"
                        if mode == "local_sign+binary"
                        else _short_legend_name(mode)
                    )
                    legend_handles_by_mode[mode] = Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=1.9 if is_baseline else 1.35,
                        marker=marker,
                        markersize=6.8 if marker == "*" else 5.2,
                        markeredgecolor="#222222",
                        markeredgewidth=0.50,
                        label=legend_label,
                    )

            if row_idx == 0:
                ax.set_title(metric, fontsize=15.5, fontweight="semibold", pad=5)
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    spec["row_label"],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12.0,
                    fontweight="semibold",
                )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xlim(cv_lo, cv_hi)
            ax.set_ylim(perf_lo, perf_hi)
            ax.tick_params(axis="both", labelsize=10.0, length=3.1, width=0.65, pad=1.6)
            ax.grid(True, color="#cfcfcf", alpha=0.21, linewidth=0.45)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#303030")

    if not plotted_any:
        plt.close(fig)
        print("[warn] contour triptych: no finite contours could be drawn.")
        return None

    fig.text(0.448, 0.196, "Mean CV", fontsize=13.5, ha="center")
    fig.text(0.012, 0.575, "Mean performance", fontsize=13.5, va="center", rotation=90)

    if use_rho_colors and color_scalar is not None:
        cax = fig.add_axes([0.230, 0.105, 0.435, 0.018])
        cbar = fig.colorbar(color_scalar, cax=cax, orientation="horizontal")
        cbar.set_label(
            r"$\Delta \rho_{\mathrm{raw}}$ from row baseline (%)",
            fontsize=13.0,
            labelpad=3,
        )
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=11.5, length=3.0, width=0.65, pad=1.8)
        cbar.outline.set_linewidth(0.45)
        cbar.outline.set_edgecolor("#333333")

    row_centers = np.linspace(0.825, 0.370, len(row_specs))
    for spec, row_center in zip(row_specs, row_centers):
        row_handles = [
            legend_handles_by_mode[mode]
            for mode in spec["modes"]
            if mode in legend_handles_by_mode
        ]
        if not row_handles:
            continue
        fig.legend(
            handles=row_handles,
            loc="center left",
            bbox_to_anchor=(0.805, row_center),
            ncol=1,
            frameon=False,
            fontsize=12.0,
            handlelength=1.55,
            handletextpad=0.48,
            labelspacing=0.55,
            borderaxespad=0.0,
        )

    os.makedirs(out_dir, exist_ok=True)
    out_png = _replace_path(
        os.path.join(out_dir, "cv_performance_density_contours_triptych.png")
    )
    fig.savefig(
        out_png,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    print(f"[saved] {out_png}")
    if show:
        plt.show()
    plt.close(fig)
    return out_png
