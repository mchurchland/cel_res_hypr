"""Legacy plotting CLI and compatibility exports."""

from cel_res_hypr.analysis.aggregation import (
    _compute_dispersion_table,
    _compute_mean_table,
    _assign_group_ids,
    _aggregate_over_hparams,
    _dispersion_cv as _dispersion,
    _unique_hparam_rows,
)
from cel_res_hypr.paths import _replace_path, _safe_path
from cel_res_hypr.plotting import mpl
from cel_res_hypr.plotting.cli import parse_args
from cel_res_hypr.plotting.comparison import (
    _metric_bounds_from_reference,
    _mode_metric_means,
    _normalize_compare_metrics,
    _pairwise_tost_results,
    compare_mode_metric_means,
    plot_mode_self_drop_comparison,
    print_tost_preservation_summary,
)
from cel_res_hypr.plotting.contours import (
    _cv_performance_joint_table,
    plot_cv_performance_contour_triptych,
    plot_cv_performance_contours_2d,
)
from cel_res_hypr.plotting.histograms import plot_mc_vs_gr_all_arch
from cel_res_hypr.plotting.io import (
    _ensure_columns,
    _load_local_sign_binary_overlay,
    _read_combined_csv,
    _read_compare_table,
)
from cel_res_hypr.plotting.performance_3d import (
    _mean_performance_differences,
    plot_cv_performance_differences_3d,
    plot_cv_performance_hills_3d,
)
from cel_res_hypr.plotting.rendering import (
    _capture_figure_rgba,
    _clear_3d_axis_background,
    _format_wt_mean_axis_as_offset,
    _pad_height_center,
    _pad_width_center,
    _save_3d_front_back,
    _save_publication_figure,
    _style_3d_axis,
    _tight_layout_quiet,
    _trim_white_border,
)
from cel_res_hypr.plotting.rho import _rho_delta_by_mode, plot_rho_cv_other_perf
from cel_res_hypr.plotting.sign_fraction import (
    plot_frac_arch_histograms,
    plot_frac_cv_meanline,
)
from cel_res_hypr.plotting.sign_normalization import (
    _infer_normalization_from_mode,
    _parse_sign_fraction_from_mode,
    _prepare_sign_norm_ablation_df,
    _save_sign_norm_combined_grid,
    _save_sign_norm_metric_grid,
    _save_sign_norm_scaling_grid,
    _sign_norm_label,
    _summarize_group_values,
    plot_sign_norm_ablation,
)
from cel_res_hypr.plotting.style import (
    COLOR_SCHEMES,
    PERFORMANCE_METRICS,
    SHORT_THESIS_NAMES,
    _SHORT_THESIS_NAME_ALIASES,
    _SIGN_BALANCE_COLOR_MAX,
    _add_sign_balance_colorbar,
    _architecture_sign_balance_color_map,
    _architecture_sign_balance_group,
    _detect_ce_sign_fraction,
    _is_grid_tenth,
    _mode_marker_map,
    _mode_numeric_value,
    _mode_sign_fraction,
    _mode_styles_for_cv_performance,
    _rho_delta_colormap,
    _short_legend_name,
    _short_thesis_name,
    _sign_balance_color,
    _sign_balance_colormap,
    _sign_balance_label,
    _sign_fraction_from_mode,
    _validate_color_scheme,
)
from cel_res_hypr.plotting.tables import (
    _cv_difference_latex_table,
    _latex_escape_table_text,
    _mean_difference_ci,
    _resolve_cv_baseline_mode,
    _write_weight_metric_difference_table,
    print_cv_difference_tables,
)
from cel_res_hypr.plotting.triads import plot_triad_sign_fraction_summary
from cel_res_hypr.plotting.weight_controls import (
    plot_weight_gauss_cv_metric_2d,
    plot_weight_gauss_mean_cv,
    plot_weight_gauss_mean_perf,
    plot_weight_gauss_perf_cv_grid,
)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys


# Optional exact mode allowlist for legacy imported workflows.
ANALYSIS_MODE_FILTER = []


def _filter_to_modes(df: pd.DataFrame, modes=None, label: str = "data") -> pd.DataFrame:
    """Restrict dataframe to an explicit mode allowlist."""
    if modes is None:
        modes = ANALYSIS_MODE_FILTER
    allow = [str(m).strip() for m in (modes or []) if str(m).strip()]
    if not allow:
        return df
    mode_values = df["mode"].astype(str).str.strip()
    mode_counts = mode_values.value_counts(dropna=False)
    out = df[mode_values.isin(allow)].copy()
    print(
        f"[info] {label}: mode filter active ({len(allow)} mode(s)); kept {len(out)}/{len(df)} rows"
    )
    requested_counts = {mode: int(mode_counts.get(mode, 0)) for mode in allow}
    print(
        f"[info] {label}: requested mode counts: "
        + ", ".join(f"{mode}={count}" for mode, count in requested_counts.items())
    )
    missing = [mode for mode, count in requested_counts.items() if count == 0]
    if missing:
        available = sorted(mode_values.dropna().unique().tolist())
        preview = ", ".join(available[:16]) + (" ..." if len(available) > 16 else "")
        print(
            f"[warn] {label}: requested mode(s) absent from input: {', '.join(missing)}"
        )
        print(f"[info] {label}: available modes: {preview}")
    return out


def plot_overlaid_arch_histograms(
    disp: pd.DataFrame,
    out_dir: str,
    bins: int,
    mode_preset: str = "all",
    rho_delta_by_mode: dict[str, float] | None = None,
    rho_baseline_label: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    present_modes = list(dict.fromkeys(disp["mode"].astype(str).dropna().tolist()))
    present_mode_set = set(present_modes)
    # Consistent ordering/colors to make panels easier to compare.
    mode_preset = str(mode_preset or "all").strip().lower()
    if ANALYSIS_MODE_FILTER:
        mode_order = [
            str(m) for m in ANALYSIS_MODE_FILTER if str(m) in present_mode_set
        ]
        if (
            "global_sign_pres" in present_mode_set
            and "global_sign_pres" not in mode_order
        ):
            mode_order.append("global_sign_pres")
    elif mode_preset == "all_shuf":
        mode_order = (
            "global_sign_pres",
            "real",
            "global_sign_pres_real_w",
            "global_sign_pres_real_weight",
            "shuffle",
            "conn_shuf_only",
            "celW+connShuf",
            "local_sign+binary",  ##this is localsign+signed binary weights
            "binary_base",  ## unsigned binary
            "binary_base_topology_shuffle",  ##unsigned binary w shuffle
            "binary+shuffle",
            "binary+conshuffle+wshuffle",
        )

    else:
        mode_order = [
            "real",
            "cel+randN",
            "er+randN",
            "ws_p0.1+randN",
            "local_sign",
            "local_sign+flat",
            "local_sign+binary",
            "global_sign_pres",
            "global_sign_pres_real_w",
            "global_sign_pres_real_weight",
            "binary_base",
        ]
    extras = [m for m in present_modes if m not in mode_order]
    modes = list(mode_order) + extras
    rho_delta_by_mode = rho_delta_by_mode or {}
    use_rho_colors = any(np.isfinite(v) for v in rho_delta_by_mode.values())
    color_norm = color_cmap = color_scalar = None
    if use_rho_colors:
        finite_delta = np.array(
            [
                rho_delta_by_mode[m]
                for m in modes
                if np.isfinite(rho_delta_by_mode.get(m, np.nan))
            ],
            dtype=float,
        )
        max_abs = float(np.nanmax(np.abs(finite_delta))) if finite_delta.size else 1.0
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
        color_norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        color_cmap = mpl.colormaps["coolwarm"]
        color_scalar = mpl.cm.ScalarMappable(norm=color_norm, cmap=color_cmap)
        color_scalar.set_array([])
        color_map = {
            mode_name: mpl.colors.to_hex(
                color_cmap(color_norm(rho_delta_by_mode[mode_name]))
            )
            for mode_name in modes
            if np.isfinite(rho_delta_by_mode.get(mode_name, np.nan))
        }
    else:
        # Keep these exact colors per request when rho diagnostics are unavailable.
        preserved_colors = {
            "real": "#32a2f2",
            "local_sign+binary": "#7488FF",
            "binary_base": "#78dee5",
        }
        # Colorblind-safe fallback palette for remaining modes.
        cb_palette = [
            "#E69F00",  # orange
            "#009E73",  # bluish green
            "#D55E00",  # vermillion
            "#CC79A7",  # reddish purple
            "#56B4E9",  # sky blue
            "#F0E442",  # yellow
            "#0072B2",  # blue
            "#000000",  # black
        ]
        color_map = dict(preserved_colors)
        palette_idx = 0
        for mode_name in modes:
            if mode_name in color_map:
                continue
            color_map[mode_name] = cb_palette[palette_idx % len(cb_palette)]
            palette_idx += 1

    if not metrics:
        return
    plt.rcParams.update(
        {
            "axes.titlesize": 25,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
        }
    )

    per_fig = 4  # 2x2 grid
    for start in range(0, len(metrics), per_fig):
        chunk = metrics[start : start + per_fig]
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 11), squeeze=False, sharex="col", sharey="row"
        )
        flat_axes = axes.ravel()
        any_plotted = False
        col_lo = [np.inf, np.inf]
        col_hi = [-np.inf, -np.inf]
        row_y_max = [0.0, 0.0]
        legend_handles, legend_labels = None, None
        data_mask = [False] * len(flat_axes)

        for idx, m in enumerate(chunk):
            ax = flat_axes[idx]
            all_vals = disp[disp["metric"] == m]["dispersion"].to_numpy()
            if all_vals.size == 0:
                ax.axis("off")
                continue
            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            col = idx % 2
            row = idx // 2
            col_lo[col] = min(col_lo[col], lo)
            col_hi[col] = max(col_hi[col], hi)

            plotted = False
            for mode in modes:
                s = disp[(disp["metric"] == m) & (disp["mode"] == mode)][
                    "dispersion"
                ].to_numpy()
                if s.size == 0:
                    continue
                counts, _ = np.histogram(s, bins=edges)
                frac = counts.astype(float) / max(
                    len(s), 1
                )  # normalize by N so areas comparable
                color = color_map.get(mode, None)
                display_mode = _short_legend_name(mode)
                ax.plot(
                    centers,
                    frac,
                    drawstyle="steps-mid",
                    linewidth=2.8,
                    alpha=0.9,
                    label=display_mode,
                    color=color,
                )
                # Keep the location marker tied to the same architecture color as its histogram.
                med = float(np.median(s))
                ax.axvline(
                    med,
                    color=color if color is not None else "#8a8a8a",
                    alpha=0.4,
                    linewidth=1.2,
                    linestyle="--",
                )
                plotted = True
                row_y_max[row] = max(row_y_max[row], float(np.max(frac)))

            if not plotted:
                ax.axis("off")
                continue

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.set_title(f"Invariance of {m}")
            if idx == 2 or idx == 3:
                ax.set_xlabel("Mean CV")
            if idx == 0 or idx == 2:
                ax.set_ylabel("fraction (normalized by N)")
            ax.grid(True, which="both", axis="both", alpha=0.18, linestyle=":")
            any_plotted = True
            data_mask[idx] = True

        for idx in range(len(chunk), len(flat_axes)):
            flat_axes[idx].axis("off")

        if not any_plotted:
            plt.close(fig)
            continue

        for col in range(2):
            if (
                np.isfinite(col_lo[col])
                and np.isfinite(col_hi[col])
                and col_lo[col] != col_hi[col]
            ):
                for row in range(2):
                    idx = row * 2 + col
                    ax = flat_axes[idx]
                    if data_mask[idx]:
                        ax.set_xlim(col_lo[col], col_hi[col])
        for row in range(2):
            if row_y_max[row] > 0:
                y_max = row_y_max[row] * 1.08
                for col in range(2):
                    idx = row * 2 + col
                    ax = flat_axes[idx]
                    if data_mask[idx]:
                        ax.set_ylim(0, y_max)
                        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if legend_handles and legend_labels:
            leg = fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=3,
                frameon=True,
                fontsize=16,
                columnspacing=1.0,
                handlelength=1.8,
                handletextpad=0.5,
                borderaxespad=0.2,
            )
            leg.get_frame().set_alpha(0.9)
        right_margin = 0.88 if use_rho_colors and color_scalar is not None else 0.995
        fig.subplots_adjust(
            left=0.09,
            right=right_margin,
            top=0.965,
            bottom=0.30,
            wspace=0.08,
            hspace=0.20,
        )
        if use_rho_colors and color_scalar is not None:
            cax = fig.add_axes([0.905, 0.38, 0.022, 0.42])
            cbar = fig.colorbar(color_scalar, cax=cax, orientation="vertical")
            baseline_text = rho_baseline_label or "baseline"
            cbar.set_label(
                rf"$\Delta\rho_{{\mathrm{{raw}}}}$ from {baseline_text}",
                fontsize=15,
                rotation=270,
                labelpad=22,
            )
            cbar.ax.tick_params(labelsize=13, width=0.7, length=3)
            cbar.outline.set_linewidth(0.6)
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        name_root = (
            "all_arch_hist_grid_all_shuf"
            if mode_preset == "all_shuf"
            else "all_arch_hist_grid"
        )
        out_fig = _safe_path(os.path.join(out_dir, f"{name_root}{suffix}.png"))
        fig.savefig(out_fig, dpi=300)
        plt.close(fig)
        print(f"[saved] {out_fig}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.triad_sign_fraction_plot:
        plot_triad_sign_fraction_summary(
            args.triad_summary_csv,
            args.out_dir,
            out_path=args.triad_plot_out,
            normalization=args.triad_normalization,
            triad_scope=args.triad_scope,
            rho_target=args.triad_rho_target,
        )
        return

    wants_compare = bool(
        args.compare_mean_a or args.compare_mean_b or args.compare_only
    )
    if wants_compare:
        if not (args.compare_mean_a and args.compare_mean_b):
            raise ValueError(
                "Both --compare-mean-a and --compare-mean-b are required for comparison mode."
            )
        if not os.path.isfile(args.compare_mean_a):
            raise FileNotFoundError(
                f"Comparison mean table not found: {args.compare_mean_a}"
            )
        if not os.path.isfile(args.compare_mean_b):
            raise FileNotFoundError(
                f"Comparison mean table not found: {args.compare_mean_b}"
            )

        out_cmp = args.compare_mean_out or _safe_path(
            os.path.join(args.out_dir, "mode_metric_mean_comparison.csv")
        )
        a_raw, _ = _read_compare_table(
            args.compare_mean_a, value_col=args.compare_value_col
        )
        b_raw, _ = _read_compare_table(
            args.compare_mean_b, value_col=args.compare_value_col
        )
        if ANALYSIS_MODE_FILTER:
            a_f = _filter_to_modes(a_raw, label="compare A")
            b_f = _filter_to_modes(b_raw, label="compare B")
            if a_f.empty:
                raise ValueError(
                    f"Mode filter removed all rows from compare A. Requested modes: {ANALYSIS_MODE_FILTER}"
                )
            if b_f.empty:
                raise ValueError(
                    f"Mode filter removed all rows from compare B. Requested modes: {ANALYSIS_MODE_FILTER}"
                )
            tmp_a = _safe_path(os.path.join(args.out_dir, "compare.filtered.A.csv"))
            tmp_b = _safe_path(os.path.join(args.out_dir, "compare.filtered.B.csv"))
            a_f.to_csv(tmp_a, index=False)
            b_f.to_csv(tmp_b, index=False)
            compare_a_path = tmp_a
            compare_b_path = tmp_b
        else:
            compare_a_path = args.compare_mean_a
            compare_b_path = args.compare_mean_b
        comp_tbl = compare_mode_metric_means(
            compare_a_path,
            compare_b_path,
            out_cmp,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
            value_col=args.compare_value_col,
        )
        out_cmp_plot = args.compare_plot_out or _safe_path(
            os.path.join(args.out_dir, "mode_metric_mean_self_drop.png")
        )
        metric_filter = _normalize_compare_metrics(args.compare_metric)
        plot_tbl = comp_tbl
        if metric_filter:
            plot_tbl = comp_tbl[comp_tbl["metric"].isin(metric_filter)].copy()
            if plot_tbl.empty:
                available = sorted(comp_tbl["metric"].dropna().unique())
                raise ValueError(
                    f"--compare-metric={args.compare_metric!r} did not match any metrics. "
                    f"Available: {available}"
                )
        plot_mode_self_drop_comparison(
            plot_tbl,
            out_cmp_plot,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
        )
        if args.compare_tost_preservation:
            ref_tbl, ref_col = _read_compare_table(
                compare_a_path, value_col=args.compare_value_col
            )
            new_tbl, new_col = _read_compare_table(
                compare_b_path, value_col=args.compare_value_col
            )
            if ref_col != new_col:
                raise ValueError(
                    f"TOST preservation needs same value column on both inputs, got: {ref_col} vs {new_col}"
                )
            out_tost = args.compare_tost_out or _safe_path(
                os.path.join(args.out_dir, "tost_preservation_summary.csv")
            )
            print_tost_preservation_summary(
                ref_tbl,
                new_tbl,
                value_col=ref_col,
                label_ref=args.compare_label_a,
                label_new=args.compare_label_b,
                alpha=float(args.compare_tost_alpha),
                bound_frac=float(args.compare_tost_bound_frac),
                metrics_filter=metric_filter,
                out_csv=out_tost,
            )
        if args.compare_only:
            return

    if not os.path.isfile(args.combined):
        raise FileNotFoundError(f"Combined CSV not found: {args.combined}")

    combined = _read_combined_csv(args.combined)
    combined = _ensure_columns(combined)
    combined = _filter_to_modes(combined, label="combined")
    if combined.empty:
        raise ValueError(
            f"Mode filter removed all rows from combined input. Requested modes: {ANALYSIS_MODE_FILTER}"
        )

    if args.sign_norm_ablation:
        plot_sign_norm_ablation(
            combined,
            args.out_dir,
            prefix=args.sign_norm_prefix,
            cv_rho_target=args.sign_norm_cv_rho_target,
            cv_include_rho=args.sign_norm_cv_include_rho,
        )
        return

    # Save a copy (non-destructive; versioned if exists)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    # combined.to_csv(out_comb, index=False)
    # print(f"[saved] {out_comb}  (rows={len(combined)})")

    # Compute and save dispersion table
    disp = _compute_dispersion_table(combined, mode="cv")
    # out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    # disp.to_csv(out_disp, index=False)

    # Plots
    # plot_frac_cv_meanline(disp, combined, args.out_dir, bins=args.frac_cv_bins)
    # plot_weight_gauss_mean_cv(disp, combined, args.out_dir,show=False)
    # if args.weight_gauss_cv_only:
    #    plot_weight_gauss_cv_metric_2d(disp, combined, args.out_dir, show=False)
    # else:
    #    plot_weight_gauss_perf_cv_grid(
    #        disp,
    #        combined,
    #        args.out_dir,
    #        show=False,
    #        local_sign_binary_csv=args.local_sign_binary_csv,
    #    )
    # plot_rho_cv_other_perf( combined, args.out_dir, show=True, model=args.model, drop_kr_gr=args.rho_cv_drop_kr_gr,)
    resolved_rho_baseline = _resolve_cv_baseline_mode(
        set(combined["mode"].astype(str).dropna()),
        args.cv_baseline_mode,
    )
    rho_delta = _rho_delta_by_mode(combined, resolved_rho_baseline, rho_col="raw_rho")
    plot_overlaid_arch_histograms(
        disp,
        args.out_dir,
        args.bins,
        rho_delta_by_mode=rho_delta,
        rho_baseline_label=_short_thesis_name(resolved_rho_baseline)
        if resolved_rho_baseline
        else None,
    )
    # plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)
    print_cv_difference_tables(disp, args.out_dir, baseline_mode=args.cv_baseline_mode)
    if (
        args.show_cv_performance_3d
        or args.show_cv_performance_contours
        or args.show_cv_performance_contour_triptych
    ):
        mean_tbl = _compute_mean_table(combined)
    triptych_axis_disp = None
    triptych_axis_mean_tbl = None
    triptych_axis_label = None
    if args.show_cv_performance_contour_triptych and args.triptych_axis_combined:
        if not os.path.isfile(args.triptych_axis_combined):
            raise FileNotFoundError(
                f"Triptych axis combined CSV not found: {args.triptych_axis_combined}"
            )
        axis_combined = _read_combined_csv(args.triptych_axis_combined)
        axis_combined = _ensure_columns(axis_combined)
        axis_combined = _filter_to_modes(axis_combined, label="triptych axis reference")
        if axis_combined.empty:
            raise ValueError(
                "Mode filter removed all rows from triptych axis reference. "
                f"Requested modes: {ANALYSIS_MODE_FILTER}"
            )
        triptych_axis_disp = _compute_dispersion_table(axis_combined, mode="cv")
        triptych_axis_mean_tbl = _compute_mean_table(axis_combined)
        triptych_axis_label = args.triptych_axis_combined
    if args.show_cv_performance_contours:
        plot_cv_performance_contours_2d(
            disp,
            mean_tbl,
            args.out_dir,
            bins=max(args.bins, 36),
            show=True,
            contour_percent=args.cv_performance_contour_percent,
            rho_delta_by_mode=None,
            rho_baseline_label=None,
            baseline_mode=resolved_rho_baseline,
            color_scheme=args.color_scheme,
        )
    if args.show_cv_performance_3d:
        plot_cv_performance_hills_3d(
            disp,
            mean_tbl,
            args.out_dir,
            bins=min(args.bins, 22),
            show=True,
        )
    if args.show_cv_performance_contour_triptych:
        plot_cv_performance_contour_triptych(
            disp,
            mean_tbl,
            combined,
            args.out_dir,
            bins=max(args.bins, 36),
            show=True,
            contour_percent=args.cv_performance_contour_percent,
            axis_disp=triptych_axis_disp,
            axis_mean_tbl=triptych_axis_mean_tbl,
            axis_label=triptych_axis_label,
            color_scheme=args.color_scheme,
        )


if __name__ == "__main__":
    main()
