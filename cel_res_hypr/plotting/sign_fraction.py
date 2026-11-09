"""Sign fraction for reservoir research figures."""

from cel_res_hypr.paths import _replace_path
from cel_res_hypr.plotting.rendering import _save_3d_front_back, _style_3d_axis
from cel_res_hypr.plotting.style import (
    PERFORMANCE_METRICS,
    _detect_ce_sign_fraction,
    _validate_color_scheme,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def plot_frac_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    if not metrics:
        return

    mode_order = [
        "sign_test_og_cel0.0",
        "sign_test_og_cel0.1",
        "sign_test_og_cel0.2",
        "sign_test_og_cel0.3",
        "sign_test_og_cel0.4",
        "sign_test_og_cel0.5",
        "sign_test_og_cel0.6",
        "sign_test_og_cel0.7",
        "sign_test_og_cel0.8",
        "sign_test_og_cel0.9",
        "sign_test_og_cel1.0",
    ]
    # mode_order = [
    #    "weight_test0.0","weight_test1.0","weight_test5.0","weight_test10.0","weight_test100.0",
    #    "weight_test1000.0", "weight_test10000.0",
    # ]
    modes = [m for m in mode_order if m in set(disp["mode"].unique())]
    if not modes:
        print("fdfsa")
        return

    def mode_to_value(mode: str) -> float:
        """Extract numeric value from a mode name.

        Supports both legacy "sign_testX" and current "weight_testX" patterns.
        Falls back to the first numeric token if no known prefix matches.
        Returns NaN when no numeric component is present.
        """
        mode_str = str(mode)
        for prefix in ("sign_test", "weight_test"):
            m = re.search(rf"{prefix}([0-9]*\.?[0-9]+)", mode_str)
            if m:
                return float(m.group(1))
        m = re.search(r"([0-9]*\.?[0-9]+)", mode_str)
        return float(m.group(1)) if m else np.nan

    mode_values = {m: mode_to_value(m) for m in modes}
    z_vals = np.array([v for v in mode_values.values() if np.isfinite(v)], dtype=float)

    # If nothing numeric was found, fall back to ordinal positions so the
    # plotting code remains robust instead of crashing.
    if z_vals.size == 0:
        mode_values = {m: float(i) for i, m in enumerate(modes)}
        z_vals = np.array(list(mode_values.values()), dtype=float)

    zmin, zmax = float(np.min(z_vals)), float(np.max(z_vals))
    if zmax == zmin:
        zmax = zmin + 1.0

    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    color_for_mode = {m: cmap(norm(mode_values[m])) for m in modes}

    if all(str(m).startswith("sign_test") for m in modes):
        x_label = "E/I Edge Balance"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value"
    else:
        x_label = "mode value (x)"

    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    per_fig = 4  # 2x2
    for start in range(0, len(metrics), per_fig):
        chunk = metrics[start : start + per_fig]
        fig = plt.figure(figsize=(14, 12))
        axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

        any_plotted = False

        for idx, metric in enumerate(chunk):
            ax = axes[idx]
            all_vals = disp.loc[disp["metric"] == metric, "dispersion"].to_numpy()
            if all_vals.size == 0:
                ax.set_axis_off()
                continue

            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0

            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])

            plotted_this = False
            for mode in modes:
                s = disp[(disp["metric"] == metric) & (disp["mode"] == mode)][
                    "dispersion"
                ].to_numpy()
                if s.size == 0:
                    continue

                counts, _ = np.histogram(s, bins=edges)
                z = counts.astype(float) / max(len(s), 1)  # histogram fraction -> now z
                x = mode_values.get(mode, np.nan)  # numeric mode value
                if not np.isfinite(x):
                    continue

                y = centers  # CV bin centers -> now y
                c = color_for_mode[mode]

                # 3D polyline at constant x
                ax.plot(
                    [x] * len(y), y, zs=z, zdir="z", linewidth=2.5, alpha=0.9, color=c
                )

                # Median CV marker line at constant x, spanning z
                # med = float(np.median(s))
                # z_max_local = float(np.max) if len else 0.0
                # ax.plot([x, x], [med, med], zs=[0.0, z_max_local], zdir="z",
                #        linewidth=1.2, alpha=0.5, color=c, linestyle="--")

                plotted_this = True

            if not plotted_this:
                ax.set_axis_off()
                continue

            any_plotted = True
            ax.set_title(f"{metric}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Mean CV")
            ax.set_zlabel("fraction (hist, z)")

            # Make x discrete and readable
            xticks = [mode_values[m] for m in modes if np.isfinite(mode_values[m])]
            xticks = sorted({float(v) for v in xticks})
            if xticks:
                ax.set_xticks(xticks)
                span = max(xticks) - min(xticks)
                pad = 0.05 * span if span > 0 else 1.0
                ax.set_xlim(min(xticks) - pad, max(xticks) + pad)
            group_max = (
                disp.loc[disp["metric"] == metric]
                .groupby("mode")["dispersion"]
                .apply(
                    lambda v: (
                        (
                            np.histogram(v.to_numpy(), bins=edges)[0].astype(float)
                            / max(len(v), 1)
                        ).max()
                        if len(v) > 0
                        else 0.0
                    )
                )
            )

            ymax = float(group_max.max()) if len(group_max) > 0 else 0.0
            ax.set_zlim(0.0, 1.05 * ymax)
            ax.set_ylim(lo, hi)

            # Better default view for vertical sheets
            ax.view_init(elev=22, azim=-35)

        # turn off unused panels
        for j in range(len(chunk), 4):
            axes[j].set_axis_off()

        if not any_plotted:
            plt.close(fig)
            continue

        # colorbar keyed to z (fraction)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        # cbar.set_label("sign_test frac ")

        fig.tight_layout()
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        out_fig = os.path.join(out_dir, f"all_arch_hist_grid_3d{suffix}.png")
        saved_paths = _save_3d_front_back(
            fig,
            axes,
            out_fig,
            front_view=(22, -35),
            back_view=(22, 145),
            dpi=600,
        )
        plt.show()
        plt.close(fig)
        for out_path in saved_paths:
            print(f"[saved] {out_path}")


def plot_frac_cv_meanline(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    bins: int = 4,
    show: bool = True,
    performance_scale: str = "linear",
    color_scheme: str = "thesis",
):
    """
    Stacked sign-sweep summary. The upper 3D panel has one colored line per
    higher-is-better performance metric (IPC, KR, MC):
      x = mode value (parsed numeric component)
      y = mean CV (dispersion) across groups
      z = mean performance across runs

    The lower panel shows the corresponding mean raw spectral radius across
    the same sign-fraction sweep, together with the mean CV across the plotted
    metrics on a second y-axis. It intentionally does not include a horizontal
    reference for the original C. elegans network.
    """
    os.makedirs(out_dir, exist_ok=True)
    _validate_color_scheme(color_scheme)
    if performance_scale not in {"linear", "sqrt", "fourth_root", "sixth_root"}:
        raise ValueError(
            "performance_scale must be 'linear', 'sqrt', 'fourth_root', or 'sixth_root'."
        )

    # Expected metrics present in both tables.
    metric_cols = [m for m in PERFORMANCE_METRICS if m in combined.columns]
    if not metric_cols:
        print("[warn] plot_frac_cv_meanline: no IPC/KR/MC columns found.")
        return
    metrics_disp = set(disp["metric"].unique())
    metrics = [m for m in metric_cols if m in metrics_disp]
    if not metrics:
        print(
            "[warn] plot_frac_cv_meanline: dispersion table missing IPC/KR/MC metrics."
        )
        return

    def mode_to_value(mode: str) -> float:
        mode_str = str(mode)
        for prefix in ("sign_test", "weight_test"):
            m = re.search(rf"{prefix}([0-9]*\.?[0-9]+)", mode_str)
            if m:
                return float(m.group(1))
        m = re.search(r"([0-9]*\.?[0-9]+)", mode_str)
        return float(m.group(1)) if m else np.nan

    # Keep only modes we can assign a numeric position to; sort by that value.
    mode_values = {m: mode_to_value(m) for m in disp["mode"].unique()}
    modes = [m for m, v in mode_values.items() if np.isfinite(v)]
    modes = sorted(modes, key=lambda m: mode_values[m])
    if not modes:
        print("[warn] plot_frac_cv_meanline: no modes with numeric value; skipping.")
        return

    # Mean performance and mean CV lookups.
    comb_long = combined.melt(
        id_vars=["mode"], value_vars=metrics, var_name="metric", value_name="value"
    )
    mean_perf = comb_long.groupby(["mode", "metric"])["value"].mean()
    mean_cv = disp.groupby(["mode", "metric"])["dispersion"].mean()

    if "raw_rho" not in combined.columns:
        print("[warn] plot_frac_cv_meanline: combined table missing raw_rho.")
        return

    fig = plt.figure(figsize=(4.5, 5.7), dpi=300)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(2.25, 1.0),
        hspace=0.12,
    )
    ax = fig.add_subplot(grid[0], projection="3d")
    ax_rho = fig.add_subplot(grid[1])

    colors = mpl.colormaps["Dark2" if color_scheme == "extended-abstract" else "tab10"]
    highlight_frac = (
        _detect_ce_sign_fraction(mode_values.values()) or 0.2425287356321839
    )
    highlight_label = "C. elegans E/I\nEdge Balance"
    highlight_atol = 1e-12
    plotted_any = False
    highlight_labeled = False
    performance_values = []
    for idx, metric in enumerate(metrics):
        rows = []
        for mode in modes:
            frac = mode_values.get(mode, np.nan)
            y = mean_cv.get((mode, metric), np.nan)
            z = mean_perf.get((mode, metric), np.nan)
            if not (np.isfinite(frac) and np.isfinite(y) and np.isfinite(z)):
                continue
            rows.append((frac, y, z))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        if performance_scale != "linear" and np.any(zs < 0):
            raise ValueError(
                "Root-transformed performance scales require non-negative values."
            )
        performance_values.extend(zs.tolist())
        if performance_scale == "sqrt":
            zs_plot = np.sqrt(zs)
        elif performance_scale == "fourth_root":
            zs_plot = np.power(zs, 0.25)
        elif performance_scale == "sixth_root":
            zs_plot = np.power(zs, 1.0 / 6.0)
        else:
            zs_plot = zs
        ax.plot(
            xs,
            ys,
            zs_plot,
            marker="o",
            color=colors(idx % 10),
            linewidth=2.0,
            markersize=4.8,
            label=metric,
        )
        highlight_mask = np.isclose(xs, highlight_frac, rtol=0.0, atol=highlight_atol)
        if np.any(highlight_mask):
            ax.scatter(
                xs[highlight_mask],
                ys[highlight_mask],
                zs_plot[highlight_mask],
                color="black",
                s=42,
                depthshade=False,
                zorder=6,
                label=highlight_label if not highlight_labeled else "_nolegend_",
            )
            highlight_labeled = True
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[warn] plot_frac_cv_meanline: no finite data to plot.")
        return

    if all(str(m).startswith("sign_test") for m in modes):
        x_label = "Negative Sign %"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value"
    else:
        x_label = "mode value"

    # Use the full set of mode positions for x-limits.
    x_positions = [mode_values[m] for m in modes if np.isfinite(mode_values[m])]
    x_min, x_max = min(x_positions), max(x_positions)
    span = x_max - x_min
    pad = 0.05 * span if span > 0 else 1.0

    axis_x_label = "E/I Edge Balance" if x_label == "Negative Sign %" else x_label
    ax.set_xlabel(axis_x_label, fontsize=10.2, labelpad=0)
    ax.set_ylabel("Mean CV", fontsize=10.5, labelpad=-3)
    ax.set_zlabel("")
    if performance_scale in {"sqrt", "fourth_root", "sixth_root"}:
        z_min = float(np.min(performance_values))
        z_max = float(np.max(performance_values))
        exponent = {
            "sqrt": 0.5,
            "fourth_root": 0.25,
            "sixth_root": 1.0 / 6.0,
        }[performance_scale]
        transformed_min = np.power(z_min, exponent)
        transformed_max = np.power(z_max, exponent)
        transformed_pad = 0.06 * (transformed_max - transformed_min)
        ax.set_zlim(
            max(0.0, transformed_min - transformed_pad),
            transformed_max + transformed_pad,
        )
        tick_candidates = np.asarray(
            [1, 2, 5, 10, 20, 50, 100, 200, 500],
            dtype=float,
        )
        tick_values = tick_candidates[
            (tick_candidates >= z_min) & (tick_candidates <= z_max)
        ]
        if tick_values.size < 2:
            fallback_ticks = mpl.ticker.MaxNLocator(
                nbins=3,
                steps=[1, 2, 2.5, 5, 10],
                min_n_ticks=2,
            ).tick_values(z_min, z_max)
            tick_values = fallback_ticks[
                (fallback_ticks >= z_min) & (fallback_ticks <= z_max)
            ]
        ax.zaxis.set_major_locator(
            mpl.ticker.FixedLocator(np.power(tick_values, exponent))
        )
        ax.zaxis.set_major_formatter(
            mpl.ticker.FixedFormatter([f"{value:g}" for value in tick_values])
        )
        ax.zaxis.set_minor_locator(mpl.ticker.NullLocator())
    else:
        ax.zaxis.set_major_locator(
            mpl.ticker.MaxNLocator(
                nbins=4,
                min_n_ticks=2,
                integer=True,
                prune="both",
            )
        )
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=bins))
    ax.set_xlim(x_min - pad, x_max + pad)
    if axis_x_label == "E/I Edge Balance":
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.18, 1.0, 0.78), zoom=1.16)
    ax.xaxis.label.set_clip_on(False)
    ax.yaxis.label.set_clip_on(False)
    ax.zaxis.label.set_clip_on(False)
    _style_3d_axis(ax, tick_labelsize=8, tick_pad=-3, z_tick_pad=-2)
    ax.view_init(elev=20, azim=-18)
    fig.text(
        0.985,
        0.64,
        "Mean performance"
        + (
            " (square-root scale)"
            if performance_scale == "sqrt"
            else " (fourth-root scale)"
            if performance_scale == "fourth_root"
            else " (sixth-root scale)"
            if performance_scale == "sixth_root"
            else ""
        ),
        fontsize=9.8,
        rotation=90,
        ha="center",
        va="center",
    )
    handles, labels = ax.get_legend_handles_labels()
    reference_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label == highlight_label
    ]
    curve_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label != highlight_label
    ]
    handles, labels = zip(*(curve_entries + reference_entries))
    fig.legend(
        handles,
        labels,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="0.86",
        framealpha=1.0,
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(-0.055, 0.975),
        borderaxespad=0.0,
        handlelength=2.0,
        borderpad=0.42,
        labelspacing=0.5,
        markerscale=1.1,
    )

    raw_group_cols = [
        col for col in ("mode", "src", "shuffle_id") if col in combined.columns
    ]
    raw_rho = combined.loc[:, raw_group_cols + ["raw_rho"]].copy()
    raw_rho["sign_frac"] = raw_rho["mode"].map(mode_values)
    raw_rho["raw_rho"] = pd.to_numeric(raw_rho["raw_rho"], errors="coerce")
    raw_rho = raw_rho[
        raw_rho["mode"].isin(modes)
        & np.isfinite(raw_rho["sign_frac"])
        & np.isfinite(raw_rho["raw_rho"])
    ]
    # A reservoir's raw rho is repeated once per hyperparameter row. Average
    # within each source/shuffle first so incomplete grids cannot overweight a
    # particular reservoir in the sweep mean.
    raw_rho = raw_rho.groupby(raw_group_cols + ["sign_frac"], as_index=False)[
        "raw_rho"
    ].mean()
    rho_curve = (
        raw_rho.groupby("sign_frac", as_index=False)["raw_rho"]
        .mean()
        .sort_values("sign_frac")
    )
    if rho_curve.empty:
        plt.close(fig)
        print("[warn] plot_frac_cv_meanline: no finite raw_rho data to plot.")
        return

    dataset_key = os.path.basename(os.path.normpath(out_dir))
    rho_color = "#7A5195" if color_scheme == "extended-abstract" else "#d55e00"
    ax_rho.plot(
        rho_curve["sign_frac"],
        rho_curve["raw_rho"],
        color=rho_color,
        linewidth=2.2,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=rho_color,
        markeredgewidth=1.1,
        zorder=3,
    )

    cv_rows = []
    for mode in modes:
        metric_cvs = np.asarray(
            [mean_cv.get((mode, metric), np.nan) for metric in metrics],
            dtype=float,
        )
        metric_cvs = metric_cvs[np.isfinite(metric_cvs)]
        frac = mode_values.get(mode, np.nan)
        if np.isfinite(frac) and metric_cvs.size:
            cv_rows.append((frac, float(metric_cvs.mean())))
    cv_curve = pd.DataFrame(cv_rows, columns=["sign_frac", "mean_cv"])
    cv_curve = (
        cv_curve.groupby("sign_frac", as_index=False)["mean_cv"]
        .mean()
        .sort_values("sign_frac")
    )

    cv_color = "#00876C" if color_scheme == "extended-abstract" else "#374151"
    ax_cv = ax_rho.twinx()
    ax_cv.plot(
        cv_curve["sign_frac"],
        cv_curve["mean_cv"],
        color=cv_color,
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor=cv_color,
        markeredgewidth=0.9,
        zorder=4,
    )
    if dataset_key != "matched_er":
        ax_rho.axvline(
            highlight_frac,
            color="#444444",
            linestyle="--",
            linewidth=1.05,
            alpha=0.9,
            zorder=2,
        )
        ax_rho.text(
            highlight_frac + 0.012,
            0.52,
            "empirical E/I Edge Balance",
            color="#444444",
            fontsize=7.5,
            ha="left",
            va="center",
            transform=ax_rho.get_xaxis_transform(),
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.8,
            },
            zorder=6,
        )
    rho_min = float(rho_curve["raw_rho"].min())
    rho_max = float(rho_curve["raw_rho"].max())
    rho_span = rho_max - rho_min
    rho_pad = max(0.6, 0.09 * rho_span)
    ax_rho.set_xlim(x_min - pad, x_max + pad)
    if axis_x_label == "E/I Edge Balance":
        ax_rho.xaxis.set_major_formatter(
            mpl.ticker.PercentFormatter(xmax=1.0, decimals=0)
        )
    ax_rho.set_ylim(rho_min - rho_pad, rho_max + rho_pad)
    ax_rho.set_xlabel(axis_x_label, fontsize=9.4)
    ax_rho.set_ylabel(r"Mean raw $\rho(W)$", fontsize=9.4, color=rho_color)
    ax_rho.grid(
        True,
        axis="y",
        color="#dfdfdf",
        linewidth=0.65,
        alpha=0.7,
    )
    ax_rho.tick_params(axis="x", labelsize=8.2)
    ax_rho.tick_params(axis="y", labelsize=8.2, colors=rho_color)
    ax_rho.spines["right"].set_visible(False)
    for spine in ax_rho.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax_rho.spines["left"].set_color(rho_color)

    cv_min = float(cv_curve["mean_cv"].min())
    cv_max = float(cv_curve["mean_cv"].max())
    cv_span = cv_max - cv_min
    cv_pad = max(0.005, 0.10 * cv_span)
    ax_cv.set_ylim(cv_min - cv_pad, cv_max + cv_pad)
    ax_cv.set_ylabel("Mean CV", fontsize=9.4, color=cv_color)
    ax_cv.tick_params(axis="y", labelsize=8.2, colors=cv_color)
    ax_cv.spines["right"].set_color(cv_color)
    ax_cv.spines["right"].set_linewidth(0.8)
    ax_cv.spines["top"].set_visible(False)

    out_fig = _replace_path(os.path.join(out_dir, "meanpoint_frac_cv_lines.png"))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.08, top=0.96, hspace=0.12)
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
