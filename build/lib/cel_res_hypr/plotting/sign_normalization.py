"""Sign normalization for reservoir research figures."""

from cel_res_hypr.analysis.aggregation import (
    _assign_group_ids,
    _dispersion_cv as _dispersion,
)
from cel_res_hypr.paths import _replace_path
from cel_res_hypr.plotting.style import _mode_numeric_value
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


def _sign_norm_label(norm_name: str) -> str:
    labels = {
        "spectral_radius": r"$W / \rho(W)$",
        "original_radius": r"$W / \rho(W_{\mathrm{orig}})$",
    }
    return labels.get(str(norm_name), str(norm_name))


def _parse_sign_fraction_from_mode(mode: str, prefix: str) -> float:
    mode_base = str(mode).split("__norm_", 1)[0]
    prefix = str(prefix or "").strip()
    if prefix:
        m = re.search(
            rf"{re.escape(prefix)}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            mode_base,
        )
        if m:
            return float(m.group(1))
        return np.nan
    return _mode_numeric_value(mode_base)


def _infer_normalization_from_mode(mode: str) -> str:
    m = re.search(r"__norm_([A-Za-z0-9_+-]+)$", str(mode))
    if m:
        return m.group(1)
    return "spectral_radius"


def _prepare_sign_norm_ablation_df(combined: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = combined.copy()
    if "normalization" not in df.columns:
        df["normalization"] = ""
    df["normalization"] = df["normalization"].astype(str).str.strip()
    inferred_norm = df["mode"].astype(str).map(_infer_normalization_from_mode)
    df.loc[df["normalization"].isin(["", "nan", "None"]), "normalization"] = (
        inferred_norm
    )

    mode_base = df["mode"].astype(str).str.split("__norm_", n=1).str[0]
    if prefix:
        df = df[mode_base.str.startswith(str(prefix))].copy()
    if df.empty:
        return df

    df["sign_frac"] = df["mode"].map(
        lambda m: _parse_sign_fraction_from_mode(m, prefix)
    )
    df = df[np.isfinite(df["sign_frac"])].copy()
    for col in (
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
        "MC",
        "IPC",
        "KR",
        "GR",
        "raw_rho",
        "ref_rho",
        "post_rho",
        "scale_factor",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_group_values(
    per_group: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    if per_group.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "std", "sem", "n_groups"])
    summary = (
        per_group.groupby(group_cols, as_index=False)[value_col]
        .agg(mean="mean", std="std", n_groups="count")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n_groups"].clip(lower=1))
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def _save_sign_norm_metric_grid(
    summary: pd.DataFrame,
    out_path: str,
    *,
    value_label: str,
    title: str,
    metric_order: list[str],
    norm_order: list[str],
):
    if summary.empty:
        print(f"[warn] {title}: no summary rows to plot.")
        return
    metrics = [m for m in metric_order if m in set(summary["metric"])]
    if not metrics:
        print(f"[warn] {title}: no requested metrics present.")
        return

    summary = summary[summary["normalization"].astype(str).isin(norm_order)].copy()
    if summary.empty:
        print(f"[warn] {title}: no rows for requested normalization modes.")
        return

    colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300, squeeze=False)
    axes_flat = axes.ravel()
    handles = []
    labels = []

    for ax_idx, metric in enumerate(metrics[:4]):
        ax = axes_flat[ax_idx]
        plotted = False
        for norm_name in norm_order:
            sub = summary[
                (summary["metric"] == metric)
                & (summary["normalization"].astype(str) == norm_name)
            ].sort_values("sign_frac")
            if sub.empty:
                continue
            xs = sub["sign_frac"].to_numpy(float)
            ys = sub["mean"].to_numpy(float)
            sem = sub["sem"].to_numpy(float)
            line = ax.errorbar(
                xs,
                ys,
                yerr=sem,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                color=colors.get(norm_name),
                label=_sign_norm_label(norm_name),
            )
            if ax_idx == 0:
                handles.append(line)
                labels.append(_sign_norm_label(norm_name))
            plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        ax.set_title(metric)
        ax.set_xlabel("E/I Edge Balance")
        ax.set_ylabel(value_label)
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(title, y=0.985, fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(2, len(handles)),
            frameon=False,
        )
    fig.align_ylabels(axes_flat)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.89])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _save_sign_norm_scaling_grid(
    summary: pd.DataFrame,
    out_path: str,
    *,
    norm_order: list[str],
):
    if summary.empty:
        print("[warn] sign normalization scaling: no summary rows to plot.")
        return
    diag_cols = [
        ("post_rho", "post spectral radius"),
        ("ref_rho", "reference spectral radius"),
        ("raw_rho", "raw spectral radius"),
        ("scale_factor", "scale factor"),
    ]
    summary = summary[summary["normalization"].astype(str).isin(norm_order)].copy()
    if summary.empty:
        print(
            "[warn] sign normalization scaling: no rows for requested normalization modes."
        )
        return

    colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300, squeeze=False)
    axes_flat = axes.ravel()
    handles = []
    labels = []

    for ax_idx, (col, label) in enumerate(diag_cols):
        ax = axes_flat[ax_idx]
        if col not in summary.columns:
            ax.set_axis_off()
            continue
        plotted = False
        for norm_name in norm_order:
            sub = summary[
                summary["normalization"].astype(str) == norm_name
            ].sort_values("sign_frac")
            if sub.empty:
                continue
            xs = sub["sign_frac"].to_numpy(float)
            ys = sub[col].to_numpy(float)
            sem_col = f"{col}_sem"
            yerr = sub[sem_col].to_numpy(float) if sem_col in sub.columns else None
            line = ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                color=colors.get(norm_name),
                label=_sign_norm_label(norm_name),
            )
            if ax_idx == 0:
                handles.append(line)
                labels.append(_sign_norm_label(norm_name))
            plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        ax.set_title(label)
        ax.set_xlabel("E/I Edge Balance")
        ax.set_ylabel(label)
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    fig.suptitle("Sign-balance normalization scaling diagnostics", y=0.985, fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(2, len(handles)),
            frameon=False,
        )
    fig.align_ylabels(axes_flat)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.89])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _save_sign_norm_combined_grid(
    performance: pd.DataFrame,
    cv: pd.DataFrame,
    scaling: pd.DataFrame,
    out_path: str,
    *,
    metric_order: list[str],
    norm_order: list[str],
):
    """Save performance, CV, and radius diagnostics as one compact figure."""
    norm_colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    norm_styles = {
        "spectral_radius": "-",
        "original_radius": "--",
    }
    norm_markers = {
        "spectral_radius": "o",
        "original_radius": "s",
    }
    metric_colors = {
        "MC": "#1f77b4",
        "IPC": "#d95f02",
        "KR": "#2ca02c",
        "GR": "#9467bd",
    }
    metrics = [
        metric for metric in metric_order if metric in set(performance["metric"])
    ]
    if not metrics:
        print("[warn] combined sign normalization figure: no metrics to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.4), dpi=300, squeeze=False)
    memory_performance_ax, kernel_performance_ax, rho_ax = axes[0]
    memory_cv_ax, kernel_cv_ax, scale_ax = axes[1]

    def draw_combined_metrics(
        summary: pd.DataFrame,
        ax: plt.Axes,
        selected_metrics: list[str],
        ylabel: str,
    ):
        for metric in selected_metrics:
            for norm_name in norm_order:
                sub = summary[
                    (summary["metric"] == metric)
                    & (summary["normalization"].astype(str) == norm_name)
                ].sort_values("sign_frac")
                if sub.empty:
                    continue
                xs = sub["sign_frac"].to_numpy(float)
                ys = sub["mean"].to_numpy(float)
                sem = sub["sem"].to_numpy(float)
                ax.plot(
                    xs,
                    ys,
                    color=metric_colors[metric],
                    linestyle=norm_styles[norm_name],
                    marker=norm_markers[norm_name],
                    linewidth=1.8,
                    markersize=3.6,
                )
                ax.fill_between(
                    xs,
                    ys - sem,
                    ys + sem,
                    color=metric_colors[metric],
                    alpha=0.06,
                    linewidth=0,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    memory_metrics = [metric for metric in ("MC", "IPC") if metric in metrics]
    kernel_metrics = [metric for metric in ("KR", "GR") if metric in metrics]
    cv_target_label = "Mean CV"
    draw_combined_metrics(
        performance,
        memory_performance_ax,
        memory_metrics,
        "Mean performance",
    )
    memory_performance_ax.set_title("Memory metrics: mean performance")
    draw_combined_metrics(
        performance,
        kernel_performance_ax,
        kernel_metrics,
        "Mean performance",
    )
    kernel_performance_ax.set_title("Kernel metrics: mean performance")

    draw_combined_metrics(cv, memory_cv_ax, memory_metrics, cv_target_label)
    memory_cv_ax.set_title("Memory metrics: Mean CV")
    draw_combined_metrics(cv, kernel_cv_ax, kernel_metrics, cv_target_label)
    kernel_cv_ax.set_title("Kernel metrics: Mean CV")

    if scaling.empty:
        rho_ax.set_axis_off()
        scale_ax.set_axis_off()
    else:
        first_norm = norm_order[0]
        base = scaling[scaling["normalization"].astype(str) == first_norm].sort_values(
            "sign_frac"
        )
        for value_col, label, color, linestyle in (
            ("raw_rho", r"$\rho(W)$", "#666666", "-."),
            ("ref_rho", r"$\rho(W_{\mathrm{orig}})$", "#111111", ":"),
        ):
            if value_col in base.columns and not base.empty:
                rho_ax.plot(
                    base["sign_frac"].to_numpy(float),
                    base[value_col].to_numpy(float),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=label,
                )
        for norm_name in norm_order:
            sub = scaling[
                scaling["normalization"].astype(str) == norm_name
            ].sort_values("sign_frac")
            if sub.empty:
                continue
            rho_ax.plot(
                sub["sign_frac"].to_numpy(float),
                sub["post_rho"].to_numpy(float),
                color=norm_colors[norm_name],
                linestyle=norm_styles[norm_name],
                marker=norm_markers[norm_name],
                linewidth=1.8,
                markersize=3.6,
                label=rf"post $\rho$: {_sign_norm_label(norm_name)}",
            )
            scale_ax.plot(
                sub["sign_frac"].to_numpy(float),
                sub["scale_factor"].to_numpy(float),
                color=norm_colors[norm_name],
                linestyle=norm_styles[norm_name],
                marker=norm_markers[norm_name],
                linewidth=1.8,
                markersize=3.6,
                label=_sign_norm_label(norm_name),
            )
        rho_ax.set_title("Spectral-radius diagnostics")
        rho_ax.set_ylabel("Spectral radius")
        rho_ax.legend(frameon=False, fontsize=8, ncol=2)
        scale_ax.set_title("Normalization scale factor")
        scale_ax.set_ylabel("Scale factor")

    for ax in axes.ravel():
        if not ax.axison:
            continue
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.set_xlabel("E/I Edge Balance")
        ax.grid(True, alpha=0.22)

    fig.suptitle("Sign-balance normalization ablation", y=0.985, fontsize=17)
    metric_handles = [
        Line2D([0], [0], color=metric_colors[metric], linewidth=2.2, label=metric)
        for metric in metrics
    ]
    norm_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=norm_styles[norm_name],
            marker=norm_markers[norm_name],
            linewidth=1.8,
            markersize=4,
            label=_sign_norm_label(norm_name),
        )
        for norm_name in norm_order
    ]
    fig.legend(
        metric_handles + norm_handles,
        [handle.get_label() for handle in metric_handles + norm_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=len(metric_handles + norm_handles),
        frameon=False,
    )
    fig.tight_layout(rect=[0.02, 0.025, 1.0, 0.90])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_sign_norm_ablation(
    combined: pd.DataFrame,
    out_dir: str,
    prefix: str = "sign_test_og_cel",
    cv_rho_target: float | None = None,
    cv_include_rho: bool = False,
):
    if cv_include_rho and cv_rho_target is not None:
        raise ValueError(
            "Choose either full-grid CV including rho or a single rho target, not both."
        )
    os.makedirs(out_dir, exist_ok=True)
    df = _prepare_sign_norm_ablation_df(combined, prefix)
    if df.empty:
        available = sorted(combined["mode"].astype(str).dropna().unique().tolist())
        preview = ", ".join(available[:16]) + (" ..." if len(available) > 16 else "")
        print(f"[warn] sign_norm_ablation: no rows matched prefix={prefix!r}.")
        print(f"[info] available modes: {preview}")
        return

    metrics = [m for m in ("MC", "IPC", "KR", "GR") if m in df.columns]
    if not metrics:
        print("[warn] sign_norm_ablation: no MC/IPC/KR/GR columns found.")
        return

    plot_norms = ("spectral_radius", "original_radius")
    norm_order = [n for n in plot_norms if n in set(df["normalization"].astype(str))]
    if not norm_order:
        available = sorted(df["normalization"].astype(str).unique())
        print(
            "[warn] sign_norm_ablation: none of the requested plotted normalization modes "
            f"are present. Available: {available}"
        )
        return
    df = df[df["normalization"].astype(str).isin(norm_order)].copy()

    df = _assign_group_ids(df)
    df["unit_id"] = df["src"].astype(str) + ":" + df["group_id"].astype(str)

    metric_long = df.melt(
        id_vars=["normalization", "sign_frac", "unit_id"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    per_group_perf = metric_long.groupby(
        ["normalization", "sign_frac", "unit_id", "metric"], as_index=False
    )["value"].mean()
    perf_summary = _summarize_group_values(
        per_group_perf,
        "value",
        ["normalization", "sign_frac", "metric"],
    )
    perf_csv = _replace_path(
        os.path.join(out_dir, "sign_norm_ablation_mean_performance.csv")
    )
    perf_summary.to_csv(perf_csv, index=False)
    print(f"[saved] {perf_csv} (rows={len(perf_summary)})")

    hparam_keys = [
        "normalization",
        "sign_frac",
        "unit_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
    ]
    hparam_agg = (
        df.groupby(hparam_keys, as_index=False)[metrics]
        .mean()
        .sort_values(hparam_keys)
        .reset_index(drop=True)
    )
    hparam_long = hparam_agg.melt(
        id_vars=["normalization", "sign_frac", "unit_id", "rho_target"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    cv_input = hparam_long
    if cv_rho_target is not None:
        available_targets = np.sort(
            cv_input["rho_target"].dropna().unique().astype(float)
        )
        target_mask = np.isclose(
            cv_input["rho_target"].to_numpy(float),
            float(cv_rho_target),
            rtol=0.0,
            atol=1e-9,
        )
        cv_input = cv_input.loc[target_mask].copy()
        if cv_input.empty:
            available = ", ".join(f"{target:g}" for target in available_targets)
            raise ValueError(
                f"Requested sign-normalization CV target {cv_rho_target:g} is absent. "
                f"Available targets: {available}"
            )

    if cv_include_rho:
        # Full-grid invariance: spectral radius is one of the hyperparameter
        # dimensions contributing to the dispersion within each repeat.
        per_group_cv = cv_input.groupby(
            ["normalization", "sign_frac", "unit_id", "metric"],
            as_index=False,
        )["value"].agg(cv=lambda x: _dispersion(x.to_numpy()))
    else:
        per_target_cv = cv_input.groupby(
            ["normalization", "sign_frac", "unit_id", "rho_target", "metric"],
            as_index=False,
        )["value"].agg(cv=lambda x: _dispersion(x.to_numpy()))
        # Keep target radius out of the CV itself: calculate invariance across
        # the remaining hyperparameters at each selected target. If no target
        # was selected explicitly, average target-specific CVs per repeat.
        per_group_cv = per_target_cv.groupby(
            ["normalization", "sign_frac", "unit_id", "metric"],
            as_index=False,
        )["cv"].mean()

    cv_summary = _summarize_group_values(
        per_group_cv,
        "cv",
        ["normalization", "sign_frac", "metric"],
    )
    if cv_include_rho:
        cv_summary["rho_target"] = np.nan
        cv_summary["n_rho_targets"] = int(cv_input["rho_target"].nunique())
        cv_summary["cv_definition"] = "full_grid_cv_including_rho"
    elif cv_rho_target is None:
        cv_summary["rho_target"] = np.nan
        cv_summary["n_rho_targets"] = int(cv_input["rho_target"].nunique())
        cv_summary["cv_definition"] = "mean_within_target_cv"
    else:
        cv_summary["rho_target"] = float(cv_rho_target)
        cv_summary["n_rho_targets"] = 1
        cv_summary["cv_definition"] = "within_selected_target_cv"
    cv_csv = _replace_path(os.path.join(out_dir, "sign_norm_ablation_cv.csv"))
    cv_summary.to_csv(cv_csv, index=False)
    print(f"[saved] {cv_csv} (rows={len(cv_summary)})")

    diag_cols = [
        c for c in ("raw_rho", "ref_rho", "post_rho", "scale_factor") if c in df.columns
    ]
    scaling_summary = pd.DataFrame()
    if diag_cols:
        per_group_diag = df.groupby(
            ["normalization", "sign_frac", "unit_id"], as_index=False
        )[diag_cols].mean()
        parts = []
        for col in diag_cols:
            sub = per_group_diag.groupby(
                ["normalization", "sign_frac"], as_index=False
            )[col].agg(**{col: "mean", f"{col}_std": "std", f"{col}_n": "count"})
            sub[f"{col}_sem"] = sub[f"{col}_std"] / np.sqrt(
                sub[f"{col}_n"].clip(lower=1)
            )
            parts.append(sub)
        scaling_summary = parts[0]
        for sub in parts[1:]:
            scaling_summary = scaling_summary.merge(
                sub, on=["normalization", "sign_frac"], how="outer"
            )
        scaling_summary = scaling_summary.sort_values(
            ["normalization", "sign_frac"]
        ).reset_index(drop=True)
        fill_zero = {
            c: 0.0 for c in scaling_summary.columns if c.endswith(("_std", "_sem"))
        }
        scaling_summary = scaling_summary.fillna(fill_zero)
        scaling_csv = _replace_path(
            os.path.join(out_dir, "sign_norm_ablation_scaling.csv")
        )
        scaling_summary.to_csv(scaling_csv, index=False)
        print(f"[saved] {scaling_csv} (rows={len(scaling_summary)})")

    _save_sign_norm_combined_grid(
        perf_summary,
        cv_summary,
        scaling_summary,
        os.path.join(out_dir, "sign_norm_ablation_combined.png"),
        metric_order=metrics,
        norm_order=norm_order,
    )
