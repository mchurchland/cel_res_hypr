"""Comparison for reservoir research figures."""

from cel_res_hypr.plotting.io import _read_compare_table
from cel_res_hypr.plotting.style import _short_thesis_name
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import warnings


def _mode_metric_means(comp_tbl: pd.DataFrame, value_col: str = "mean") -> pd.DataFrame:
    cols = ["mode", "metric", value_col]
    if "n_hparams" in comp_tbl.columns:
        cols.append("n_hparams")
    tbl = comp_tbl[cols].copy()
    if "n_hparams" not in tbl.columns:
        tbl["n_hparams"] = 1.0

    tbl = tbl.dropna(subset=["mode", "metric", value_col])
    tbl.loc[tbl["n_hparams"] <= 0, "n_hparams"] = np.nan
    tbl["weighted_sum"] = tbl[value_col] * tbl["n_hparams"]

    grp = tbl.groupby(["mode", "metric"], as_index=False, dropna=False)
    agg = grp.agg(
        weighted_sum=("weighted_sum", "sum"),
        weight_sum=("n_hparams", "sum"),
        simple_mean=(value_col, "mean"),
        n_groups=(value_col, "size"),
    )
    agg["mode_metric_mean"] = agg["weighted_sum"] / agg["weight_sum"]
    agg["mode_metric_mean"] = agg["mode_metric_mean"].where(
        np.isfinite(agg["mode_metric_mean"]),
        agg["simple_mean"],
    )
    return agg[["mode", "metric", "mode_metric_mean", "n_groups"]]


def compare_mode_metric_means(
    mean_csv_a: str,
    mean_csv_b: str,
    out_csv: str,
    label_a: str = "full",
    label_b: str = "frac025",
    value_col: str = "auto",
) -> pd.DataFrame:
    a_raw, col_a = _read_compare_table(mean_csv_a, value_col=value_col)
    b_raw, col_b = _read_compare_table(mean_csv_b, value_col=value_col)
    if col_a != col_b:
        raise ValueError(
            f"Comparison value columns do not match: A uses '{col_a}', B uses '{col_b}'. "
            "Set --compare-value-col explicitly."
        )
    print(f"[compare] using value column: {col_a}")

    a = _mode_metric_means(a_raw, value_col=col_a).rename(
        columns={
            "mode_metric_mean": f"mean_{label_a}",
            "n_groups": f"n_groups_{label_a}",
        }
    )
    b = _mode_metric_means(b_raw, value_col=col_b).rename(
        columns={
            "mode_metric_mean": f"mean_{label_b}",
            "n_groups": f"n_groups_{label_b}",
        }
    )

    out = a.merge(b, on=["mode", "metric"], how="outer")
    out["delta"] = out[f"mean_{label_b}"] - out[f"mean_{label_a}"]
    denom = out[f"mean_{label_a}"].abs()
    out["pct_delta"] = np.where(denom > 0, 100.0 * out["delta"] / denom, np.nan)
    out["drop_from_a_to_b"] = out[f"mean_{label_a}"] - out[f"mean_{label_b}"]
    out["pct_drop_from_a_to_b"] = np.where(
        denom > 0,
        100.0 * out["drop_from_a_to_b"] / denom,
        np.nan,
    )
    out = out.sort_values(["metric", "mode"], kind="stable").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} (rows={len(out)})")

    for metric in out["metric"].dropna().unique():
        sub = out[(out["metric"] == metric) & np.isfinite(out["delta"])].copy()
        if sub.empty:
            continue
        top = sub.iloc[sub["delta"].abs().argmax()]
        print(
            f"[compare] {metric}: max |delta| mode={top['mode']} "
            f"delta={top['delta']:.6g} ({label_b} - {label_a})"
        )
    return out


def _normalize_compare_metrics(spec: str):
    if not spec:
        return []
    aliases = {
        "cov": "cosine_similarity",
        "covariance": "cosine_similarity",
        "covariance_mean": "cosine_similarity",
        "cosine": "cosine_similarity",
        "kl": "kl_to_gaussian",
        "kl_gaussian": "kl_to_gaussian",
    }
    out = []
    for tok in str(spec).split(","):
        key = tok.strip()
        if not key:
            continue
        out.append(aliases.get(key.lower(), key))
    return out


def plot_mode_self_drop_comparison(
    comp_tbl: pd.DataFrame,
    out_png: str,
    label_a: str = "full",
    label_b: str = "frac025",
):
    req = {"mode", "metric"}
    if not req.issubset(comp_tbl.columns):
        print(f"[warn] comparison table missing columns for plotting: {sorted(req)}")
        return

    plot_value_col = (
        "pct_drop_from_a_to_b"
        if "pct_drop_from_a_to_b" in comp_tbl.columns
        else "drop_from_a_to_b"
    )
    dat = comp_tbl.dropna(subset=["mode", "metric", plot_value_col]).copy()
    if dat.empty:
        print("[warn] no overlapping mode/metric rows to plot self-drop comparison.")
        return

    metric_order = [
        "MC",
        "IPC",
        "KR",
        "GR",
        "kl_to_gaussian",
        "cosine_similarity",
        "wt_mean",
    ]
    metrics_present = list(dat["metric"].dropna().unique())
    metrics = [m for m in metric_order if m in metrics_present] + [
        m for m in metrics_present if m not in metric_order
    ]
    pivot = dat.pivot_table(
        index="mode",
        columns="metric",
        values=plot_value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=metrics)
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    vals = pivot.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig_w = max(8.0, 1.8 * len(metrics) + 3.0)
    fig_h = max(5.0, 0.45 * len(pivot.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.colormaps["coolwarm"].copy()
    cmap.set_bad(color="#e6e6e6")
    masked = np.ma.masked_invalid(vals)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mode")
    if plot_value_col == "pct_drop_from_a_to_b":
        ax.set_title(f"Per-mode self drop heatmap (%: {label_a} vs {label_b})")
    else:
        ax.set_title(f"Per-mode self drop heatmap ({label_a} - {label_b})")

    # Annotate cell values for fast lookup.
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(
                    j, i, "NA", ha="center", va="center", fontsize=8, color="#666666"
                )
                continue
            text_color = "white" if abs(v) > (0.55 * vmax) else "black"
            if plot_value_col == "pct_drop_from_a_to_b":
                label = f"{v:.1f}%"
            else:
                label = f"{v:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if plot_value_col == "pct_drop_from_a_to_b":
        cbar.set_label(f"Percent drop (%): ({label_a} - {label_b}) / |{label_a}|")
    else:
        cbar.set_label(f"Drop ({label_a} - {label_b})")
    fig.suptitle(
        f"Positive means {label_b} is lower than {label_a}", y=0.995, fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def _metric_bounds_from_reference(
    ref_tbl: pd.DataFrame, value_col: str, bound_frac: float
):
    bounds = {}
    for metric in sorted(ref_tbl["metric"].dropna().unique()):
        vals = (
            ref_tbl.loc[ref_tbl["metric"] == metric, value_col]
            .dropna()
            .to_numpy(dtype=float)
        )
        if vals.size == 0:
            continue
        med = float(np.nanmedian(vals))
        bounds[metric] = abs(med) * float(bound_frac)
    return bounds


def _pairwise_tost_results(
    tbl: pd.DataFrame,
    value_col: str,
    bounds_by_metric: dict,
    alpha: float = 0.05,
):
    out_rows = []
    metrics = sorted(tbl["metric"].dropna().unique())
    for metric in metrics:
        sub_metric = tbl[tbl["metric"] == metric]
        modes = sorted(sub_metric["mode"].dropna().unique())
        bound = float(bounds_by_metric.get(metric, np.nan))
        if not np.isfinite(bound):
            continue
        for mode_a, mode_b in itertools.combinations(modes, 2):
            vals_a = (
                sub_metric.loc[sub_metric["mode"] == mode_a, value_col]
                .dropna()
                .to_numpy(dtype=float)
            )
            vals_b = (
                sub_metric.loc[sub_metric["mode"] == mode_b, value_col]
                .dropna()
                .to_numpy(dtype=float)
            )
            if vals_a.size == 0 or vals_b.size == 0:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, module="pingouin"
                )
                tost = pg.tost(vals_a, vals_b, bound, paired=False)
            pval = float(tost["pval"].iloc[0])
            sig = bool(np.isfinite(pval) and (pval < alpha))
            out_rows.append(
                {
                    "metric": metric,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "bound_ref": bound,
                    "pval": pval,
                    "sig": sig,
                }
            )
    return pd.DataFrame(out_rows)


def print_tost_preservation_summary(
    ref_tbl: pd.DataFrame,
    new_tbl: pd.DataFrame,
    value_col: str,
    label_ref: str,
    label_new: str,
    alpha: float = 0.05,
    bound_frac: float = 0.05,
    metrics_filter=None,
    out_csv: str = "",
):
    ref = ref_tbl.copy()
    new = new_tbl.copy()
    ref[value_col] = pd.to_numeric(ref[value_col], errors="coerce")
    new[value_col] = pd.to_numeric(new[value_col], errors="coerce")
    ref = ref.dropna(subset=["mode", "metric", value_col])
    new = new.dropna(subset=["mode", "metric", value_col])

    common_modes = sorted(set(ref["mode"].unique()) & set(new["mode"].unique()))
    ref = ref[ref["mode"].isin(common_modes)].copy()
    new = new[new["mode"].isin(common_modes)].copy()
    common_metrics = sorted(set(ref["metric"].unique()) & set(new["metric"].unique()))
    if metrics_filter:
        common_metrics = [m for m in common_metrics if m in set(metrics_filter)]
    ref = ref[ref["metric"].isin(common_metrics)].copy()
    new = new[new["metric"].isin(common_metrics)].copy()

    bounds = _metric_bounds_from_reference(
        ref, value_col=value_col, bound_frac=bound_frac
    )
    ref_res = _pairwise_tost_results(
        ref, value_col=value_col, bounds_by_metric=bounds, alpha=alpha
    )
    new_res = _pairwise_tost_results(
        new, value_col=value_col, bounds_by_metric=bounds, alpha=alpha
    )

    if ref_res.empty or new_res.empty:
        print("[tost] no pairwise tests available for preservation summary.")
        return

    merged = ref_res.merge(
        new_res,
        on=["metric", "mode_a", "mode_b", "bound_ref"],
        how="outer",
        suffixes=(f"_{label_ref}", f"_{label_new}"),
    )
    sig_ref_col = f"sig_{label_ref}"
    sig_new_col = f"sig_{label_new}"
    p_ref_col = f"pval_{label_ref}"
    p_new_col = f"pval_{label_new}"
    merged[sig_ref_col] = merged[sig_ref_col].fillna(False).astype(bool)
    merged[sig_new_col] = merged[sig_new_col].fillna(False).astype(bool)
    merged["preserved_sig"] = merged[sig_ref_col] & merged[sig_new_col]

    base_sig = int(merged[sig_ref_col].sum())
    base_sig_testable = int(merged.loc[merged[p_new_col].notna(), sig_ref_col].sum())
    preserved = int(merged["preserved_sig"].sum())
    new_sig = int(merged[sig_new_col].sum())
    pct_all = (100.0 * preserved / base_sig) if base_sig > 0 else np.nan
    pct_testable = (
        (100.0 * preserved / base_sig_testable) if base_sig_testable > 0 else np.nan
    )

    print("\n=== TOST Significance Preservation (pairwise mode comparisons) ===")
    print(
        f"[tost] baseline significant in {label_ref}: {base_sig} "
        f"(alpha={alpha:.3g}, bound_frac={bound_frac:.3g})"
    )
    print(f"[tost] significant in {label_new}: {new_sig}")
    print(
        f"[tost] preserved baseline significances in {label_new}: "
        f"{preserved}/{base_sig} ({pct_all:.1f}%)"
        if np.isfinite(pct_all)
        else f"[tost] preserved baseline significances in {label_new}: {preserved}/{base_sig}"
    )
    print(
        f"[tost] testable-preservation (common tested pairs): "
        f"{preserved}/{base_sig_testable} ({pct_testable:.1f}%)"
        if np.isfinite(pct_testable)
        else f"[tost] testable-preservation (common tested pairs): {preserved}/{base_sig_testable}"
    )

    for metric in common_metrics:
        sub = merged[merged["metric"] == metric]
        b = int(sub[sig_ref_col].sum())
        p = int(sub["preserved_sig"].sum())
        pct = (100.0 * p / b) if b > 0 else np.nan
        if np.isfinite(pct):
            print(f"[tost] {metric}: preserved {p}/{b} ({pct:.1f}%)")
        else:
            print(f"[tost] {metric}: preserved {p}/{b}")
    print()

    # Show baseline-significant comparisons that are not preserved in new run.
    not_preserved = merged[(merged[sig_ref_col]) & (~merged[sig_new_col])].copy()
    if not not_preserved.empty:
        ref_mean_lookup = (
            ref.groupby(["metric", "mode"], dropna=False)[value_col].mean().to_dict()
        )
        new_mean_lookup = (
            new.groupby(["metric", "mode"], dropna=False)[value_col].mean().to_dict()
        )

        def _fmt_mean(v):
            return f"{v:.6g}" if np.isfinite(v) else "NA"

        not_preserved = not_preserved.sort_values(
            ["metric", "mode_a", "mode_b"], kind="stable"
        )
        print(
            "[tost] not preserved comparisons (baseline significant, new not significant):"
        )
        for _, row in not_preserved.iterrows():
            p_ref = row.get(p_ref_col, np.nan)
            p_new = row.get(p_new_col, np.nan)
            mode_a = row["mode_a"]
            mode_b = row["mode_b"]
            mode_a_label = _short_thesis_name(mode_a)
            mode_b_label = _short_thesis_name(mode_b)
            metric = row["metric"]
            ref_mean_a = ref_mean_lookup.get((metric, mode_a), np.nan)
            ref_mean_b = ref_mean_lookup.get((metric, mode_b), np.nan)
            new_mean_a = new_mean_lookup.get((metric, mode_a), np.nan)
            new_mean_b = new_mean_lookup.get((metric, mode_b), np.nan)
            if np.isfinite(p_new):
                print(
                    f"[tost] {metric}: {mode_a_label} vs {mode_b_label} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}={p_new:.4g}, bound={row['bound_ref']:.4g}, "
                    f"means_{label_ref}=[{mode_a_label}:{_fmt_mean(ref_mean_a)}, {mode_b_label}:{_fmt_mean(ref_mean_b)}], "
                    f"means_{label_new}=[{mode_a_label}:{_fmt_mean(new_mean_a)}, {mode_b_label}:{_fmt_mean(new_mean_b)}])"
                )
            else:
                print(
                    f"[tost] {metric}: {mode_a_label} vs {mode_b_label} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}=NA, bound={row['bound_ref']:.4g}, "
                    f"means_{label_ref}=[{mode_a_label}:{_fmt_mean(ref_mean_a)}, {mode_b_label}:{_fmt_mean(ref_mean_b)}], "
                    f"means_{label_new}=[{mode_a_label}:{_fmt_mean(new_mean_a)}, {mode_b_label}:{_fmt_mean(new_mean_b)}])"
                )
        print()
