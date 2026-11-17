"""Tables for reservoir research figures."""

from cel_res_hypr.paths import _safe_path
from cel_res_hypr.plotting.style import _short_thesis_name
from scipy.stats import t as student_t
import numpy as np
import os
import pandas as pd
import re


def _write_weight_metric_difference_table(
    y_lookup,
    y_label: str,
    out_dir: str,
    tex_name: str,
    mode_vals,
    metrics,
    zero_mode=None,
    overlay_lookup=None,
    log_prefix: str = "weight_metric_difference_table",
    mode_value_symbol: str = r"\alpha",
):
    baseline_info = {}
    for metric in metrics:
        baseline_value = np.nan
        baseline_mode = ""
        baseline_source = ""

        if overlay_lookup is not None:
            baseline_value = overlay_lookup.get(("real", metric), np.nan)
            if np.isfinite(baseline_value):
                baseline_mode = "real"
                baseline_source = "overlay"

        if zero_mode is not None:
            zero_value = y_lookup.get((zero_mode, metric), np.nan)
            if np.isfinite(zero_value):
                baseline_value = zero_value
                baseline_mode = zero_mode
                baseline_source = "zero-noise mode"

        if not np.isfinite(baseline_value):
            for mode, _ in mode_vals:
                cand = y_lookup.get((mode, metric), np.nan)
                if np.isfinite(cand):
                    baseline_value = cand
                    baseline_mode = str(mode)
                    baseline_source = "first finite mode"
                    break

        if np.isfinite(baseline_value):
            baseline_info[metric] = {
                "baseline_value": float(baseline_value),
                "baseline_mode": baseline_mode,
                "baseline_source": baseline_source,
            }

    if not baseline_info:
        return

    def _latex_escape(text: str) -> str:
        out = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "_": r"\_",
            "#": r"\#",
            "$": r"\$",
            "{": r"\{",
            "}": r"\}",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        return out

    def _fmt_num(x: float, digits: int = 3) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{x:.{digits}g}"

    def _fmt_signed(x: float, digits: int = 3) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{x:+.{digits}g}"

    def _fmt_mode_value(raw_value: float) -> str:
        if not np.isfinite(raw_value):
            return "NA"
        if abs(raw_value - round(raw_value)) < 1e-12:
            return str(int(round(raw_value)))
        return f"{raw_value:.3g}"

    data_modes = []
    for mode, raw_value in mode_vals:
        if mode == zero_mode:
            continue
        if any(np.isfinite(y_lookup.get((mode, metric), np.nan)) for metric in metrics):
            data_modes.append((mode, raw_value))

    lines = [
        "\\begin{tabular}{" + ("l" + "c" * (1 + len(data_modes))) + "}",
        "\\toprule",
    ]

    header_cells = ["Metric", "Baseline"]
    for _, raw_value in data_modes:
        header_cells.append(f"${mode_value_symbol}={_fmt_mode_value(raw_value)}$")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append("\\midrule")

    for metric in metrics:
        info = baseline_info.get(metric)
        if info is None:
            continue
        baseline_value = info["baseline_value"]
        denom = abs(baseline_value)
        row_cells = [_latex_escape(metric), _fmt_num(baseline_value, digits=4)]
        for mode, _ in data_modes:
            value = y_lookup.get((mode, metric), np.nan)
            if not np.isfinite(value):
                row_cells.append("--")
                continue
            delta = float(value - baseline_value)
            if denom > 1e-12:
                pct_delta = 100.0 * delta / denom
                cell = f"{_fmt_signed(delta)} ({_fmt_signed(pct_delta)}\\%)"
            else:
                cell = _fmt_signed(delta)
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    latex_table = "\n".join(lines)

    out_tex = _safe_path(os.path.join(out_dir, tex_name))
    with open(out_tex, "w", encoding="utf-8") as handle:
        handle.write(latex_table + "\n")

    print(f"[latex] {log_prefix}: {y_label}")
    print(latex_table)
    print(f"[saved] {out_tex}")


def _mean_difference_ci(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Student-t confidence interval for a one-sample mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=values.size - 1))
    return mean - critical * sem, mean + critical * sem


def _latex_escape_table_text(value: str) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def _cv_difference_latex_table(
    table: pd.DataFrame,
    metrics: tuple[str, ...],
    baseline_label: str,
) -> str:
    sub = table[table["metric"].isin(metrics)].copy()
    escaped_baseline = _latex_escape_table_text(baseline_label)
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        f"Metric & Control & {escaped_baseline} mean CV & Control mean CV & "
        r"$\Delta$CV & $\Delta$CV (\%) & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        comparison = _latex_escape_table_text(row["control_label"])
        ci = f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
        lines.append(
            f"{_latex_escape_table_text(row['metric'])} & {comparison} & "
            f"{row['baseline_mean_cv']:.4f} & {row['control_mean_cv']:.4f} & "
            f"{row['delta_cv']:+.4f} & {row['pct_delta_cv']:+.2f} & {ci} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% Differences are control minus {_latex_escape_table_text(baseline_label)}; paired by trial where available.",
        ]
    )
    return "\n".join(lines)


def _resolve_cv_baseline_mode(modes: set[str], requested: str) -> str | None:
    """Resolve an explicit or experiment-appropriate CV baseline."""
    requested = str(requested or "auto").strip()
    if requested.lower() != "auto":
        if requested.lower() in {"pm1", "pm1_base", "local_pm1"}:
            return "local_sign+binary" if "local_sign+binary" in modes else None
        if requested in modes:
            return requested
        if requested == "real" and "CE-real" in modes:
            return "CE-real"
        return None

    for candidate in ("real", "CE-real"):
        if candidate in modes:
            return candidate
    if "local_sign+binary" in modes and (
        "global_sign_pres" in modes
        or "binary+shuffle" in modes
        or "binary+conshuffle+wshuffle" in modes
    ):
        return "local_sign+binary"
    if {"binary_base", "binary_base_topology_shuffle"}.issubset(modes):
        return "binary_base"
    if "binary_base" in modes:
        return "binary_base"
    return None


def print_cv_difference_tables(
    disp: pd.DataFrame,
    out_dir: str,
    baseline_mode: str = "auto",
) -> pd.DataFrame:
    """Compare each architecture's mean CV with the experiment-appropriate baseline."""
    required = {"mode", "metric", "dispersion"}
    missing = required.difference(disp.columns)
    if missing:
        print(f"[warn] CV-difference table missing columns: {sorted(missing)}")
        return pd.DataFrame()

    modes = set(disp["mode"].dropna().astype(str))
    requested_baseline = baseline_mode
    baseline_mode = _resolve_cv_baseline_mode(modes, requested_baseline)
    if baseline_mode is None:
        print(
            f"[warn] CV-difference table could not resolve baseline {requested_baseline!r}. "
            f"Available modes: {sorted(modes)}"
        )
        return pd.DataFrame()
    print(
        f"[info] CV-difference baseline: {_short_thesis_name(baseline_mode)} ({baseline_mode})"
    )

    metrics = [
        metric
        for metric in ("MC", "IPC", "KR", "GR")
        if metric in set(disp["metric"].dropna())
    ]
    pairing_keys = [key for key in ("src", "group_id") if key in disp.columns]
    rows: list[dict] = []

    for metric in metrics:
        metric_df = disp[disp["metric"] == metric].copy()
        baseline = metric_df[metric_df["mode"] == baseline_mode].dropna(
            subset=["dispersion"]
        )
        if baseline.empty:
            continue
        for control_mode in sorted(set(metric_df["mode"].dropna()) - {baseline_mode}):
            control = metric_df[metric_df["mode"] == control_mode].dropna(
                subset=["dispersion"]
            )
            if control.empty:
                continue

            paired = pd.DataFrame()
            if pairing_keys:
                base_pair = baseline[pairing_keys + ["dispersion"]].rename(
                    columns={"dispersion": "baseline_cv"}
                )
                control_pair = control[pairing_keys + ["dispersion"]].rename(
                    columns={"dispersion": "control_cv"}
                )
                paired = base_pair.merge(control_pair, on=pairing_keys, how="inner")

            if len(paired) >= 2:
                baseline_values = paired["baseline_cv"].to_numpy(float)
                control_values = paired["control_cv"].to_numpy(float)
                differences = control_values - baseline_values
                comparison_type = "paired"
                n_comparisons = int(len(paired))
                ci_low, ci_high = _mean_difference_ci(differences)
            else:
                baseline_values = baseline["dispersion"].to_numpy(float)
                control_values = control["dispersion"].to_numpy(float)
                delta = float(np.mean(control_values) - np.mean(baseline_values))
                se = (
                    np.sqrt(
                        np.var(control_values, ddof=1) / len(control_values)
                        + np.var(baseline_values, ddof=1) / len(baseline_values)
                    )
                    if len(control_values) >= 2 and len(baseline_values) >= 2
                    else np.nan
                )
                ci_low = delta - 1.96 * se if np.isfinite(se) else np.nan
                ci_high = delta + 1.96 * se if np.isfinite(se) else np.nan
                differences = np.array([delta], dtype=float)
                comparison_type = "unpaired"
                n_comparisons = int(min(len(baseline_values), len(control_values)))

            baseline_mean = float(np.mean(baseline_values))
            control_mean = float(np.mean(control_values))
            delta_cv = float(control_mean - baseline_mean)
            pct_delta = (
                100.0 * delta_cv / abs(baseline_mean)
                if abs(baseline_mean) > 1e-12
                else np.nan
            )
            rows.append(
                {
                    "metric": metric,
                    "baseline_mode": baseline_mode,
                    "baseline_label": _short_thesis_name(baseline_mode),
                    "control_mode": control_mode,
                    "control_label": _short_thesis_name(control_mode),
                    "baseline_mean_cv": baseline_mean,
                    "control_mean_cv": control_mean,
                    "delta_cv": delta_cv,
                    "pct_delta_cv": pct_delta,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "comparison_type": comparison_type,
                    "n": n_comparisons,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        print("[warn] no CV differences could be computed.")
        return table
    metric_order = {metric: idx for idx, metric in enumerate(("MC", "IPC", "KR", "GR"))}
    table["_metric_order"] = table["metric"].map(metric_order)
    table = (
        table.sort_values(["_metric_order", "control_label"], kind="stable")
        .drop(columns="_metric_order")
        .reset_index(drop=True)
    )

    os.makedirs(out_dir, exist_ok=True)
    baseline_is_ce = baseline_mode in {"real", "CE-real"}
    baseline_tag = (
        "ce" if baseline_is_ce else re.sub(r"[^A-Za-z0-9._-]+", "_", baseline_mode)
    )
    baseline_label = _short_thesis_name(baseline_mode)
    latex = _cv_difference_latex_table(table, tuple(metrics), baseline_label)
    tex_name = (
        "cv_mean_differences_table.tex"
        if baseline_is_ce
        else f"cv_mean_differences_vs_{baseline_tag}_table.tex"
    )
    out_tex = _safe_path(os.path.join(out_dir, tex_name))
    with open(out_tex, "w", encoding="utf-8") as handle:
        handle.write(latex + "\n")
    print(f"\n=== Mean CV differences vs {baseline_label} ===")
    print(latex)
    print(f"[saved] {out_tex}")
    print()
    return table
