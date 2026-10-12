#!/usr/bin/env python3
"""Summarize high-to-low raw-radius contrasts within each control family.

The E/I sweeps use both 0%-to-40% and 100%-to-60% negative-edge branches;
shuffle controls use the highest and lowest condition-mean raw radii within
each plotted family. Each seed contributes one trial summary over 96
hyperparameter settings; seed IDs are paired across endpoints.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_NAME = "combined.ALL.GRKR_erank.rank_updated.csv"
METRICS = ("IPC", "KR", "MC", "GR")
OUTCOMES = ("raw_rho",) + tuple(
    f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "cv")
)
SIGN_FAMILIES = (
    ("CE matched", "cel_matched", "sign_test_og_cel"),
    ("CE known edges", "cel_removed", "sign_test_og_cel"),
    ("ER matched", "matched_er", "sign_test_er"),
)
SIGN_CONTRASTS = (
    ("0_to_40", 0.0, 0.4),
    ("100_to_60", 1.0, 0.6),
)
SHUFFLE_FAMILIES = (
    ("CE shuffles", ("real", "shuffle", "celW+connShuf", "conn_shuf_only")),
    (
        "PM1 controls",
        (
            "local_sign+binary",
            "binary+shuffle",
            "global_sign_pres",
            "binary+conshuffle+wshuffle",
        ),
    ),
    ("Binary controls", ("binary_base", "binary_base_topology_shuffle")),
)
MODE_LABELS = {
    "real": "empirical",
    "shuffle": "weight shuf.",
    "celW+connShuf": "conn.+wt. shuf.",
    "conn_shuf_only": "conn. shuf.",
    "local_sign+binary": "PM1",
    "binary+shuffle": "PM1+conn. shuf.",
    "global_sign_pres": "PM1+wt. shuf.",
    "binary+conshuffle+wshuffle": "PM1+conn.+wt. shuf.",
    "binary_base": "binary",
    "binary_base_topology_shuffle": "binary shuf.",
}


def read_trials(path: Path) -> pd.DataFrame:
    columns = ["mode", "seed", "raw_rho", *METRICS]
    data = pd.read_csv(path, usecols=columns)
    if data[["raw_rho", *METRICS]].isna().any().any():
        raise ValueError(f"Missing raw radius or metric values in {path}")
    grouped = data.groupby(["mode", "seed"], sort=True)
    if not grouped.size().eq(96).all():
        raise ValueError(f"Expected 96 hyperparameter settings per trial in {path}")
    if not grouped["raw_rho"].nunique().eq(1).all():
        raise ValueError(f"Raw radius varies within a trial in {path}")

    means = grouped[list(METRICS)].mean()
    cvs = grouped[list(METRICS)].std(ddof=0).div(means)
    result = grouped["raw_rho"].first().to_frame()
    for metric in METRICS:
        result[f"{metric}_mean"] = means[metric]
        result[f"{metric}_cv"] = cvs[metric]
    if not np.isfinite(result[list(OUTCOMES)].to_numpy()).all():
        raise ValueError(f"Non-finite trial summary in {path}")
    return result


def sign_fraction(mode: str, prefix: str) -> float:
    match = re.fullmatch(re.escape(prefix) + r"(\d+(?:\.\d+)?)", mode)
    if match is None:
        raise ValueError(f"Unexpected sign-sweep mode: {mode}")
    return float(match.group(1))


def endpoint_modes(trials: pd.DataFrame, modes: list[str]) -> tuple[str, str]:
    rho_by_mode = trials.groupby(level="mode")["raw_rho"].mean().loc[modes]
    return str(rho_by_mode.idxmax()), str(rho_by_mode.idxmin())


def compare(
    trials: pd.DataFrame,
    family: str,
    contrast_id: str,
    high_mode: str,
    low_mode: str,
    n_boot: int,
    rng: np.random.Generator,
    labels: dict[str, str],
) -> list[dict[str, object]]:
    high = trials.xs(high_mode, level="mode").sort_index()
    low = trials.xs(low_mode, level="mode").sort_index()
    if not high.index.equals(low.index):
        raise ValueError(f"Trial seeds do not match for {family}")
    if len(high) != 200:
        raise ValueError(f"Expected 200 paired trials for {family}, got {len(high)}")
    if high["raw_rho"].mean() <= low["raw_rho"].mean():
        raise ValueError(f"The specified high-to-low contrast is reversed for {family}")

    indices = rng.integers(0, len(high), size=(n_boot, len(high)))
    rows = []
    for outcome in OUTCOMES:
        a = high[outcome].to_numpy(float)
        b = low[outcome].to_numpy(float)
        differences = b - a
        boot_means = differences[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(boot_means, (0.025, 0.975))
        rows.append(
            {
                "family": family,
                "contrast_id": contrast_id,
                "high_mode": high_mode,
                "low_mode": low_mode,
                "high_label": labels[high_mode],
                "low_label": labels[low_mode],
                "outcome": outcome,
                "n_paired_trials": len(high),
                "high_mean": a.mean(),
                "low_mean": b.mean(),
                "change": differences.mean(),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def latex_change_with_ci(row: pd.Series, digits: int) -> str:
    """Display a symmetric, outward-rounded margin enclosing the 95% CI."""
    displayed = float(f"{float(row['change']):.{digits}f}")
    margin = max(
        displayed - float(row["ci95_low"]),
        float(row["ci95_high"]) - displayed,
    )
    scale = 10**digits
    margin = np.ceil(margin * scale - 1e-12) / scale
    return f"${displayed:+.{digits}f}\\pm{margin:.{digits}f}$"


def write_latex(table: pd.DataFrame, path: Path) -> None:
    lookup = table.set_index(["family", "contrast_id", "outcome"])
    row_order = [
        (family, contrast_id)
        for family, *_ in SIGN_FAMILIES
        for contrast_id, _, _ in SIGN_CONTRASTS
    ] + [
        (family, "high_to_low") for family, _ in SHUFFLE_FAMILIES
    ]
    lines = [
        r"\begin{table}[tbp]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Family & Conditions (high $\to$ low) & $\rho_{\rm raw}$ (high $\to$ low) & IPC & KR & MC & GR \\",
    ]
    for stat, title, digits in (
        ("mean", "Mean metric change", 2),
        ("cv", "Hyperparameter CV change", 3),
    ):
        lines.extend((r"\hline", rf"\multicolumn{{7}}{{l}}{{\textbf{{{title}}}}} \\"))
        for family, contrast_id in row_order:
            rho = lookup.loc[(family, contrast_id, "raw_rho")]
            contrast = f"{rho['high_label']} $\\to$ {rho['low_label']}"
            if family in {name for name, *_ in SIGN_FAMILIES}:
                contrast += " negative edges"
            radius = f"${rho['high_mean']:.2f}\\to{rho['low_mean']:.2f}$"
            cells = [family, contrast, radius]
            cells.extend(
                latex_change_with_ci(
                    lookup.loc[(family, contrast_id, f"{metric}_{stat}")], digits
                )
                for metric in METRICS
            )
            lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Within-family high-to-low raw-radius contrasts. The 0\% to 40\% negative-edge branch and shuffle controls appear in Figure~\ref{fig:rawrho}; 100\% to 60\% is the complementary E/I sweep. Radii are condition means; metric entries are lower-minus-higher changes $\pm$ outward-rounded 95\% paired-bootstrap margins (200 seeds, 96 hyperparameter settings per seed). Lower GR indicates better generalization across earlier histories. Contrasts are descriptive under spectral-radius normalization.}",
            r"\label{tab:raw-rho-support}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Path to final_results")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    rng = np.random.default_rng(args.seed)
    rows = []
    sign_trials = {}
    for family, folder, prefix in SIGN_FAMILIES:
        trials = read_trials(args.results_dir / "sign_frac" / folder / INPUT_NAME)
        modes = list(trials.index.get_level_values("mode").unique())
        fractions = {mode: sign_fraction(mode, prefix) for mode in modes}
        mode_by_fraction = {round(fraction, 9): mode for mode, fraction in fractions.items()}
        labels = {mode: f"{100 * fraction:g}\\%" for mode, fraction in fractions.items()}
        sign_trials[family] = (trials, mode_by_fraction, labels)
    for contrast_id, high_fraction, low_fraction in SIGN_CONTRASTS:
        for family, *_ in SIGN_FAMILIES:
            trials, mode_by_fraction, labels = sign_trials[family]
            high_mode = mode_by_fraction[high_fraction]
            low_mode = mode_by_fraction[low_fraction]
            rows.extend(
                compare(
                    trials, family, contrast_id, high_mode, low_mode,
                    args.bootstrap, rng, labels,
                )
            )

    shuffle_trials = read_trials(args.results_dir / "shuf" / INPUT_NAME)
    for family, modes in SHUFFLE_FAMILIES:
        high_mode, low_mode = endpoint_modes(shuffle_trials, list(modes))
        rows.extend(
            compare(
                shuffle_trials,
                family,
                "high_to_low",
                high_mode,
                low_mode,
                args.bootstrap,
                rng,
                MODE_LABELS,
            )
        )

    table = pd.DataFrame(rows)
    csv_path = args.results_dir / "raw_rho_support_table.csv"
    tex_path = args.results_dir / "raw_rho_support_table.tex"
    table.to_csv(csv_path, index=False, float_format="%.8g")
    write_latex(table, tex_path)
    print(f"Wrote {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
