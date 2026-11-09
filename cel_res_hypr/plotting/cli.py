"""Cli for reservoir research figures."""

from cel_res_hypr.plotting.style import COLOR_SCHEMES
import argparse


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute dispersion from a merged CSV and plot histograms."
    )
    ap.add_argument(
        "--combined",
        default="experiment_full_merged/combined.ALL.csv",
        help="Path to merged combined CSV.",
    )
    ap.add_argument(
        "--out-dir",
        default="experiment_full_merged",
        help="Output directory for dispersion CSV and figures.",
    )
    ap.add_argument(
        "--bins", type=int, default=40, help="Histogram bins for dispersion plots."
    )
    ap.add_argument(
        "--frac-cv-bins",
        type=int,
        default=4,
        help="Mean-CV axis tick bins for the frac/CV/performance mean-line plot.",
    )
    ap.add_argument(
        "--scatter-alpha", type=float, default=0.55, help="Alpha for MC-vs-GR scatter."
    )
    ap.add_argument(
        "--model",
        default="",
        help="Optional exact mode/model name filter (e.g., 'weight_test10.0') for rho/CV/performance plot.",
    )
    ap.add_argument(
        "--rho-cv-drop-kr-gr",
        action="store_true",
        help="For rho/CV/performance plot, exclude KR and GR curves (plot MC/IPC only).",
    )
    ap.add_argument(
        "--compare-mean-a",
        default="",
        help="Optional path to first comparison CSV (mean_by_group.ALL.csv or dispersion_by_group.ALL.csv).",
    )
    ap.add_argument(
        "--compare-mean-b",
        default="",
        help="Optional path to second comparison CSV (mean_by_group.ALL.csv or dispersion_by_group.ALL.csv).",
    )
    ap.add_argument(
        "--compare-label-a",
        default="full",
        help="Column label for --compare-mean-a output columns.",
    )
    ap.add_argument(
        "--compare-label-b",
        default="frac025",
        help="Column label for --compare-mean-b output columns.",
    )
    ap.add_argument(
        "--compare-mean-out",
        default="",
        help="Optional output CSV path for mode/metric mean comparison table.",
    )
    ap.add_argument(
        "--compare-plot-out",
        default="",
        help="Optional output PNG path for all-modes mean-difference plot.",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Run mean-table comparison only, then exit.",
    )
    ap.add_argument(
        "--compare-value-col",
        default="auto",
        help="Column to compare: 'auto', 'mean', or 'dispersion' (use 'dispersion' for CV means).",
    )
    ap.add_argument(
        "--compare-metric",
        default="",
        help="Optional metric filter for comparison plot (e.g., 'cosine_similarity', 'covariance', or 'kl_to_gaussian').",
    )
    ap.add_argument(
        "--compare-tost-preservation",
        action="store_true",
        help="Print how many baseline-significant pairwise TOST comparisons remain significant in B.",
    )
    ap.add_argument(
        "--compare-tost-alpha",
        type=float,
        default=0.05,
        help="Alpha threshold for TOST significance preservation reporting.",
    )
    ap.add_argument(
        "--compare-tost-bound-frac",
        type=float,
        default=0.05,
        help="Equivalence bound as fraction of |baseline metric median|, reused for both datasets.",
    )
    ap.add_argument(
        "--compare-tost-out",
        default="",
        help="Optional CSV path for per-pair TOST preservation details.",
    )
    ap.add_argument(
        "--local-sign-binary-csv",
        default="",
        help="Optional extra combined CSV used only to overlay mode='real' as a wall on weight-gauss plots.",
    )
    ap.add_argument(
        "--weight-gauss-cv-only",
        action="store_true",
        help="Plot weight-test modes as 2D interpolation fraction vs mean CV and mean metric value; do not use KL.",
    )
    ap.add_argument(
        "--sign-norm-ablation",
        action="store_true",
        help="Plot the sign-balance normalization ablation and exit before default histogram plots.",
    )
    ap.add_argument(
        "--sign-norm-prefix",
        default="sign_test_og_cel",
        help="Mode prefix used to identify sign-normalization ablation rows.",
    )
    ap.add_argument(
        "--sign-norm-cv-rho-target",
        type=float,
        default=None,
        help=(
            "Optional nominal spectral-radius target used for the sign-normalization "
            "CV. When omitted, calculate CV within each target and average targets."
        ),
    )
    ap.add_argument(
        "--sign-norm-cv-include-rho",
        action="store_true",
        help=(
            "Calculate one sign-normalization CV across spectral radius, leak, "
            "input scale, and neuron bias. Cannot be combined with "
            "--sign-norm-cv-rho-target."
        ),
    )
    ap.add_argument(
        "--triad-sign-fraction-plot",
        action="store_true",
        help=(
            "Plot triad sign-fraction results and exit. Reads the grouped triad summary CSV "
            "and plots sign_frac on x versus triad_avg_abs_w_mean_mean on y."
        ),
    )
    ap.add_argument(
        "--triad-summary-csv",
        default="network_stats/triad_sign_fraction_results/triad_sign_fraction_group_summary.ALL.csv",
        help="Grouped triad summary CSV produced by triad_sign_fraction_experiment.py --merge-chunks.",
    )
    ap.add_argument(
        "--triad-plot-out",
        default="",
        help="Optional output path for --triad-sign-fraction-plot.",
    )
    ap.add_argument(
        "--triad-normalization",
        default="spectral_radius",
        help="Normalization filter for triad plot. Use 'all' to plot every normalization present.",
    )
    ap.add_argument(
        "--triad-scope",
        default="closed",
        help="Triad scope filter for triad plot. Use 'all' to facet every scope present.",
    )
    ap.add_argument(
        "--triad-rho-target",
        type=float,
        default=None,
        help="Optional rho_target filter for triad plot. Default facets all rho_target values.",
    )
    ap.add_argument(
        "--show-cv-performance-3d",
        action="store_true",
        help=(
            "Show interactive joint-distribution hills with CV on x, mean "
            "performance on y, and normalized trial fraction as height."
        ),
    )
    ap.add_argument(
        "--show-cv-performance-contours",
        action="store_true",
        help=(
            "Show the publication-oriented top-down joint CV/performance density "
            "contours for each metric."
        ),
    )
    ap.add_argument(
        "--show-cv-performance-contour-triptych",
        action="store_true",
        help=(
            "Plot the binary, sign-preserving, and C. elegans shuffle-control "
            "CV/performance contour comparisons together in one 3x4 figure."
        ),
    )
    ap.add_argument(
        "--triptych-axis-combined",
        default="",
        help=(
            "Optional combined.ALL.csv used only to set CV/performance triptych axis limits. "
            "Use this to plot one dataset with axes matched to another dataset."
        ),
    )
    ap.add_argument(
        "--cv-performance-contour-percent",
        "--contour-percent",
        dest="cv_performance_contour_percent",
        type=float,
        default=50.0,
        help="Highest-density percentage shown by CV/performance contours (default: 50).",
    )
    ap.add_argument(
        "--color-scheme",
        choices=COLOR_SCHEMES,
        default="thesis",
        help=(
            "Figure palette. 'extended-abstract' uses a visibly distinct palette "
            "while leaving the thesis defaults unchanged."
        ),
    )
    ap.add_argument(
        "--cv-baseline-mode",
        default="auto",
        help=(
            "Mode used as the CV-difference baseline. The default 'auto' uses "
            "real/CE-real when present and binary_base for binary shuffle runs."
        ),
    )
    return ap.parse_args()
