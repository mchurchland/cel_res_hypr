"""Compare two paired real-connectome STP runs at different input scales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RUNS = (
    ("Input scale 0.1", "real_seed_12345", "#176087"),
    ("Input scale 1.0", "real_seed_12345_input_1", "#c15b2e"),
)

fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
fig.subplots_adjust(left=0.10, right=0.70, top=0.86, bottom=0.12, hspace=0.13)
for ax in axes:
    ax.axvspan(0, 500, color="#e8eef4")
    ax.axvspan(500, 2000, color="#f4f4f4")
    ax.axvspan(2000, 2500, color="#eaf3e7")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_xlim(0, 2500)

initial_rho = initial_abs = None
for label, directory, color in RUNS:
    trace = pd.read_csv(HERE / directory / "effective_rho_trace.csv")
    metrics = pd.read_csv(HERE / directory / "invariance_variants.csv")
    if len(trace) != 26 or len(metrics) != 1:
        raise ValueError(f"Expected one complete sampled trial in {directory}.")
    trial = metrics.iloc[0]
    if int(trial["seed"]) != 12346 or float(trial["leak"]) != 0.6:
        raise ValueError(f"Unpaired trial parameters in {directory}.")
    if initial_rho is None:
        initial_rho = float(trace["effective_rho"].iloc[0])
        initial_abs = float(trace["mean_abs_nonzero_weight"].iloc[0])
    axes[0].plot(
        trace["step"], trace["effective_rho"], color=color,
        linewidth=1.8, marker="o", markersize=3.6,
        label=f"{label} (MC {trial['MC']:.3f})",
    )
    axes[1].plot(
        trace["step"], trace["mean_abs_nonzero_weight"], color=color,
        linewidth=1.8, marker="o", markersize=3.6,
        label=label,
    )

axes[0].axhline(initial_rho, color="#666666", linestyle="--", linewidth=1.0)
axes[0].set_ylabel("Effective spectral radius")
axes[1].axhline(initial_abs, color="#666666", linestyle="--", linewidth=1.0)
axes[1].set_ylabel("Mean absolute effective weight\n(nonzero edges)")
axes[1].set_xlabel("Simulation step")
axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
fig.text(0.11, 0.94, "Baseline-compensated STP: input scale 0.1 versus 1.0", fontsize=14)
fig.text(
    0.11, 0.90,
    "Same reservoir seed 12346, leak 0.6, bias 0  ·  washout 0–500, training 500–2000, test 2000–2500",
    fontsize=9, color="#444444",
)
fig.text(0.11, 0.045, "Dashed lines show the initial unscaled values. Samples are 100 steps apart.",
         fontsize=9, color="#555555")
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"input_scale_comparison.{suffix}", dpi=220)
plt.close(fig)
