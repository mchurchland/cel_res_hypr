"""Compare paired recurrent-gain conditions and tanh saturation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd


HERE = Path(__file__).resolve().parent
CONDITIONS = (
    ("Baseline-compensated STP", "real_seed_12345_input_1", "#777777"),
    ("Raw STP, recovery 10 / facilitation 50", "real_seed_12345_raw_input_1", "#176087"),
    ("Raw STP, recovery 50 / facilitation 10", "real_seed_12345_raw_depressing_r50_f10", "#268457"),
    ("Fixed 0.1× weight control", "real_seed_12345_static_gain_u_input_1", "#c15b2e"),
)

fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
fig.subplots_adjust(left=0.10, right=0.66, top=0.84, bottom=0.12, hspace=0.13)
for ax in axes:
    ax.axvspan(0, 500, color="#e8eef4")
    ax.axvspan(500, 2000, color="#f4f4f4")
    ax.axvspan(2000, 2500, color="#eaf3e7")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_xlim(0, 2500)

for label, directory, color in CONDITIONS:
    if label.startswith("Fixed"):
        trace = pd.read_csv(HERE / "static_gain_u_saturation.csv")
    else:
        trace = pd.read_csv(HERE / directory / "effective_rho_trace.csv")
    metrics = pd.read_csv(HERE / directory / "invariance_variants.csv")
    if len(trace) != 26 or len(metrics) != 1:
        raise ValueError(f"Expected one complete sampled trial in {directory}.")
    mc = float(metrics["MC"].iloc[0])
    axes[0].plot(
        trace["step"], trace["effective_rho"], color=color,
        linewidth=1.7, marker="o", markersize=3.2,
        label=f"{label} (MC {mc:.2f})",
    )
    axes[1].plot(
        trace["step"], trace["fraction_abs_pre_ge_2"], color=color,
        linewidth=1.7, marker="o", markersize=3.2,
    )

axes[0].set_yscale("log")
axes[0].set_yticks([4, 8, 16, 32, 64, 128, 256])
axes[0].yaxis.set_major_formatter(ScalarFormatter())
axes[0].set_ylim(3, 250)
axes[0].set_ylabel("Effective spectral radius\n(log axis)")
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Fraction of preactivations\nwith |pre| ≥ 2")
axes[1].set_xlabel("Simulation step")
axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.8)
fig.text(0.10, 0.935, "How STP gain changes recurrent drive and saturation", fontsize=14)
fig.text(
    0.10, 0.89,
    "Real connectome · input scale 1.0 · leak 0.6 · reservoir seed 12346 · one trial per condition",
    fontsize=9.2, color="#444444",
)
fig.text(
    0.10, 0.045,
    "Saturation is measured over each preceding 100-step window; |tanh(pre)| ≥ 0.964 when |pre| ≥ 2.",
    fontsize=9, color="#555555",
)
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"saturation_diagnostic.{suffix}", dpi=220)
plt.close(fig)
