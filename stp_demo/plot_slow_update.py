"""Plot paired memory capacity and STP dynamics for the 20-seed pilot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FAST = HERE / "fast_depression_20_seeds"
SLOW = HERE / "slow_depression_20_seeds"
BASELINE = HERE.parent / "final_results/main/combined.ALL.GRKR_PIL.csv"


def main() -> None:
    fast = pd.read_csv(FAST / "invariance_variants.csv")
    slow = pd.read_csv(SLOW / "invariance_variants.csv")
    baseline = pd.read_csv(BASELINE)
    baseline = baseline.loc[
        (baseline["mode"] == "real")
        & (baseline["normalization"] == "spectral_radius")
        & (baseline["rho_target"] == 1.05)
        & (baseline["leak"] == 0.6)
        & (baseline["input_scale"] == 1.0)
        & (baseline["neuron_bias"] == 0.0),
        ["seed", "MC"],
    ]
    paired = (
        fast[["seed", "MC"]].rename(columns={"MC": "fast"})
        .merge(slow[["seed", "MC"]].rename(columns={"MC": "slow"}), on="seed")
        .merge(baseline.rename(columns={"MC": "nonstp"}), on="seed")
        .sort_values("seed")
    )
    if len(paired) != 20:
        raise ValueError(f"Expected 20 paired seeds, got {len(paired)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    colors = {"fast": "#36749e", "slow": "#238455", "nonstp": "#bd6134"}
    x = np.arange(3)
    order = ["fast", "slow", "nonstp"]
    for _, row in paired.iterrows():
        axes[0].plot(x, [row[key] for key in order], color="0.72", alpha=0.55, lw=0.8)
    for xi, key in enumerate(order):
        axes[0].scatter(np.full(len(paired), xi), paired[key], color=colors[key], s=18, alpha=0.7, zorder=3)
        axes[0].hlines(paired[key].mean(), xi - 0.22, xi + 0.22, colors="black", lw=2.2, zorder=4)
    axes[0].set_xticks(x, ["Fast STP\nrate 1.0", "Slow STP\nrate 0.1", "Non-STP\nρ target 1.05"])
    axes[0].set_ylabel("Memory capacity")
    axes[0].set_title("Paired MC across 20 seeds")
    axes[0].grid(axis="y", alpha=0.2)

    for path, label, color in [
        (FAST, "Fast STP", colors["fast"]),
        (SLOW, "Slow STP", colors["slow"]),
    ]:
        trace = pd.read_csv(path / "effective_rho_trace.csv")
        for ax, column in [
            (axes[1], "effective_rho"),
            (axes[2], "fraction_abs_pre_ge_2"),
        ]:
            summary = trace.groupby("step")[column].agg(["mean", "std"])
            steps = summary.index.to_numpy()
            mean = summary["mean"].to_numpy()
            std = summary["std"].fillna(0).to_numpy()
            ax.plot(steps, mean, color=color, label=label, lw=2.2)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)

    axes[1].axhline(1.05, color=colors["nonstp"], ls="--", lw=1.5, label="Non-STP target")
    axes[1].set_title("Effective spectral radius (mean ± SD)")
    axes[1].set_xlabel("Simulation step")
    axes[1].set_ylabel("Radius")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.2)
    axes[2].set_title("100-step saturation (mean ± SD)")
    axes[2].set_xlabel("Simulation step")
    axes[2].set_ylabel("Fraction with |pre| ≥ 2")
    axes[2].set_ylim(0, 0.5)
    axes[2].grid(alpha=0.2)
    fig.suptitle("Slower Tsodyks–Markram updates: recovery 50, facilitation 10, input scale 1.0")
    fig.savefig(HERE / "slow_update_comparison.png", dpi=180)
    fig.savefig(HERE / "slow_update_comparison.pdf")


if __name__ == "__main__":
    main()
