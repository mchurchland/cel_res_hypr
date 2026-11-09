"""Plot the single real-connectome STP demonstration run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent / "real_seed_12345"
trace = pd.read_csv(HERE / "effective_rho_trace.csv")
metrics = pd.read_csv(HERE / "invariance_variants.csv")
if len(metrics) != 1 or len(trace) != 26:
    raise ValueError("Expected one metrics row and the 26 samples from steps 0–2500.")
trial = metrics.iloc[0]

fig, ax = plt.subplots(figsize=(10.2, 5.2))
fig.subplots_adjust(left=0.10, right=0.77, top=0.84, bottom=0.22)
ax.axvspan(0, 500, color="#e8eef4", label="Washout")
ax.axvspan(500, 2000, color="#f5f5f5", label="Training")
ax.axvspan(2000, 2500, color="#eaf3e7", label="Test")
ax.plot(
    trace["step"], trace["effective_rho"],
    color="#166088", marker="o", markersize=4.5, linewidth=1.8,
    label="Effective spectral radius",
)
ax.axhline(
    float(trial["raw_rho"]), color="#bd563a", linestyle="--", linewidth=1.3,
    label=f"Initial unscaled radius = {trial['raw_rho']:.1f}",
)
ax.set_xlim(0, 2500)
ax.set_ylim(bottom=0)
ax.set_xlabel("Simulation step")
ax.set_ylabel("Spectral radius of effective recurrent matrix")
ax.set_title("Real connectome with baseline-compensated STP gain", loc="left", fontsize=13)
ax.grid(axis="y", color="#dddddd", linewidth=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5, frameon=False)

fig.text(
    0.10, 0.89,
    "Base seed 12345 (reservoir seed 12346)  ·  leak 0.6  ·  input scale 0.1  ·  bias 0  ·  sampled every 100 steps",
    fontsize=9.2, color="#454545",
)
fig.text(
    0.10, 0.08,
    f"MC {trial['MC']:.3f}    IPC {trial['IPC']:.3f}    "
    f"KR {trial['KR']:.2f}    GR {trial['GR']:.2f}",
    fontsize=10, color="#222222",
)
for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"effective_rho_over_time.{suffix}", dpi=220)
plt.close(fig)
