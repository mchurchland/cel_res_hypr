#!/usr/bin/env python3
"""Run-271800 narrative supplement, preserving all primary estimates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crossover-analysis")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/crossover-analysis-xdg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.analysis_dir.resolve()
    files = {name: out / "tables" / (name + ".csv") for name in
             ["crossovers", "contrasts", "validation_selected_narma", "parameter_contributions", "narma_seed_tail_diagnostics", "largest_narma_test_errors"]}
    hashes = {name: sha(path) for name, path in files.items()}
    data = {name: pd.read_csv(path) for name, path in files.items()}
    manifest = json.loads((out / "manifest.json").read_text())
    spec = manifest["experiment_spec"]
    validation = json.loads((out / "VALIDATION.json").read_text())
    if validation.get("run_id") != "271800":
        raise ValueError("This inspected-run narrative is specific to 271800; use the generic analysis for other runs.")
    roots = data["crossovers"]
    roots = roots[roots.scope.str.startswith("input_") & (roots.n_mean_crossings == 1)].copy()
    roots["input_scale"] = roots.scope.str.removeprefix("input_").astype(float)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    for ax, metric in zip(axes, ["MC", "NARMA10_test_nmse"]):
        for norm, name, color, marker in [("spectral_radius", "Own-radius", "#555555", "o"),
                                           ("original_radius", "Fixed-reference", "#009988", "s")]:
            g = roots[(roots.metric == metric) & (roots.normalization == norm)].sort_values("input_scale")
            ax.plot(g.input_scale, g.root, marker + "-", color=color, label=name, ms=5)
            # Quantile intervals need not contain a point estimate in general;
            # draw endpoints directly instead of assuming nonnegative yerr.
            ax.vlines(g.input_scale, g.ci_low, g.ci_high, color=color, lw=1.4)
        ax.set_xscale("log")
        inputs = sorted(spec["input_scales"])
        ax.set_xticks(inputs, [f"{x:g}" for x in inputs])
        ax.set_xlabel("Input scale (log axis)")
        ax.set_title("Memory capacity" if metric == "MC" else "NARMA10 test NMSE")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Interpolated CE-to-ER crossover (% inhibitory edges)")
    axes[0].legend(frameon=False)
    fig.suptitle("The performance boundary depends on input drive\nCoarse E/I sampling; bars show seed uncertainty, not between-sample resolution", fontsize=11)
    fig.savefig(out / "figures/09_crossover_vs_input.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def contrast(scope, norm, metric, fraction):
        g = data["contrasts"]
        r = g[(g.scope == scope) & (g.normalization == norm) & (g.metric == metric) & np.isclose(g.ei_percent, fraction)].iloc[0]
        return f"{r.ce_advantage:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"

    lines = [f"# Run {validation['run_id']}: important findings", "",
             f"{validation['repeats']} paired computational realizations; {validation['rows']:,} rows; "
             f"{len(spec['rho_values'])*len(spec['leak_values'])*len(spec['input_scales'])} settings per condition. "
             "Static reservoirs, with empirical-weight CE versus binary-weight ER; this is not the STP/weighted-control sweep.", "",
             "**The clearest extension is an advantage at strongly imbalanced E/I that also appears on NARMA10 "
             "after validation-based tuning.** CE has lower selected-setting NARMA test error at 0% and 10% "
             "inhibition under both normalization protocols; ER has lower selected-setting test error at "
             "20–50%. All twelve corresponding pointwise paired intervals exclude zero.", "",
             "| Inhibitory edges | Normalization | CE selected test NMSE | ER selected test NMSE | ER − CE error [95% interval] |",
             "| ---: | --- | ---: | ---: | --- |"]
    for r in data["validation_selected_narma"].query("ei_percent <= 10").itertuples():
        name = "Own-radius" if r.normalization == "spectral_radius" else "Fixed-reference"
        lines.append(f"| {r.ei_percent:g}% | {name} | {r.cel_matched_test_nmse:.3f} | {r.matched_er_test_nmse:.3f} | {r.ce_advantage:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |")
    lines += ["", "![Validation-selected NARMA](figures/04_validation_selected_narma.png)", "",
              "**Weaker input moves the interpolated CE–ER boundary toward higher inhibitory fractions.** "
              "This is a conditional regime, not a universal advantage at weaker input. Crossovers depend on "
              "the input setting, task and normalization.", "",
              "| Input scale | MC own-radius crossover | MC fixed-reference crossover | NARMA own-radius crossover | NARMA fixed-reference crossover |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for scale in sorted(spec["input_scales"]):
        vals = []
        for metric, norm in [("MC", "spectral_radius"), ("MC", "original_radius"),
                             ("NARMA10_test_nmse", "spectral_radius"), ("NARMA10_test_nmse", "original_radius")]:
            r = roots[(roots.metric == metric) & (roots.normalization == norm) & np.isclose(roots.input_scale, scale)]
            vals.append(f"{r.iloc[0].root:.1f}%" if len(r) else "No unique crossing")
        lines.append(f"| {scale:g} | " + " | ".join(vals) + " |")
    lines += ["", "![Boundary versus input drive](figures/09_crossover_vs_input.png)", "",
              "**These roots are descriptive interpolations, not measurements to 0.1 percentage points.** "
              "The sweep samples 0, 10, 20, 30, 40 and 50%. For example, input 0.1 gives an own-radius MC "
              "crossing near 22.7% using this grid, versus 25.2% using the earlier finer 20–40% grid. "
              "Matched overlapping measurements are nearly identical; the difference demonstrates sensitivity "
              "to interpolation resolution. Use the earlier fine sweep for that setting, and do not infer a "
              "biological optimum from proximity to the known-edge sign fraction.", "",
              "**At 20% inhibition, mean comparisons and tuned-task comparisons differ.** Across this newly "
              f"chosen grid, CE's mean MC advantage is {contrast('grid_mean','spectral_radius','MC',20)} "
              f"under own-radius scaling and {contrast('grid_mean','original_radius','MC',20)} under fixed-reference scaling. "
              "Nevertheless, validation-selected NARMA favors ER there. Grid averages also change when "
              "the input levels change; they are not a direct replication of the previous grid average.", "",
              "**Input scale is not the dominant MC variance contributor on every grid.** On this mostly "
              "weak-input grid with own-radius normalization, allocated CE MC variance is 38.1% leak, "
              "34.0% radius and 27.9% input; ER is 50.1% leak, 34.0% radius and 15.9% input. "
              "Those shares include equally allocated interactions and characterize these sampled levels.", "",
              "**One catastrophic NARMA error strongly affects a mean.** CE repeat 63, 40% inhibition, "
              "radius 1.05, leak 0.6 and input 0.01 has test NMSE 604.04 despite validation NMSE 0.233. "
              "It is retained. After averaging the six radius/leak settings within each seed at that input/fraction, "
              "CE's mean error is 1.233 but its median seed-mean error is 0.221; ER's corresponding values "
              "are 0.197 and 0.195. The median paired advantage also favors ER, so the large error exaggerates "
              "the mean gap without explaining away ER's typical-seed advantage. Tail and whole-seed "
              "leave-one-out diagnostics are exported without filtering the primary analysis.", "",
              "**Provenance:** all source shards, simulation code and connectome hashes validate. "
              "The run reuses the previous 100 seed IDs; 9,600 overlapping rows agree to small numerical "
              "differences. Treat it as an expanded design, not an independent 100-seed replication. "
              "Weight distribution remains confounded with architecture until the new weighted controls return.", "",
              "The full report, pointwise intervals, configuration-specific comparisons and allocation tables "
              "are in `REPORT.md` and `tables/`. No new simulations, row filtering or PDFs."]
    (out / "HIGHLIGHTS.md").write_text("\n".join(lines) + "\n")
    if any(sha(files[k]) != v for k, v in hashes.items()):
        raise ValueError("Primary analysis tables changed.")
    metadata = dict(source_table_hashes=hashes, code_sha256=sha(Path(__file__)),
                    supplementary_only=True, primary_estimates_unchanged=True,
                    note="Narrative highlights are specific to the inspected 271800 design.")
    (out / "supplementary_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved HIGHLIGHTS.md and figures/09_crossover_vs_input.png to {out}")


if __name__ == "__main__":
    main()
