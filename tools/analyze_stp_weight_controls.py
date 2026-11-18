#!/usr/bin/env python3
"""Paired realization bootstrap and PNG-only plots for the cluster STP sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-stp-weight-controls")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-stp-weight-controls")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stp_weight_controls import KEYS, expected_keys, publish_json, sha, validate_frame

METRICS = ("MC", "NARMA10_test_nmse")
GROUPS = ["ei_percent", "normalization", "input_scheme", "input_scale", "stp_condition", "gain_control"]
COMPARISONS = (
    ("ce_empirical", "er_signed_empirical", "CE vs ER, signed empirical weights"),
    ("ce_shuffled", "er_empirical", "CE-shuffled vs ER, empirical weights"),
    ("ce_binary", "er_binary", "CE vs ER, binary weights"),
    ("ce_empirical", "er_empirical", "CE vs ER, empirical magnitudes"),
    ("ce_empirical", "ce_shuffled", "CE empirical vs shuffled weight placement"),
    ("er_empirical", "er_halfnormal", "ER empirical vs half-normal magnitudes"),
    ("er_empirical", "er_binary", "ER empirical vs binary magnitudes"),
    ("ce_empirical", "er_binary", "CE empirical vs ER binary (legacy confounded comparison)"),
)


def estimate(values, n_boot=2000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite, nonempty paired realizations.")
    mean = float(values.mean())
    # A single realization cannot provide an uncertainty interval.
    lo = hi = np.nan
    if len(values) > 1:
        rng = np.random.default_rng(20261120)
        draws = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
        lo, hi = np.quantile(draws, [.025, .975])
    return dict(mean=mean, ci_low=float(lo), ci_high=float(hi), n_realizations=len(values))


def summaries(data, n_boot):
    rows = []
    for key, g in data.groupby(["family", *GROUPS], sort=True):
        g = g.sort_values("repeat_id")
        for metric in (*METRICS, "MC_gain_mean", "NARMA10_gain_mean", "initial_post_rho", "initial_scale"):
            rows.append(dict(zip(["family", *GROUPS], key), metric=metric, **estimate(g[metric], n_boot)))
    return pd.DataFrame(rows)


def paired_tables(data, n_boot):
    contrasts, changes, frozen = [], [], []
    for key, group in data.groupby(GROUPS, sort=True):
        info = dict(zip(GROUPS, key))
        for left, right, label in COMPARISONS:
            if left not in set(group.family) or right not in set(group.family):
                continue
            for metric in METRICS:
                panel = group[group.family.isin([left, right])].pivot(index="repeat_id", columns="family", values=metric).sort_index()
                if panel.isna().any().any():
                    raise ValueError("Unpaired architecture comparison.")
                delta = panel[left] - panel[right]
                if metric != "MC":
                    delta = -delta
                contrasts.append(dict(**info, comparison=label, left_family=left, right_family=right,
                                      metric=metric, positive_means="left family performs better",
                                      **estimate(delta, n_boot)))
    baseline_keys = ["family", "ei_percent", "normalization", "input_scheme", "input_scale"]
    for key, group in data.groupby(baseline_keys, sort=True):
        info = dict(zip(baseline_keys, key))
        baseline = group[(group.stp_condition == "static") & (group.gain_control == "none")].set_index("repeat_id").sort_index()
        for label, dynamic in group[(group.stp_condition != "static") & (group.gain_control == "none")].groupby("stp_condition"):
            dynamic = dynamic.set_index("repeat_id").sort_index()
            if not dynamic.index.equals(baseline.index):
                raise ValueError("Unpaired static/STP comparison.")
            for metric in METRICS:
                delta = dynamic[metric] - baseline[metric]
                if metric != "MC":
                    delta = -delta
                changes.append(dict(**info, stp_condition=label, metric=metric,
                                    positive_means="STP improves performance", **estimate(delta, n_boot)))
            matched = group[(group.stp_condition == label) & (group.gain_control == "training_mean_frozen")]
            if len(matched):
                matched = matched.set_index("repeat_id").sort_index()
                if not matched.index.equals(dynamic.index):
                    raise ValueError("Unpaired dynamic/frozen comparison.")
                for metric in METRICS:
                    delta = dynamic[metric] - matched[metric]
                    if metric != "MC":
                        delta = -delta
                    frozen.append(dict(**info, stp_condition=label, metric=metric,
                                       positive_means="dynamic STP beats training-mean frozen gain",
                                       **estimate(delta, n_boot)))
    return pd.DataFrame(contrasts), pd.DataFrame(changes), pd.DataFrame(frozen)


def make_plots(out, summary, contrasts, spec):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    families = [("ce_empirical", "CE empirical", "#332288"),
                ("ce_shuffled", "CE shuffled", "#aa4499"),
                ("ce_binary", "CE binary", "#882255"),
                ("er_empirical", "ER empirical", "#0077bb"),
                ("er_signed_empirical", "ER signed empirical", "#0099cc"),
                ("er_halfnormal", "ER half-normal", "#44aa99"),
                ("er_binary", "ER binary", "#999933")]
    for scheme, norm in itertools_product(spec["input_schemes"], spec["normalizations"]):
        selected = summary[(summary.input_scheme == scheme) & (summary.normalization == norm) & (summary.gain_control == "none")]
        # Direct absolute-performance curves for the static and reference presets.
        for condition in [c[0] for c in spec["stp_conditions"] if c[1] in ("none", "reference")]:
            fig, axes = plt.subplots(2, len(spec["input_scales"]), figsize=(6 * len(spec["input_scales"]), 7), squeeze=False, layout="constrained")
            for col, scale in enumerate(spec["input_scales"]):
                for row, metric in enumerate(METRICS):
                    ax = axes[row, col]
                    for family, label, color in families:
                        g = selected[(selected.family == family) & (selected.input_scale == scale) & (selected.stp_condition == condition) & (selected.metric == metric)].sort_values("ei_percent")
                        if g.empty:
                            continue
                        ax.plot(g.ei_percent, g["mean"], "o-", ms=3, lw=1.5, color=color, label=label)
                        ax.fill_between(g.ei_percent, g.ci_low, g.ci_high, alpha=.10, color=color)
                    ax.set_title(f"Input scale {scale:g}")
                    ax.set_xlabel("Inhibitory edges (%)")
                    ax.set_xticks(spec["ei_percent"])
                    ax.set_ylabel("MC (higher is better)" if metric == "MC" else "NARMA10 test NMSE (lower is better)")
                    if row == 1:
                        ax.set_yscale("log")
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{condition} · {scheme} · {norm}")
            fig.savefig(out / f"performance_{scheme}_{norm}_{condition}.png")
            plt.close(fig)
        for metric in METRICS:
            for scale in spec["input_scales"]:
                selected_c = contrasts[(contrasts.input_scheme == scheme) & (contrasts.normalization == norm) &
                                       (contrasts.input_scale == scale) & (contrasts.metric == metric) & (contrasts.gain_control == "none")]
                pairs = [(left, right, label) for left, right, label in COMPARISONS[:3]
                         if ((selected_c.left_family == left) & (selected_c.right_family == right)).any()]
                if not pairs:
                    continue
                fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 3.5 * len(pairs)), squeeze=False, layout="constrained")
                labels = [c[0] for c in spec["stp_conditions"]]
                values = []
                for left, right, _ in pairs:
                    g = selected_c[(selected_c.left_family == left) & (selected_c.right_family == right)]
                    values.append(g.pivot(index="stp_condition", columns="ei_percent", values="mean").reindex(index=labels, columns=spec["ei_percent"]).to_numpy())
                limit = max(1e-9, max(float(np.max(abs(v))) for v in values))
                for ax, (left, right, title), value in zip(axes[:, 0], pairs, values):
                    im = ax.imshow(value, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
                    ax.set_yticks(range(len(labels)), labels)
                    ax.set_xticks(range(len(spec["ei_percent"])), spec["ei_percent"])
                    ax.set_xlabel("Inhibitory edges (%)")
                    ax.set_title(title)
                    for i, label in enumerate(labels):
                        for j, fraction in enumerate(spec["ei_percent"]):
                            cell = selected_c[(selected_c.left_family == left) & (selected_c.right_family == right) &
                                              (selected_c.stp_condition == label) & (selected_c.ei_percent == fraction)].iloc[0]
                            marker = "*" if cell.ci_low > 0 or cell.ci_high < 0 else ""
                            ax.text(j, i, f"{value[i,j]:.2f}{marker}", ha="center", va="center", fontsize=8,
                                    color="white" if abs(value[i,j]) > .65 * limit else "black")
                fig.colorbar(im, ax=axes[:, 0], label="CE − ER MC" if metric == "MC" else "ER − CE test NMSE")
                fig.suptitle(f"{metric} · input {scale:g} · {scheme} · {norm}\nPositive favors CE; * pointwise paired 95% interval excludes zero")
                fig.savefig(out / f"paired_advantages_{scheme}_{norm}_{metric}_input{scale:g}.png")
                plt.close(fig)


def itertools_product(*args):
    from itertools import product
    return product(*args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.n_bootstrap < 100:
        parser.error("Use at least 100 bootstrap draws.")
    root, out = args.results_dir.resolve(), args.out_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    spec, identity = manifest["spec"], manifest["experiment_id"]
    source = root / "stp_weight_controls.ALL.csv"
    audit = json.loads((root / "merge_audit.json").read_text())
    if audit["csv_sha256"] != sha(source) or audit["experiment_id"] != identity:
        raise ValueError("Merged CSV does not match merge audit.")
    data = pd.read_csv(source)
    validate_frame(data, spec, identity, expected_keys(spec))
    out.mkdir(parents=True, exist_ok=True)
    publish_json(out / "analysis_manifest.json", dict(experiment_id=identity, source_csv_sha256=sha(source),
                                                      analysis_code_sha256=sha(Path(__file__)), n_bootstrap=args.n_bootstrap,
                                                      unit="paired computational realization; configurations are not independent biological samples"))
    summary = summaries(data, args.n_bootstrap)
    contrasts, changes, frozen = paired_tables(data, args.n_bootstrap)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    contrasts.to_csv(out / "paired_architecture_comparisons.csv", index=False)
    changes.to_csv(out / "stp_changes_from_static.csv", index=False)
    if not frozen.empty:
        frozen.to_csv(out / "dynamic_vs_frozen_gain.csv", index=False)
    make_plots(out, summary, contrasts, spec)
    report = f"""# STP and weighted-control follow-up

Source: `{source}`; SHA256 `{sha(source)}`.
Experiment `{identity}`. {len(data):,} paired MC/NARMA evaluations from
{spec['n_repeats']} computational realizations. E/I percentages: {spec['ei_percent']}.

## How to assess the proposed CE advantage

1. Compare CE empirical against ER signed empirical: their exact signed weight multiset matches.
   ER empirical additionally matches the unsigned magnitude multiset, with unconstrained
   positive/negative magnitude allocation. Signed matching redistributes magnitudes
   across fractions to preserve the signed multiset, while the other controls keep
   their unsigned placements fixed throughout the sweep.
2. Compare CE shuffled against ER empirical: both randomize magnitude placement.
3. Compare CE binary against ER binary: removes magnitude heterogeneity in both.
4. Read absolute performance alongside paired differences and changes from static.
   A relative CE advantage can reflect ER deterioration, not CE improvement.
5. Check each STP setting, input scale and input scheme separately. Do not select
   a winning setting on test scores or pool configurations as independent samples.

`paired_architecture_comparisons.csv` contains paired realization differences.
Positive values favor the left family (MC: left minus right; NARMA: right error
minus left error). Intervals are pointwise 95% percentile bootstrap intervals
over paired seeds ({args.n_bootstrap} draws), not multiplicity-adjusted discoveries.
`stp_changes_from_static.csv` distinguishes improvement from control degradation.
The PNG heatmaps show all prespecified parameter sets, including unfavorable ones.

## Scope and limitations

- This extends the original continuous-STP/tanh reservoir, not c302 or BAAIWorm.
- Simulation backend: {spec['device']}; readout fits remain on CPU. GPU runs batch
  independent trajectories, retain float64 STP variables and float32 neural states,
  and record device metadata for each realization attempt.
- Both utilization and depletion operate in every plastic neuron. Parameters
  are generic sensitivity settings in model-step units, not measured worm fits.
- STP uses abs(previous signed tanh state), one state per presynaptic neuron,
  and recovered-rest gain normalization u*x/U. Signs are independently randomized
  at the edge level; these are neither fixed biological signs nor Dale neurons.
- Unknown-sign edges are restored before the E/I sweep. The sweep randomizes
  ALL signs, not only unknown edges.
- The 0–50% restriction is a sampling choice. No symmetry is imposed or inferred;
  biased NARMA inputs, nonlinear dynamics and STP can break apparent symmetry.
- Own-radius scaling targets 0.95 only at initialization. Shared-CE scaling uses
  the CE empirical multiplier at EACH fraction for every family; it does not
  fix scale across fractions. Exact magnitude matching survives shared scaling,
  while own-radius scaling can change magnitude distributions differently.
- No radius/leak/bias grid and no time-dependent renormalization. Leak=1, bias=0.
- Full input and CE sensory/ER random input subsets are separate experiments;
  restricted input matches the full input vector L2 norm, not driven activity.
- Scalar input and all-neuron linear readouts; no claim about biological memory
  behavior follows directly from MC/NARMA performance.
- Weight-shuffled CE does not preserve node strengths; ER does not preserve CE
  degree sequences. These controls narrow, but do not isolate, all topology effects.
- Optional frozen gain controls enabled: {spec['gain_control']}. When present,
  they freeze each neuron's average transmission coefficient measured on training
  samples only. They do not match subsequent realized activity, nor constitute a
  unique causal isolation of synaptic history.
"""
    (out / "REPORT.md").write_text(report)
    print(f"Saved paired tables, PNG figures and report to {out}")


if __name__ == "__main__":
    main()
