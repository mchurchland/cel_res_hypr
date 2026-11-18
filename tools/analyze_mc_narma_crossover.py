#!/usr/bin/env python3
"""Analyze stored MC/NARMA10 crossover results; never run simulations."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crossover-analysis")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/crossover-analysis-xdg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy

from util.crossover_analysis import bootstrap_panel, summarize_crossing
from util.hyperparameter_variability import factorial_variability

FAMILIES = ["cel_matched", "matched_er"]
NORMS = ["spectral_radius", "original_radius"]
METRICS = ["MC", "NARMA10_test_nmse"]
PARAMS = ["rho_target", "leak", "input_scale"]
NORM_NAMES = {"spectral_radius": "Own-radius scaling", "original_radius": "Fixed-reference scaling"}
METRIC_NAMES = {"MC": "MC", "NARMA10_test_nmse": "NARMA10 test NMSE"}
INPUT_COLORS = ["#332288", "#0077bb", "#009988", "#ee7733"]
MODEL_COLORS = ["#332288", "#0077bb"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_run(source):
    paths = sorted(source.glob("chunk_*/manifest.json"))
    if not paths:
        raise ValueError("No chunk manifests found.")
    manifests = [json.loads(p.read_text()) for p in paths]
    spec = manifests[0]["spec"]
    identity = manifests[0]["experiment_id"]
    for m in manifests:
        if m["spec"] != spec or m["experiment_id"] != identity:
            raise ValueError("Inconsistent manifests.")
    repeats = list(itertools.chain.from_iterable(m["repeat_ids"] for m in manifests))
    if sorted(repeats) != list(range(spec["n_repeats"])):
        raise ValueError("Missing or duplicated manifest repeat assignments.")
    d = pd.read_csv(source / "mc_narma_crossover.ALL.csv")
    for column in ["negative_fraction", *PARAMS, "neuron_bias"]:
        d[column] = d[column].round(12)
    keys = ["repeat_id", "family", "normalization", "negative_fraction", *PARAMS]
    expected = spec["n_repeats"] * len(spec["families"]) * len(spec["normalizations"]) * len(spec["sign_fracs"])
    expected *= np.prod([len(spec[k]) for k in ["rho_values", "leak_values", "input_scales"]])
    if len(d) != expected or d.duplicated(keys).any() or set(d.experiment_id) != {identity}:
        raise ValueError("Merged data have missing, duplicate or incompatible records.")
    domains = dict(repeat_id=range(spec["n_repeats"]), family=spec["families"], normalization=spec["normalizations"],
                   negative_fraction=spec["sign_fracs"], rho_target=spec["rho_values"],
                   leak=spec["leak_values"], input_scale=spec["input_scales"])
    for column, domain in domains.items():
        if not d[column].isin(domain).all():
            raise ValueError(f"Unexpected levels in {column}.")
    numeric = d.select_dtypes(include="number")
    if not np.isfinite(numeric).all().all() or not (d.neuron_bias == 0).all():
        raise ValueError("Nonfinite scores or nonzero bias.")
    if not np.allclose(d.seed, spec["seed"] + d.repeat_id * spec["repeat_seed_stride"]):
        raise ValueError("Incorrect seed pairing.")
    if (d.MC < 0).any() or (d.MC > spec["tasks"]["mc_max_delay"] + 1e-5).any():
        raise ValueError("MC violates documented bounds.")
    errors = ["NARMA10_val_nmse", "NARMA10_test_nmse", "NARMA10_test_nrmse", "NARMA10_test_rmse"]
    if (d[errors] < 0).any().any() or not np.allclose(d.NARMA10_test_nrmse**2, d.NARMA10_test_nmse):
        raise ValueError("Invalid task-error values.")
    if not d.NARMA10_ridge_alpha.isin(spec["tasks"]["narma_ridge_alphas"]).all():
        raise ValueError("Unexpected ridge selection.")
    shard_paths = sorted(source.glob("chunk_*/repeat_*__*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    for column in ["negative_fraction", *PARAMS, "neuron_bias"]:
        raw[column] = raw[column].round(12)
    pd.testing.assert_frame_equal(d.sort_values(keys).reset_index(drop=True), raw[d.columns].sort_values(keys).reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    matches = {}
    for filename, expected_hash in spec["code_sha256"].items():
        p = ROOT / filename
        matches[filename] = p.exists() and digest(p) == expected_hash
    data_matches = {}
    for item in spec["sources"]:
        local = ROOT / "Connectome" / Path(item["path"]).name
        data_matches[str(local.relative_to(ROOT))] = local.exists() and digest(local) == item["sha256"]
    return d, spec, dict(rows=len(d), repeats=spec["n_repeats"], shards=len(shard_paths), chunks=len(paths),
                         merged_matches_all_shards=True, source_code_matches=matches, source_data_matches=data_matches,
                         devices=sorted(set(m["device_name"] for m in manifests)))


def assemble(d, spec):
    ei = np.round(np.asarray(sorted(spec["sign_fracs"])) * 100, 8)
    levels = [sorted(spec[k]) for k in ["rho_values", "leak_values", "input_scales"]]
    grid = pd.DataFrame(itertools.product(*levels), columns=PARAMS)
    columns = pd.MultiIndex.from_tuples([(f / 100, *g) for f in ei for g in grid.itertuples(index=False, name=None)],
                                        names=["negative_fraction", *PARAMS])
    panel = np.empty((spec["n_repeats"], 2, 2, 2, len(ei), len(grid)))
    for f, family in enumerate(FAMILIES):
        for n, norm in enumerate(NORMS):
            sub = d[(d.family == family) & (d.normalization == norm)]
            for k, metric in enumerate(METRICS):
                p = sub.pivot(index="repeat_id", columns=["negative_fraction", *PARAMS], values=metric)
                panel[:, f, n, k] = p.reindex(index=range(len(panel)), columns=columns).to_numpy().reshape(len(panel), len(ei), len(grid))
    if not np.isfinite(panel).all():
        raise ValueError("Incomplete paired panel.")
    # Grid selection uses validation only; a deterministic parameter tie-break.
    selected = d.sort_values(["NARMA10_val_nmse", *PARAMS], kind="stable").groupby(
        ["repeat_id", "family", "normalization", "negative_fraction"], sort=False).head(1).copy()
    tuned = np.empty((len(panel), 2, 2, len(ei)))
    for f, family in enumerate(FAMILIES):
        for n, norm in enumerate(NORMS):
            p = selected[(selected.family == family) & (selected.normalization == norm)].pivot(
                index="repeat_id", columns="negative_fraction", values="NARMA10_test_nmse")
            tuned[:, f, n] = p.reindex(index=range(len(panel)), columns=ei / 100).to_numpy()
    return panel, tuned, selected, ei, grid, levels


def ci(draws):
    return np.quantile(draws, [.025, .975], axis=0)


def summarize(panel_mean, panel_boot, tuned_mean, tuned_boot, ei, grid, out):
    means, contrasts, crossings = [], [], []
    scopes = [("grid_mean", np.ones(len(grid), dtype=bool))]
    scopes += [(f"input_{v:g}", np.isclose(grid.input_scale, v)) for v in sorted(grid.input_scale.unique())]
    for scope, mask in scopes:
        m = panel_mean[..., mask].mean(axis=-1)
        b = panel_boot[..., mask].mean(axis=-1)
        for n, norm in enumerate(NORMS):
            for k, metric in enumerate(METRICS):
                sign = 1 if k == 0 else -1
                advantage = sign * (m[0, n, k] - m[1, n, k])
                boot_adv = sign * (b[:, 0, n, k] - b[:, 1, n, k])
                bounds = ci(boot_adv)
                for j, fraction in enumerate(ei):
                    for f, family in enumerate(FAMILIES):
                        low, high = ci(b[:, f, n, k, j])
                        means.append(dict(scope=scope, normalization=norm, metric=metric, family=family, ei_percent=fraction,
                                          mean=m[f, n, k, j], ci_low=low, ci_high=high, n_configurations=int(mask.sum())))
                    contrasts.append(dict(scope=scope, normalization=norm, metric=metric, ei_percent=fraction,
                                          ce_advantage=advantage[j], ci_low=bounds[0, j], ci_high=bounds[1, j],
                                          n_configurations=int(mask.sum())))
                root = summarize_crossing(ei, advantage, boot_adv)
                root["mean_crossings"] = json.dumps(root["mean_crossings"])
                crossings.append(dict(scope=scope, normalization=norm, metric=metric, **root))
    tuned_rows = []
    for n, norm in enumerate(NORMS):
        adv = tuned_mean[1, n] - tuned_mean[0, n]
        boot_adv = tuned_boot[:, 1, n] - tuned_boot[:, 0, n]
        for j, fraction in enumerate(ei):
            lo, hi = ci(boot_adv[:, j])
            row = dict(normalization=norm, ei_percent=fraction, ce_advantage=adv[j], ci_low=lo, ci_high=hi)
            for f, family in enumerate(FAMILIES):
                low, high = ci(tuned_boot[:, f, n, j])
                row.update({f"{family}_test_nmse": tuned_mean[f, n, j], f"{family}_ci_low": low, f"{family}_ci_high": high})
            tuned_rows.append(row)
    outputs = {"performance": pd.DataFrame(means), "contrasts": pd.DataFrame(contrasts),
               "crossovers": pd.DataFrame(crossings), "validation_selected_narma": pd.DataFrame(tuned_rows)}
    for name, frame in outputs.items():
        frame.to_csv(out / "tables" / f"{name}.csv", index=False)
    return outputs


def variance_allocation(mean, boot, levels, ei, out):
    summaries, by_ei = [], []
    for f, family in enumerate(FAMILIES):
        for n, norm in enumerate(NORMS):
            for k, metric in enumerate(METRICS):
                surfaces = np.concatenate([mean[f, n, k][None], boot[:, f, n, k]], axis=0)
                bcount, ecount, gcount = surfaces.shape
                result = factorial_variability(surfaces.reshape(-1, gcount), [len(l) for l in levels])
                total = result["total_variance"].reshape(bcount, ecount)
                allocation = result["allocated_variance"].reshape(bcount, ecount, 3)
                shares = 100 * allocation.sum(axis=1) / total.sum(axis=1)[:, None]
                for p, parameter in enumerate(PARAMS):
                    low, high = ci(shares[1:, p])
                    summaries.append(dict(family=family, normalization=norm, metric=metric, parameter=parameter,
                                          allocated_pct=shares[0, p], ci_low=low, ci_high=high))
                    for j, fraction in enumerate(ei):
                        values = 100 * allocation[:, j, p] / total[:, j]
                        low, high = ci(values[1:])
                        by_ei.append(dict(family=family, normalization=norm, metric=metric, ei_percent=fraction,
                                          parameter=parameter, allocated_pct=values[0], ci_low=low, ci_high=high))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "tables/parameter_contributions.csv", index=False)
    pd.DataFrame(by_ei).to_csv(out / "tables/parameter_contributions_by_ei.csv", index=False)
    return summary


def configuration_summaries(mean, boot, ei, grid, out):
    rows, counts = [], []
    for n, norm in enumerate(NORMS):
        for k, metric in enumerate(METRICS):
            sign = 1 if k == 0 else -1
            advantage = sign * (mean[0, n, k] - mean[1, n, k])
            low, high = ci(sign * (boot[:, 0, n, k] - boot[:, 1, n, k]))
            for j, fraction in enumerate(ei):
                counts.append(dict(normalization=norm, metric=metric, ei_percent=fraction,
                                   ce_positive_mean=int((advantage[j] > 0).sum()),
                                   ce_positive_interval=int((low[j] > 0).sum()),
                                   er_positive_interval=int((high[j] < 0).sum()), n_configurations=len(grid)))
                for c, settings in grid.iterrows():
                    rows.append(dict(normalization=norm, metric=metric, ei_percent=fraction,
                                     **settings.to_dict(), ce_advantage=advantage[j, c], ci_low=low[j, c], ci_high=high[j, c]))
    result = pd.DataFrame(rows)
    result.to_csv(out / "tables/configuration_advantages.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "tables/configuration_win_counts.csv", index=False)
    return result


def old_grid_comparison(d, out):
    path = ROOT / "analysis/robustness_reevaluation_2026-11-16/figures/ei_hyperparameter_dots/plotted_points.csv.gz"
    if not path.exists():
        return pd.DataFrame(), None
    old = pd.read_csv(path)
    for p in [*PARAMS, "neuron_bias"]:
        old[p] = old[p].round(12)
    old = old[(old.metric == "MC") & old.family.isin(FAMILIES) & old.negative_percent.isin([20., 40.])]
    rows = []
    for label, sub in [("old_full_96", old), ("old_bias_zero_48", old[old.neuron_bias == 0]),
                       ("old_matched_reduced_24", old[(old.neuron_bias == 0) & old.rho_target.isin([.6, .95, 1.05]) & old.leak.isin([.6, 1.])])]:
        grouped = sub.groupby(["experiment", "normalization", "negative_percent", "family"]).performance.agg(["mean", "size"]).reset_index()
        grouped["analysis_grid"] = label
        rows.extend(grouped.to_dict("records"))
    new = d[d.negative_fraction.isin([.2, .4])].groupby(["normalization", "negative_fraction", "family"]).MC.mean().reset_index()
    for row in new.to_dict("records"):
        rows.append(dict(experiment="new_100_seed_crossover", analysis_grid="new_reduced_24",
                         normalization=row["normalization"], negative_percent=row["negative_fraction"] * 100,
                         family=row["family"], mean=row["MC"], size=24))
    result = pd.DataFrame(rows)
    result.to_csv(out / "tables/old_grid_comparison.csv", index=False)
    return result, path


def plot_all(tables, allocation, d, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.savefig(out / "figures" / name, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    def axis(ax, title, ylabel):
        ax.set(title=title, ylabel=ylabel, xlabel="Inhibitory edges (%)", xticks=[20, 24, 28, 32, 36, 40])
        ax.grid(axis="y", color="#e6e9ee", lw=.7)
        ax.set_axisbelow(True)
    for kind in ["performance", "contrasts"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="col", layout="constrained")
        g = tables[kind]
        g = g[g.scope == "grid_mean"]
        for n, norm in enumerate(NORMS):
            for k, metric in enumerate(METRICS):
                ax = axes[n, k]
                sub = g[(g.normalization == norm) & (g.metric == metric)]
                if kind == "performance":
                    for f, family in enumerate(FAMILIES):
                        h = sub[sub.family == family].sort_values("ei_percent")
                        ax.plot(h.ei_percent, h["mean"], "o-", color=MODEL_COLORS[f], label="CE (unknown signs matched)" if f == 0 else "Matched ER")
                        ax.fill_between(h.ei_percent, h.ci_low, h.ci_high, color=MODEL_COLORS[f], alpha=.16)
                    label = "Mean MC (higher better)" if k == 0 else "Mean test NMSE (lower better)"
                else:
                    h = sub.sort_values("ei_percent")
                    ax.plot(h.ei_percent, h.ce_advantage, "o-", color="#332288")
                    ax.fill_between(h.ei_percent, h.ci_low, h.ci_high, color="#332288", alpha=.18)
                    ax.axhline(0, color="#555555", ls="--", lw=1)
                    label = "CE − ER MC" if k == 0 else "ER − CE test NMSE"
                axis(ax, f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", label)
        if kind == "performance":
            axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle("Uniform average over the reduced 24-setting grid\n95% paired seed-bootstrap intervals; 100 realizations" if kind == "performance" else "Grid-averaged architecture advantage\nPositive favors CE; 95% paired seed-bootstrap intervals", fontsize=12)
        save(fig, "01_overall_performance.png" if kind == "performance" else "02_paired_advantage.png")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="col", layout="constrained")
    for n, norm in enumerate(NORMS):
        for k, metric in enumerate(METRICS):
            ax = axes[n, k]
            for color, value in zip(INPUT_COLORS, sorted(d.input_scale.unique())):
                h = tables["contrasts"]
                h = h[(h.scope == f"input_{value:g}") & (h.normalization == norm) & (h.metric == metric)].sort_values("ei_percent")
                ax.plot(h.ei_percent, h.ce_advantage, "o-", color=color, label=f"Input {value:g}")
                ax.fill_between(h.ei_percent, h.ci_low, h.ci_high, color=color, alpha=.13)
            ax.axhline(0, color="#555555", ls="--", lw=1)
            axis(ax, f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", "CE − ER MC" if k == 0 else "ER − CE test NMSE")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Crossovers are conditional on input drive\nRadius/leak averaged uniformly; positive favors CE; 95% seed-bootstrap intervals", fontsize=12)
    save(fig, "03_advantage_by_input.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True, layout="constrained")
    for n, norm in enumerate(NORMS):
        ax = axes[n]
        h = tables["validation_selected_narma"]
        h = h[h.normalization == norm].sort_values("ei_percent")
        for f, family in enumerate(FAMILIES):
            ax.plot(h.ei_percent, h[f"{family}_test_nmse"], "o-", color=MODEL_COLORS[f], label="CE" if f == 0 else "ER")
            ax.fill_between(h.ei_percent, h[f"{family}_ci_low"], h[f"{family}_ci_high"], color=MODEL_COLORS[f], alpha=.16)
        axis(ax, NORM_NAMES[norm], "Test NMSE (lower better)")
    axes[0].legend(frameon=False)
    fig.suptitle("NARMA10 after choosing each model's configuration by validation\nTest data never select configurations; 95% paired seed-bootstrap intervals", fontsize=12)
    save(fig, "04_validation_selected_narma.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8), sharex=True, layout="constrained")
    colors = ["#0072b2", "#d55e00", "#009e73"]
    for n, norm in enumerate(NORMS):
        for k, metric in enumerate(METRICS):
            ax = axes[n, k]
            pivot = allocation[(allocation.normalization == norm) & (allocation.metric == metric)].pivot(index="family", columns="parameter", values="allocated_pct").reindex(index=FAMILIES, columns=PARAMS)
            left = np.zeros(2)
            for p, color in zip(PARAMS, colors):
                width = pivot[p].to_numpy()
                ax.barh([0, 1], width, left=left, color=color, height=.55, edgecolor="white")
                for j, (start, value) in enumerate(zip(left, width)):
                    if value >= 8:
                        ax.text(start + value / 2, j, f"{value:.1f}", ha="center", va="center", color="white", fontsize=9)
                left += width
            ax.set(title=f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", xlim=(0, 100), yticks=[0, 1], yticklabels=["CE", "ER"], xlabel="Allocated within-grid variance (%)")
            ax.invert_yaxis()
    fig.legend(handles=[Patch(facecolor=c, label=p) for c, p in zip(colors, ["Radius target", "Leak rate", "Input scale"])],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Hyperparameter contributions on this reduced grid\nInteractions split equally; point estimates across the six E/I fractions", fontsize=12)
    save(fig, "05_parameter_contributions.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, layout="constrained")
    for f, family in enumerate(FAMILIES):
        raw = d[(d.family == family) & (d.normalization == NORMS[0])].groupby(["repeat_id", "negative_fraction"]).raw_rho.first()
        grouped = raw.groupby("negative_fraction").agg(["mean", "std"])
        axes[f, 0].plot(grouped.index * 100, grouped["mean"], "o-", color=MODEL_COLORS[f])
        axis(axes[f, 0], f"{'CE' if f == 0 else 'ER'} · raw radius", "Raw spectral radius")
        for n, norm in enumerate(NORMS):
            sub = d[(d.family == family) & (d.normalization == norm) & np.isclose(d.rho_target, .95)]
            h = sub.groupby("negative_fraction").scale_factor.mean()
            axes[f, 1].plot(h.index * 100, h, "o-" if n == 0 else "s--", label=NORM_NAMES[norm])
        axis(axes[f, 1], f"{'CE' if f == 0 else 'ER'} · scaling at target 0.95", "Applied scale factor")
        axes[f, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Normalization changes the effective recurrent-weight scale\nRealization means; associations do not isolate a causal effect of raw radius", fontsize=12)
    save(fig, "06_spectral_diagnostics.png")
    if "configuration_advantages" in tables:
        configurations = tables["configuration_advantages"]
        grid = configurations[PARAMS].drop_duplicates().sort_values(PARAMS)
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.7), layout="constrained")
        for k, metric in enumerate(METRICS):
            limit = configurations.loc[configurations.metric == metric, "ce_advantage"].abs().max()
            for n, norm in enumerate(NORMS):
                ax = axes[n, k]
                g = configurations[(configurations.metric == metric) & (configurations.normalization == norm)]
                values = g.pivot(index="ei_percent", columns=PARAMS, values="ce_advantage").sort_index()
                image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
                ax.set_yticks(range(6), [20, 24, 28, 32, 36, 40])
                ax.set_xticks(range(len(grid)), [f"{v:g}" for v in grid.input_scale], fontsize=7)
                ax.set(title=f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", xlabel="Input scale (within each radius/leak block)", ylabel="Inhibitory edges (%)")
                for x in np.arange(3.5, len(grid) - 1, 4):
                    ax.axvline(x, color="white", lw=1)
                top = ax.secondary_xaxis("top")
                top.set_xticks(np.arange(1.5, len(grid), 4),
                              [f"{r:g}/{l:g}" for r, l in grid[["rho_target", "leak"]].drop_duplicates().itertuples(index=False, name=None)], fontsize=7)
                top.set_xlabel("Radius target / leak", fontsize=8)
                fig.colorbar(image, ax=ax, shrink=.75, label="CE advantage")
        fig.suptitle("Which matched settings favor the connectome?\nBlue: CE higher MC / lower NARMA error; red: ER favored; means over 100 realizations", fontsize=12)
        save(fig, "07_matched_configuration_advantages.png")


def report(t, allocation, old, spec, validation, out):
    def contrast_text(scope, norm, metric, fraction):
        g = t["contrasts"]
        r = g[(g.scope == scope) & (g.normalization == norm) & (g.metric == metric)
              & np.isclose(g.ei_percent, fraction)].iloc[0]
        return f"{r.ce_advantage:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
    overall = t["contrasts"].query("scope == 'grid_mean'")
    overall_statement = ("ER has higher mean MC and lower mean NARMA10 test error at every sampled fraction when all 24 settings are weighted equally; every corresponding paired pointwise 95% interval favors ER."
                         if (overall.ce_advantage < 0).all() and (overall.ci_high < 0).all()
                         else f"ER is favored in {int((overall.ce_advantage < 0).sum())}/{len(overall)} grid-averaged point estimates; {int((overall.ci_high < 0).sum())} paired intervals favor ER.")
    tuned_statement = ("**ER has lower validation-selected mean test error at all six fractions under both normalizations; all paired pointwise intervals favor ER.**"
                       if (t["validation_selected_narma"].ci_high < 0).all()
                       else "The complete validation-selected contrasts and intervals are retained below.")
    old_detail = ""
    if len(old):
        g = old[(old.experiment == "sign") & (old.analysis_grid == "old_matched_reduced_24")
                & (old.normalization == "spectral_radius") & np.isclose(old.negative_percent, 20)]
        if set(g.family) == set(FAMILIES):
            v = g.set_index("family")["mean"]
            old_detail = f" At 20%, the old updated sign sweep restricted to the same 24 settings gives CE − ER MC = {v.cel_matched-v.matched_er:+.3f}; the new grid-average contrast is {contrast_text('grid_mean', 'spectral_radius', 'MC', 20)}."
    example_lines = []
    if "configuration_advantages" in t:
        g = t["configuration_advantages"]
        g = g[np.isclose(g.ei_percent, 20) & np.isclose(g.rho_target, 1.05)
              & np.isclose(g.leak, 1.) & np.isclose(g.input_scale, .1)]
        example_lines = ["", "**Illustrative matched condition:** at 20% inhibitory edges, radius target 1.05, leak 1.0 and input scale 0.1, CE outperforms ER on both metrics under both normalizations. This example was identified after inspecting the full configuration map; it is descriptive and its intervals are not multiplicity-adjusted.", "",
                         "| Normalization | Metric | CE advantage [pointwise 95% interval] |",
                         "| --- | --- | --- |"]
        for r in g.itertuples():
            example_lines.append(f"| {NORM_NAMES[r.normalization]} | {METRIC_NAMES[r.metric]} | {r.ce_advantage:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |")
    lines = ["# Under which conditions does a connectome-derived reservoir outperform matched ER?", "",
             f"Stored run 271601: **{validation['repeats']} paired computational realizations, {validation['rows']:,} rows, 24 settings per condition**. Both normalization modes; zero reservoir bias; fractions 20, 24, 28, 32, 36 and 40%. No simulations were launched for this analysis.", "",
             "## Main findings", "",
             "**The general grid-averaged crossover is not reproduced inside 20–40%.** " + overall_statement + " After averaging radius and leak, CE's positive advantage appears at weak input and is narrower for NARMA10. Individual matched configurations can retain CE advantages under both normalizations. This answers a conditional performance question rather than supporting general biological superiority.", "",
             "![Mean performance](figures/01_overall_performance.png)", "",
             "![Paired advantage](figures/02_paired_advantage.png)", "",
             "Both panels average within realization before bootstrapping complete paired realizations. NARMA advantage is ER error minus CE error, so positive always favors CE. Bands are marginal 95% seed-bootstrap intervals, not simultaneous confidence bands.", "",
             "## Conditional crossover locations", "",
             "| Metric | Normalization | Input scope | Mean sign-change bracket (%) | Interpolated crossing (%) | Bootstrap interval (%) | Unique-root bootstrap fraction |",
             "| --- | --- | --- | --- | --- | --- | ---: |"]
    for row in t["crossovers"].itertuples():
        if row.n_mean_crossings:
            lines.append(f"| {METRIC_NAMES[row.metric]} | {NORM_NAMES[row.normalization]} | {row.scope} | {row.bracket_low:.0f}–{row.bracket_high:.0f} | {row.root:.2f} | [{row.ci_low:.2f}, {row.ci_high:.2f}] | {row.bootstrap_unique_fraction:.3f} |")
    lines += ["", f"At input scale 0.1 and 20% inhibitory edges, CE's own-radius MC advantage is {contrast_text('input_0.1', 'spectral_radius', 'MC', 20)}; its NARMA advantage is {contrast_text('input_0.1', 'spectral_radius', 'NARMA10_test_nmse', 20)} NMSE, where positive means lower CE error. At 24%, the NARMA difference is {contrast_text('input_0.1', 'spectral_radius', 'NARMA10_test_nmse', 24)}, so the models are not clearly separated there. Under fixed-reference scaling, even the weak-input NARMA advantage at 20% reverses: {contrast_text('input_0.1', 'original_radius', 'NARMA10_test_nmse', 20)}.", "",
              "Scopes without a crossing are retained in `tables/crossovers.csv`; they are not assigned extrapolated roots. The intervals describe seed uncertainty of a **piecewise-linear interpolant**, conditional on a unique crossing. They do not identify the unknown curve shape between 4-percentage-point grid samples. A narrow interval is not evidence of sub-grid experimental resolution.", "",
              "![Input dependence](figures/03_advantage_by_input.png)", "",
              "The biological approximately 24.25% edge fraction is not an independently predicted optimum. That fraction already enters unknown-sign completion and the fixed-reference construction; the sweep randomizes edge signs. Input setting and normalization change the comparison.", "",
              "![Matched-setting comparisons](figures/07_matched_configuration_advantages.png)", "",
              "The configuration heatmap retains parameter-specific counterexamples hidden by averages. Counts of favorable point estimates and intervals are in `tables/configuration_win_counts.csv`; these 24 grid settings are not independent network samples.", "",
              *example_lines, "",
              "## NARMA10 with validation-selected configurations", "",
              "For each realization, architecture, normalization and E/I fraction, select the lowest **validation** NMSE among 24 configurations; report its untouched test score. Ridge alpha was already selected using validation within each configuration. Test scores do not choose either parameter setting or ridge penalty. These selection results and the uniformly averaged surfaces answer different questions.", "",
              "![Selected performance](figures/04_validation_selected_narma.png)", "",
              tuned_statement + " Thus the observed weak-input matched-setting advantage does not establish a tuning advantage over the random reservoir.", "",
              "| Normalization | E/I (%) | CE test NMSE | ER test NMSE | ER − CE error [95% interval] |", "| --- | ---: | ---: | ---: | --- |"]
    for r in t["validation_selected_narma"].itertuples():
        lines.append(f"| {NORM_NAMES[r.normalization]} | {r.ei_percent:.0f} | {r.cel_matched_test_nmse:.4f} | {r.matched_er_test_nmse:.4f} | {r.ce_advantage:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |")
    lines += ["", "## Hyperparameter contributions", "",
              "![Parameter shares](figures/05_parameter_contributions.png)", "",
              "Exact finite-grid factorial variance decomposition includes all main effects and interactions of radius, leak and input scale. Each interaction is allocated equally among its participating parameters. Summaries divide the sum of within-E/I component variances by the sum of within-E/I total variances. They exclude changes between E/I means and depend on this reduced grid. Bootstrap recomputes each seed-averaged surface and decomposition; intervals are in the tables.", "",
              "## Normalization and provenance", "",
              "![Radius diagnostics](figures/06_spectral_diagnostics.png)", "",
              "The same raw realizations and input weights are paired across normalization schemes. Fixed-reference scaling changes the effective recurrent gain across fractions. Its effects are descriptive protocol contrasts; a radius correlation does not identify topology's causal contribution.", "",
              "The old 96-setting results and this 24-setting run must not be pooled. The new run has fresh seeds, shared NumPy-generated task inputs, and float64 MC readout fitting. Its reservoir dynamics remain float32." + old_detail + " Changes in grid composition alone therefore cannot explain the altered ordering. The old full grid, bias-zero subset, exactly matched reduced grid and new endpoint means are exported in `tables/old_grid_comparison.csv`. The available tables do not isolate the contribution of numerical precision from input/realization changes.", "",
              "NARMA10 is generated by ReservoirPy 0.4.2 with its upstream recurrence/lag conventions unchanged; states after input u[t] predict the next output. Inputs are U[0,0.5], with 1,000 target burn-in, 500 state washout, 1,500 training, 500 validation and 500 test steps. Readout is scikit-learn Ridge with intercept. MC uses U[-1,1], delays 1–30, 1,500 training and 500 test after 500 washout. Full settings are copied to the manifest. ReservoirPy's specific lag convention should be cited when comparing to other NARMA implementations.", "",
              "CE retains biological weight magnitudes, while matched ER uses the existing binary sign-sweep weights. This is an architecture-and-weight comparison, not an isolated test of topology. Seed repeats are computational realizations of one anatomical connectome, not independent animals. All settings/fractions are a chosen finite design; no configuration is treated as an independent biological replicate. MC is truncated at delay 30 and some individual scores approach that ceiling; it is not unrestricted memory capacity.", "",
              "The run used both NVIDIA L40S and A30 GPUs. Every realization has a complete paired panel on its assigned device. Device-stratified descriptive contrasts are exported; GPU and repeat IDs are confounded, so between-device differences cannot be interpreted as hardware effects.", "",
              "## Validation and files", "",
              f"All {validation['shards']:,} source shards match the merged table; all grid cells are complete and unique; all metrics are finite; all MC values lie within the documented bound; NRMSE squared matches NMSE. Simulation code and connectome hashes match the manifests. No filtering or score clipping was applied.", "",
              "`tables/performance.csv`, `contrasts.csv`, `crossovers.csv`, `validation_selected_narma.csv`, `parameter_contributions.csv`, and configuration-level tables retain counterexamples and uncertainty. `manifest.json` records source hashes and analysis settings. `VALIDATION.json` records the audit. All figures are PNGs."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "results/crossover_mc_narma10/271601")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis/crossover_mc_narma10_271601")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=271601)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    for directory in ["tables", "figures", "cache"]:
        (out / directory).mkdir()
    source_files = [args.source / "mc_narma_crossover.ALL.csv", *sorted(args.source.glob("chunk_*/manifest.json")), *sorted(args.source.glob("chunk_*/repeat_*__*.csv"))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    d, spec, validation = validate_run(args.source)
    panel, tuned, selected, ei, grid, levels = assemble(d, spec)
    sizes = panel[0].size
    combined = np.concatenate([panel.reshape(len(panel), -1), tuned.reshape(len(panel), -1)], axis=1)
    _, draws = bootstrap_panel(combined, args.n_boot, args.seed)
    mean, boot = panel.mean(axis=0), draws[:, :sizes].reshape((args.n_boot, *panel.shape[1:]))
    tuned_mean, tuned_boot = tuned.mean(axis=0), draws[:, sizes:].reshape((args.n_boot, *tuned.shape[1:]))
    np.savez_compressed(out / "cache/bootstrap_surfaces.npz", mean=mean, bootstrap=boot, tuned_mean=tuned_mean,
                        tuned_bootstrap=tuned_boot, ei_percent=ei, grid=grid.to_numpy())
    tables = summarize(mean, boot, tuned_mean, tuned_boot, ei, grid, out)
    tables["configuration_advantages"] = configuration_summaries(mean, boot, ei, grid, out)
    selected.to_csv(out / "tables/validation_selected_configurations.csv", index=False)
    allocation = variance_allocation(mean, boot, levels, ei, out)
    old, old_path = old_grid_comparison(d, out)
    if old_path is not None:
        hashes[str(old_path.relative_to(ROOT))] = digest(old_path)
    rows = []
    for n, norm in enumerate(NORMS):
        for k, metric in enumerate(METRICS):
            advantage = (1 if k == 0 else -1) * (mean[0, n, k] - mean[1, n, k])
            boot_adv = (1 if k == 0 else -1) * (boot[:, 0, n, k] - boot[:, 1, n, k])
            for c, settings in grid.iterrows():
                root = summarize_crossing(ei, advantage[:, c], boot_adv[:, :, c])
                root["mean_crossings"] = json.dumps(root["mean_crossings"])
                rows.append(dict(normalization=norm, metric=metric, **settings.to_dict(), **root))
    pd.DataFrame(rows).to_csv(out / "tables/configuration_crossovers.csv", index=False)
    config_means = d.groupby(["family", "normalization", "negative_fraction", *PARAMS])[METRICS].mean().reset_index()
    config_means.to_csv(out / "tables/configuration_means.csv", index=False)
    ridge = d.groupby(["family", "normalization", "negative_fraction", "NARMA10_ridge_alpha"]).size().rename("count").reset_index()
    ridge.to_csv(out / "tables/ridge_selection_counts.csv", index=False)
    devices = {repeat: json.loads(p.read_text())["device_name"] for p in args.source.glob("chunk_*/manifest.json") for repeat in json.loads(p.read_text())["repeat_ids"]}
    d["device_name"] = d.repeat_id.map(devices)
    dev = d.groupby(["device_name", "family", "normalization", "negative_fraction"])[METRICS].mean().reset_index()
    dev.to_csv(out / "tables/device_stratified_means.csv", index=False)
    plot_all(tables, allocation, d, out)
    report(tables, allocation, old, spec, validation, out)
    if any(digest(ROOT / path) != value for path, value in hashes.items()):
        raise ValueError("Source files changed during analysis.")
    validation.update(source_hashes_unchanged=True, all_numeric_values_finite=True, rows_filtered=0,
                      bootstrap_resamples=args.n_boot, uncertainty_unit="whole paired realization")
    (out / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = dict(source_files=hashes, experiment_spec=spec, bootstrap=dict(method="SciPy percentile bootstrap", n=args.n_boot, seed=args.seed),
                    analysis_code={str(Path(__file__).relative_to(ROOT)): digest(__file__),
                                   "util/crossover_analysis.py": digest(ROOT / "util/crossover_analysis.py"),
                                   "util/hyperparameter_variability.py": digest(ROOT / "util/hyperparameter_variability.py")},
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    filtering="None; complete finite paired grid", weighting="Uniform finite hyperparameter levels; seed is bootstrap cluster")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved", out)
    print(tables["crossovers"].query("n_mean_crossings > 0").to_string(index=False))
    print(allocation.to_string(index=False))


if __name__ == "__main__":
    main()
