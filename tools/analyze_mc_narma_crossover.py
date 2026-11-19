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
INPUT_COLORS = ["#332288", "#0077bb", "#009988", "#ee7733", "#cc3311", "#aa4499"]
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
        numeric_domain = np.asarray(list(domain))
        if np.issubdtype(numeric_domain.dtype, np.number):
            valid = np.isclose(d[column].to_numpy()[:, None], numeric_domain.astype(float)[None, :], atol=1e-12, rtol=0).any(axis=1)
        else:
            valid = d[column].isin(domain)
        if not np.all(valid):
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
    return d, spec, dict(run_id=source.name, rows=len(d), repeats=spec["n_repeats"], shards=len(shard_paths), chunks=len(paths),
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
    grid_size = len(d[PARAMS].drop_duplicates())
    new = d[d.negative_fraction.isin([.2, .4])].groupby(["normalization", "negative_fraction", "family"]).MC.mean().reset_index()
    for row in new.to_dict("records"):
        rows.append(dict(experiment="current_crossover_run", analysis_grid=f"current_grid_{grid_size}",
                         normalization=row["normalization"], negative_percent=row["negative_fraction"] * 100,
                         family=row["family"], mean=row["MC"], size=grid_size))
    result = pd.DataFrame(rows)
    result.to_csv(out / "tables/old_grid_comparison.csv", index=False)
    return result, path


def reference_run_comparison(d, spec, reference, out):
    """Audit exact overlap without treating the expanded seed grid as replication."""
    merged = reference / "mc_narma_crossover.ALL.csv"
    manifests = sorted(reference.glob("chunk_*/manifest.json"))
    if not merged.exists() or not manifests:
        return []
    metadata = [json.loads(p.read_text()) for p in manifests]
    reference_spec = metadata[0]["spec"]
    old = pd.read_csv(merged)
    keys = ["repeat_id", "family", "normalization", "negative_fraction", *PARAMS]
    for c in ["negative_fraction", *PARAMS]:
        old[c] = old[c].round(12)
    if old.duplicated(keys).any():
        raise ValueError("Reference run has duplicate configurations.")
    metrics = ["MC", "NARMA10_val_nmse", "NARMA10_test_nmse"]
    overlap = d[keys + metrics].merge(old[keys + metrics], on=keys, suffixes=("_current", "_reference"), validate="one_to_one")
    if overlap.empty:
        return []
    for metric in metrics:
        overlap[metric + "_difference"] = overlap[metric + "_current"] - overlap[metric + "_reference"]
    overlap.to_csv(out / "tables/reference_run_overlap.csv", index=False)
    same = {k: spec.get(k) == reference_spec.get(k) for k in
            ["protocol", "seed", "repeat_seed_stride", "seed_offsets", "tasks", "code_sha256", "packages", "sources"]}
    audit = dict(reference_run=reference.name, current_run=out.name, matched_rows=len(overlap),
                 paired_seed_ids=sorted(overlap.repeat_id.unique().tolist()),
                 same_seed_ids_not_independent_replication=True, shared_protocol_checks=same,
                 overlap_fractions_percent=sorted((overlap.negative_fraction.unique() * 100).tolist()),
                 overlap_input_scales=sorted(overlap.input_scale.unique().tolist()),
                 differences={m: dict(mean=float(overlap[m + "_difference"].mean()),
                                      mean_abs=float(overlap[m + "_difference"].abs().mean()),
                                      max_abs=float(overlap[m + "_difference"].abs().max())) for m in metrics})
    (out / "tables/reference_run_overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return [merged, *manifests]


def narma_tail_diagnostics(d, out):
    """Retain extreme errors; contrast means with typical paired realizations."""
    columns = ["repeat_id", "family", "normalization", "negative_fraction", *PARAMS,
               "NARMA10_val_nmse", "NARMA10_test_nmse", "NARMA10_ridge_alpha", "MC"]
    d.nlargest(20, "NARMA10_test_nmse")[columns].to_csv(out / "tables/largest_narma_test_errors.csv", index=False)
    rows = []
    scopes = [("grid_mean", d)] + [(f"input_{v:g}", d[np.isclose(d.input_scale, v)]) for v in sorted(d.input_scale.unique())]
    for scope, subset in scopes:
        paired = subset.groupby(["repeat_id", "family", "normalization", "negative_fraction"]).NARMA10_test_nmse.mean().reset_index()
        for (norm, fraction), g in paired.groupby(["normalization", "negative_fraction"]):
            panel = g.pivot(index="repeat_id", columns="family", values="NARMA10_test_nmse").sort_index()
            delta = panel.matched_er - panel.cel_matched
            # Influence diagnostic removes a whole paired realization, not an
            # isolated unfavorable configuration; no primary analysis is filtered.
            leave_one = (delta.sum() - delta) / (len(delta) - 1)
            rows.append(dict(scope=scope, normalization=norm, ei_percent=fraction * 100,
                             n_realizations=len(panel), ce_mean_error=float(panel.cel_matched.mean()),
                             er_mean_error=float(panel.matched_er.mean()),
                             ce_median_seed_mean_error=float(panel.cel_matched.median()),
                             er_median_seed_mean_error=float(panel.matched_er.median()),
                             mean_ce_advantage=float(delta.mean()), median_paired_ce_advantage=float(delta.median()),
                             fraction_seeds_favoring_ce=float((delta > 0).mean()),
                             leave_one_seed_out_mean_advantage_min=float(leave_one.min()),
                             leave_one_seed_out_mean_advantage_max=float(leave_one.max())))
    pd.DataFrame(rows).to_csv(out / "tables/narma_seed_tail_diagnostics.csv", index=False)


def plot_advantage_by_input(contrasts, out, nmse_limits=(-.5, .5)):
    """Render figure 03 from stored contrasts; limits only change the viewport."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        inputs = sorted(float(s.removeprefix("input_")) for s in contrasts.scope.unique() if s.startswith("input_"))
        fractions = sorted(contrasts.ei_percent.unique())
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="col", layout="constrained")
        for n, norm in enumerate(NORMS):
            for k, metric in enumerate(METRICS):
                ax = axes[n, k]
                for i, value in enumerate(inputs):
                    color = INPUT_COLORS[i % len(INPUT_COLORS)]
                    h = contrasts[(contrasts.scope == f"input_{value:g}") & (contrasts.normalization == norm) & (contrasts.metric == metric)].sort_values("ei_percent")
                    ax.plot(h.ei_percent, h.ce_advantage, "o-", color=color, label=f"Input {value:g}")
                    ax.fill_between(h.ei_percent, h.ci_low, h.ci_high, color=color, alpha=.13)
                ax.axhline(0, color="#555555", ls="--", lw=1)
                ax.set(title=f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}",
                       ylabel="CE − ER MC" if k == 0 else "ER − CE test NMSE",
                       xlabel="Inhibitory edges (%)", xticks=fractions)
                ax.grid(axis="y", color="#e6e9ee", lw=.7)
                ax.set_axisbelow(True)
                if k == 1:
                    ax.set_ylim(*nmse_limits)
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle("Crossovers are conditional on input drive\nRadius/leak averaged uniformly; positive favors CE; 95% seed-bootstrap intervals", fontsize=12)
        path = out / "figures/03_advantage_by_input.png"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path


def plot_all(tables, allocation, d, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        fig.savefig(out / "figures" / name, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    def axis(ax, title, ylabel):
        ax.set(title=title, ylabel=ylabel, xlabel="Inhibitory edges (%)", xticks=sorted(d.negative_fraction.unique() * 100))
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
        fig.suptitle(f"Uniform average over the {len(d[PARAMS].drop_duplicates())}-setting grid\n95% paired seed-bootstrap intervals; {d.repeat_id.nunique()} realizations" if kind == "performance" else "Grid-averaged architecture advantage\nPositive favors CE; 95% paired seed-bootstrap intervals", fontsize=12)
        save(fig, "01_overall_performance.png" if kind == "performance" else "02_paired_advantage.png")
    plot_advantage_by_input(tables["contrasts"], out)
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
    fig.suptitle(f"Hyperparameter contributions on this grid\nInteractions split equally; point estimates across {d.negative_fraction.nunique()} E/I fractions", fontsize=12)
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
                ax.set_yticks(range(len(values.index)), [f"{v:g}" for v in values.index])
                ax.set_xticks(range(len(grid)), [f"{v:g}" for v in grid.input_scale], fontsize=7)
                ax.set(title=f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", xlabel="Input scale (within each radius/leak block)", ylabel="Inhibitory edges (%)")
                n_inputs = grid.input_scale.nunique()
                for x in np.arange(n_inputs - .5, len(grid) - 1, n_inputs):
                    ax.axvline(x, color="white", lw=1)
                top = ax.secondary_xaxis("top")
                top.set_xticks(np.arange((n_inputs - 1) / 2, len(grid), n_inputs),
                              [f"{r:g}/{l:g}" for r, l in grid[["rho_target", "leak"]].drop_duplicates().itertuples(index=False, name=None)], fontsize=7)
                top.set_xlabel("Radius target / leak", fontsize=8)
                fig.colorbar(image, ax=ax, shrink=.75, label="CE advantage")
        fig.suptitle(f"Which matched settings favor the connectome?\nBlue: CE higher MC / lower NARMA error; red: ER favored; means over {d.repeat_id.nunique()} realizations", fontsize=12)
        save(fig, "07_matched_configuration_advantages.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="col", layout="constrained")
    for n, norm in enumerate(NORMS):
        for k, metric in enumerate(METRICS):
            ax = axes[n, k]
            for i, value in enumerate(sorted(d.input_scale.unique())):
                color = INPUT_COLORS[i % len(INPUT_COLORS)]
                for f, family in enumerate(FAMILIES):
                    h = tables["performance"]
                    h = h[(h.scope == f"input_{value:g}") & (h.normalization == norm) &
                          (h.metric == metric) & (h.family == family)].sort_values("ei_percent")
                    ax.plot(h.ei_percent, h["mean"], "o-" if f == 0 else "s--", color=color,
                            ms=3, lw=1.4, label=f"{'CE' if f == 0 else 'ER'} · input {value:g}")
                    ax.fill_between(h.ei_percent, h.ci_low, h.ci_high, color=color, alpha=.06)
            axis(ax, f"{NORM_NAMES[norm]} · {METRIC_NAMES[metric]}", "MC" if k == 0 else "NARMA10 test NMSE (log)")
            if k == 1:
                ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(labels)), fontsize=8, frameon=False)
    fig.suptitle("Absolute performance by input drive\nRadius/leak averaged within realization; paired seed-bootstrap intervals", fontsize=12)
    save(fig, "08_performance_by_input.png")


def report(t, allocation, old, spec, validation, out):
    from util.crossover_report import write_report
    write_report(t, allocation, old, spec, validation, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "results/crossover_mc_narma10/271601")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis/crossover_mc_narma10_271601")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=271601)
    p.add_argument("--reference-source", type=Path, default=ROOT / "results/crossover_mc_narma10/271601")
    args = p.parse_args()
    args.source = args.source.resolve()
    args.out_dir = args.out_dir.resolve()
    args.reference_source = args.reference_source.resolve()
    if args.n_boot < 100:
        p.error("Use at least 100 bootstrap resamples.")
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
    if args.reference_source.resolve() != args.source.resolve():
        reference_files = reference_run_comparison(d, spec, args.reference_source, out)
        hashes.update({str(p.relative_to(ROOT)): digest(p) for p in reference_files})
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
    narma_tail_diagnostics(d, out)
    plot_all(tables, allocation, d, out)
    report(tables, allocation, old, spec, validation, out)
    if any(digest(ROOT / path) != value for path, value in hashes.items()):
        raise ValueError("Source files changed during analysis.")
    validation.update(source_hashes_unchanged=True, all_numeric_values_finite=True, rows_filtered=0,
                      bootstrap_resamples=args.n_boot, uncertainty_unit="whole paired realization")
    (out / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = dict(source_files=hashes, experiment_spec=spec, bootstrap=dict(method="SciPy percentile bootstrap", n=args.n_boot, seed=args.seed),
                    analysis_code={str(Path(__file__).relative_to(ROOT)): digest(__file__),
                                   "util/crossover_report.py": digest(ROOT / "util/crossover_report.py"),
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
