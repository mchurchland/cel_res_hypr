#!/usr/bin/env python3
"""Falsification-oriented robustness analysis of stored final_results only.

Run from the repository root:
  python tools/reevaluate_robustness.py
Use --out-dir for a new output directory, or --resume to reuse fingerprinted
intermediate tables in this analysis directory. Existing source results are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import numpy as np
import pandas as pd
import scipy
from util.robustness import (PARAMETERS, COORDINATES, performance_metrics, grid_edges,
                             local_sensitivity, variance_decomposition, feasible_variance,
                             cluster_spearman, spearman, bootstrap_mean)

METRICS = ("MC", "IPC", "KR")
# Implementation-based ceilings, explicitly conditional on archived protocol.
BOUNDS = {"MC": (0, 30), "IPC": (0, 150), "KR": (0, 297)}
SOURCES = {
    "main": "final_results/main/combined.ALL.GRKR_erank.rank_updated.csv",
    "shuf": "final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv",
    "baseline_shuf": "final_results/Rho_stuff/baseline_rho_shuf/combined.ALL.csv",
    **{f"sign_{family}": f"final_results/sign_frac/{family}/combined.ALL.GRKR_erank.rank_updated.csv"
       for family in ("cel_matched", "cel_removed", "matched_er")},
    **{f"norm_{family}": f"final_results/Rho_stuff/sign_norm/{family}/combined.ALL.csv"
       for family in ("cel_matched", "cel_removed", "matched_er")},
}
KEYS = ["dataset", "architecture", "normalization", "metric"]
STATS = ["mean", "cv", "sd", "iqr", "mad", "minimum", "p10", "bottom10", "maximum",
         "retention80", "retention90", "range", "variance", "attain_q25", "attain_q50", "attain_q75",
         "low_fraction", "above_q25_sd", "above_q25_mean", "above_q25_n", "sensitivity_mean",
         *[f"sensitivity_{p}_{s}" for p in PARAMETERS for s in ("mean", "p90")]]
ASSOCIATIONS = ["cv", "sd", "iqr", "mad", "minimum", "p10", "bottom10", "attain_q25",
                "attain_q50", "attain_q75", "retention80", "retention90", "range", "sensitivity_mean",
                "low_fraction", "above_q25_sd", "feasible_fraction"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(frame, path):
    compression = {"method":"gzip", "compresslevel":1} if str(path).endswith(".gz") else None
    frame.to_csv(path, index=False, float_format="%.17g", compression=compression)


def load_sources():
    sources, audits, frames = [], [], {}
    for dataset, relpath in SOURCES.items():
        path = ROOT / relpath
        frame = pd.read_csv(path)
        required = {"mode", "normalization", "seed", "shuffle_id", "src", *PARAMETERS, *METRICS,
                    "raw_rho", "ref_rho", "post_rho", "scale_factor"}
        if required-set(frame):
            raise ValueError(f"Missing columns in {path}: {required-set(frame)}")
        frame["architecture"] = frame["mode"].str.replace(r"__norm_.*$", "", regex=True)
        # No src-based block reconstruction: explicit seed identities are available.
        key = ["architecture", "normalization", "seed", *PARAMETERS]
        if frame.duplicated(key).any():
            raise ValueError(f"Duplicate seed/configuration identity: {path}")
        if frame.groupby(["architecture", "normalization", "seed"])["shuffle_id"].nunique().max() > 1:
            raise ValueError(f"Multiple network realizations for one seed: {path}")
        if not np.isfinite(frame[["seed", *PARAMETERS]].to_numpy(float)).all():
            raise ValueError(f"Missing seed/grid identity: {path}")
        for (arch, norm), group in frame.groupby(["architecture", "normalization"]):
            grid = group[list(PARAMETERS)].drop_duplicates()
            counts = group.groupby("seed").size()
            for metric in METRICS:
                values = group[metric].to_numpy(float)
                lower, upper = BOUNDS[metric]
                audits.append(dict(dataset=dataset, architecture=arch, normalization=norm, metric=metric,
                    n_rows=len(group), n_seeds=group.seed.nunique(), n_configurations=len(grid),
                    min_configurations_per_seed=counts.min(), max_configurations_per_seed=counts.max(),
                    n_nonfinite=int((~np.isfinite(values)).sum()), n_negative=int((values < 0).sum()),
                    observed_min=np.nanmin(values), observed_max=np.nanmax(values),
                    n_outside_conditional_bounds=int(((values < lower-1e-6)|(values > upper+1e-6)).sum())))
        sources.append(dict(dataset=dataset, path=relpath, sha256=sha256(path), bytes=path.stat().st_size,
                            n_rows=len(frame), columns=list(frame),
                            grid={p:sorted(frame[p].unique().tolist()) for p in PARAMETERS},
                            rank_vintage="updated_erank" if "rank_updated" in relpath else "archival_unverified_protocol"))
        frames[dataset] = frame
    return frames, sources, pd.DataFrame(audits)


def threshold_settings(frames):
    # Only main defines the reference quantiles; each architecture has equal rows.
    # Duplicated architectures in shuf do not reweight the reference distribution.
    thresholds = {}
    for metric in METRICS:
        x = frames["main"][metric].to_numpy(float)
        x = x[np.isfinite(x)]
        anchors = {f"attain_q{int(q*100)}":float(np.quantile(x, q)) for q in (.25, .5, .75)}
        maximum = max(np.nanmax(d[metric]) for d in frames.values())
        # Keep the zero-attainment endpoint distinct even after CSV serialization.
        endpoint = maximum+max(abs(maximum)*1e-6, 1e-8)
        curve = np.unique(np.r_[np.linspace(0, maximum, 81), list(anchors.values()), endpoint])
        thresholds[metric] = dict(anchors=anchors, curve=curve.tolist(),
            source="Exploratory quantiles of main pooled finite performance, equally weighted architectures; not externally validated usefulness thresholds")
    return thresholds


def extract(frames, thresholds, out):
    rows, surfaces, variance, curves, raw_diagnostics = [], [], [], [], []
    matrices = out / "cache" / "matrices"
    matrices.mkdir(parents=True, exist_ok=True)
    for dataset, data in frames.items():
        print(f"Extracting {dataset}", flush=True)
        for group_index, ((arch, norm), group) in enumerate(data.groupby(["architecture", "normalization"], sort=True)):
            grid = group[list(PARAMETERS)].drop_duplicates().sort_values(list(PARAMETERS)).reset_index(drop=True)
            index = pd.MultiIndex.from_frame(grid)
            seeds = np.sort(group.seed.unique())
            edges = grid_edges(grid)
            for metric in METRICS:
                meta = dict(dataset=dataset, architecture=arch, normalization=norm, metric=metric)
                y = group.pivot(index="seed", columns=list(PARAMETERS), values=metric).reindex(index=seeds, columns=index).to_numpy(float)
                np.savez_compressed(matrices/f"{dataset}_{group_index}_{metric}.npz", values=y, seeds=seeds, grid=grid.to_numpy(float))
                anchors = list(thresholds[metric]["anchors"].items())
                ts = np.asarray(thresholds[metric]["curve"])
                for seed, values in zip(seeds, y):
                    summary = performance_metrics(values, anchors)
                    summary.update(local_sensitivity(values, edges))
                    lower, upper = BOUNDS[metric]
                    summary["feasible_variance"] = float(feasible_variance(summary["mean"], lower, upper))
                    summary["feasible_fraction"] = summary["variance"]/summary["feasible_variance"] if summary["feasible_variance"] > 0 else np.nan
                    summary["distance_lower"] = summary["mean"]-lower
                    summary["distance_upper"] = upper-summary["mean"]
                    finite = values[np.isfinite(values)]
                    t25 = anchors[0][1]
                    above = finite[finite >= t25]
                    summary.update(low_fraction=np.mean(finite < t25) if len(finite) else np.nan,
                                   above_q25_n=len(above), above_q25_sd=np.std(above) if len(above) >= 20 else np.nan,
                                   above_q25_mean=above.mean() if len(above) else np.nan)
                    rows.append(dict(**meta, seed=seed, **summary))
                surface = np.nanmean(np.where(np.isfinite(y), y, np.nan), axis=0)
                surfaces.append(dict(**meta, **performance_metrics(surface, anchors), **local_sensitivity(surface, edges)))
                if np.isfinite(y).all() and min(y.shape) >= 2:
                    variance.append(dict(**meta, status="balanced", **variance_decomposition(y)))
                else:
                    variance.append(dict(**meta, status="incomplete: not decomposed"))
                valid = np.isfinite(y)
                den = valid.sum(axis=1)
                attain = ((y[:, :, None] >= ts)&valid[:, :, None]).sum(axis=1)/den[:, None]
                curves.append(pd.DataFrame(dict(**meta, seed=np.repeat(seeds, len(ts)),
                    threshold=np.tile(ts, len(seeds)), attainment=attain.ravel())))
            for seed, seed_data in group.groupby("seed"):
                expected = seed_data.rho_target/seed_data[norm == "spectral_radius" and "raw_rho" or "ref_rho"]
                raw_diagnostics.append(dict(dataset=dataset, architecture=arch, normalization=norm, seed=seed,
                    raw_rho=seed_data.raw_rho.mean(), raw_rho_span=np.ptp(seed_data.raw_rho),
                    scale_factor=seed_data.scale_factor.mean(), post_rho=seed_data.post_rho.mean(),
                    max_scaling_identity_error=np.max(np.abs(expected-seed_data.scale_factor)),
                    max_post_identity_error=np.max(np.abs(seed_data.raw_rho*seed_data.scale_factor-seed_data.post_rho))))
    tables = dict(per_realization=pd.DataFrame(rows), averaged_surface=pd.DataFrame(surfaces),
                  variance_components=pd.DataFrame(variance), threshold_per_seed=pd.concat(curves, ignore_index=True),
                  spectral_per_seed=pd.DataFrame(raw_diagnostics))
    for col in KEYS:
        tables["threshold_per_seed"][col] = tables["threshold_per_seed"][col].astype("category")
    for name, frame in tables.items():
        write_csv(frame, out/"cache"/f"{name}.csv.gz")
    return tables


def summaries(per, curve, n_boot, rng):
    records, threshold_rows = [], []
    for keys, group in per.groupby(KEYS, sort=True):
        columns = STATS + ["feasible_fraction"]
        mean, ci = bootstrap_mean(group[columns], n_boot, rng)
        row = dict(zip(KEYS, keys), n_realizations=len(group), min_valid_configurations=group.n_valid.min(), max_valid_configurations=group.n_valid.max())
        row.update(zip(columns, mean))
        for j, name in enumerate(columns):
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci[:, j]
        records.append(row)
    for keys, group in curve.groupby(KEYS, sort=True, observed=True):
        p = group.pivot(index="seed", columns="threshold", values="attainment")
        mean, ci = bootstrap_mean(p.to_numpy(), n_boot, rng)
        threshold_rows.extend(dict(zip(KEYS, keys), threshold=t, attainment=a, ci_low=lo, ci_high=hi, n_seeds=len(p))
                              for t, a, lo, hi in zip(p.columns, mean, ci[0], ci[1]))
    return pd.DataFrame(records), pd.DataFrame(threshold_rows)


def associations(per, n_boot, rng):
    rows = []
    for (dataset, metric, norm), group in per.groupby(["dataset", "metric", "normalization"]):
        print(f"Associations: {dataset}, {metric}, {norm}", flush=True)
        meta = dict(dataset=dataset, metric=metric, normalization=norm)
        # Fixed architecture panel, jointly resampled seed clusters.
        for measure in ASSOCIATIONS:
            rows.append(dict(**meta, scope="pooled_realizations", architecture="ALL", measure=measure,
                             **cluster_spearman(group, "mean", measure, n_boot=n_boot, rng=rng)))
        if dataset in ("main", "shuf"):
            for arch, sub in group.groupby("architecture"):
                for measure in ("cv", "sd", "iqr", "bottom10", "attain_q50", "sensitivity_mean"):
                    rows.append(dict(**meta, scope="within_architecture", architecture=arch, measure=measure,
                                     **cluster_spearman(sub, "mean", measure, n_boot=n_boot, rng=rng)))
            centroids = group.groupby("architecture")[list(set(["mean", *ASSOCIATIONS]))].mean()
            for measure in ASSOCIATIONS:
                rows.append(dict(**meta, scope="architecture_centroids_descriptive", architecture="ALL", measure=measure,
                                 rho=spearman(centroids["mean"], centroids[measure]), n=len(centroids),
                                 n_clusters=group.seed.nunique(), ci_low=np.nan, ci_high=np.nan, n_boot_valid=0))
            # Partial ranks account for an explicitly defined low-performance fraction.
            for measure in ("cv", "sd", "iqr", "mad"):
                d = group[["mean", measure, "low_fraction"]].dropna()
                ranked = d.rank().to_numpy()
                design = np.column_stack([np.ones(len(d)), ranked[:, 2]])
                residual = ranked[:, :2]-design@np.linalg.lstsq(design, ranked[:, :2], rcond=None)[0]
                rho = np.corrcoef(residual.T)[0, 1] if np.all(residual.std(axis=0)>1e-10) else np.nan
                rows.append(dict(**meta, scope="partial_rank_low_fraction_descriptive", architecture="ALL", measure=measure,
                                 rho=rho, n=len(d), n_clusters=group.seed.nunique(), ci_low=np.nan, ci_high=np.nan, n_boot_valid=0))
    return pd.DataFrame(rows)


def mean_matched(per):
    """Empirical overlap then nearest one-to-one caliper matches, no replacement.

    Caliper is 5% of the combined mean span. These conditioned observations are
    descriptive and are not used for causal or population significance claims.
    """
    rows, matches = [], []
    for (dataset, metric), group in per[per.dataset.isin(["main", "shuf"])].groupby(["dataset", "metric"]):
        for a, b in itertools.combinations(sorted(group.architecture.unique()), 2):
            ga, gb = (group[group.architecture == arch] for arch in (a, b))
            lo, hi = max(ga["mean"].min(), gb["mean"].min()), min(ga["mean"].max(), gb["mean"].max())
            aa, bb = (g[g["mean"].between(lo, hi)].copy() for g in (ga, gb))
            caliper = .05*(max(ga["mean"].max(), gb["mean"].max())-min(ga["mean"].min(), gb["mean"].min()))
            distance = np.abs(aa["mean"].to_numpy()[:, None]-bb["mean"].to_numpy()[None, :])
            ai, bi = np.where(distance <= caliper)
            candidates = sorted(zip(distance[ai, bi], aa.index.to_numpy()[ai], bb.index.to_numpy()[bi]))
            used_a, used_b, pairs = set(), set(), []
            for delta, i, j in candidates:
                if i not in used_a and j not in used_b:
                    used_a.add(i); used_b.add(j); pairs.append((i, j))
            meta = dict(dataset=dataset, metric=metric, architecture_a=a, architecture_b=b,
                        overlap_low=lo, overlap_high=hi, n_overlap_a=len(aa), n_overlap_b=len(bb),
                        n_matched=len(pairs), caliper=caliper)
            for measure in ["mean", "cv", "sd", "iqr", "mad", "bottom10", "attain_q50", "sensitivity_mean"]:
                difference = [group.loc[i, measure]-group.loc[j, measure] for i, j in pairs]
                rows.append(dict(**meta, measure=measure, matched_difference_a_minus_b=np.mean(difference) if difference else np.nan,
                                 full_difference_a_minus_b=ga[measure].mean()-gb[measure].mean()))
            matches.extend(dict(**meta, seed_a=group.loc[i, "seed"], seed_b=group.loc[j, "seed"],
                                mean_a=group.loc[i, "mean"], mean_b=group.loc[j, "mean"]) for i, j in pairs)
    return pd.DataFrame(rows), pd.DataFrame(matches)


def normalization_differences(per, n_boot, rng):
    rows = []
    for dataset in [x for x in per.dataset.unique() if x.startswith("norm_")] + ["shuffle_cross_run_MC_IPC"]:
        if dataset == "shuffle_cross_run_MC_IPC":
            own = per[(per.dataset == "shuf") & per.metric.isin(["MC", "IPC"])]
            fixed = per[(per.dataset == "baseline_shuf") & per.metric.isin(["MC", "IPC"])]
        else:
            own = per[(per.dataset == dataset)&(per.normalization == "spectral_radius")]
            fixed = per[(per.dataset == dataset)&(per.normalization == "original_radius")]
        joined = own.merge(fixed, on=["architecture", "metric", "seed"], suffixes=("_own", "_fixed"), validate="one_to_one")
        for (arch, metric), group in joined.groupby(["architecture", "metric"]):
            columns = ["mean", "cv", "sd", "iqr", "mad", "bottom10", "attain_q50", "retention90", "sensitivity_mean"]
            diff = group[[f"{c}_own" for c in columns]].to_numpy()-group[[f"{c}_fixed" for c in columns]].to_numpy()
            mean, ci = bootstrap_mean(diff, n_boot, rng)
            rows.extend(dict(dataset=dataset, architecture=arch, metric=metric, measure=c,
                             own_minus_fixed=d, ci_low=lo, ci_high=hi, n_paired_seeds=len(group),
                             rank_protocol="internal_archival_comparison" if metric == "KR" else "capacity_comparison")
                        for c, d, lo, hi in zip(columns, mean, ci[0], ci[1]))
    return pd.DataFrame(rows)


def spectral_associations(per, spectral, n_boot, rng):
    d = per.merge(spectral, on=["dataset", "architecture", "normalization", "seed"], validate="many_to_one")
    rows = []
    for (dataset, metric, norm), group in d.groupby(["dataset", "metric", "normalization"]):
        for x in ("raw_rho", "scale_factor"):
            for y in ("mean", "cv", "sd", "iqr", "bottom10", "sensitivity_mean"):
                rows.append(dict(dataset=dataset, metric=metric, normalization=norm, predictor=x, measure=y,
                                 **cluster_spearman(group, x, y, n_boot=n_boot, rng=rng)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"analysis/robustness_reevaluation_2026-11-16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.is_relative_to(ROOT/"final_results"):
        raise ValueError("Output must be outside final_results")
    if out.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume for this analysis only")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    out.mkdir(parents=True, exist_ok=True)
    (out/"cache").mkdir(exist_ok=True)
    (out/"tables").mkdir(exist_ok=True)
    (out/"figures").mkdir(exist_ok=True)
    frames, sources, audit = load_sources()
    thresholds = threshold_settings(frames)
    settings = dict(seed=args.seed, bootstrap=args.bootstrap, confidence=.95, ddof=0,
                    coordinate_transform=COORDINATES, sensitivity_coordinate_span="unit span for full design",
                    quantile_method="linear", bottom_decile="ceil(0.1*n) lowest observations; 10/96",
                    thresholds=thresholds, conditional_bounds=BOUNDS,
                    mean_matching="empirical pairwise overlap, greedy global nearest without replacement, 5% combined mean-span caliper",
                    missing="finite values only; no imputation; per-cell identity duplicates fail; complete matrix required for variance decomposition",
                    sources=sources)
    # Extraction cache also depends on analysis code, not only on source mtimes.
    code_hash = {str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__), ROOT/"util/robustness.py"]}
    fingerprint = hashlib.sha256(json.dumps(dict(settings=settings, code=code_hash), sort_keys=True).encode()).hexdigest()
    manifest = dict(settings=settings, fingerprint=fingerprint, code_hashes=code_hash,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    command=sys.argv, source_preservation="SHA256 verified again after analysis")
    manifest_path = out/"manifest.json"
    cached = manifest_path.exists() and json.loads(manifest_path.read_text()).get("fingerprint") == fingerprint
    cache_names = ["per_realization", "averaged_surface", "variance_components", "threshold_per_seed", "spectral_per_seed"]
    if args.resume and cached and all((out/"cache"/f"{n}.csv.gz").exists() for n in cache_names):
        tables = {n:pd.read_csv(out/"cache"/f"{n}.csv.gz") for n in cache_names}
        print("Using fingerprint-verified extraction cache", flush=True)
    else:
        tables = extract(frames, thresholds, out)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    del frames
    rng = np.random.default_rng(args.seed)
    print("Bootstrap summaries and associations", flush=True)
    comparison, curves = summaries(tables["per_realization"], tables["threshold_per_seed"], args.bootstrap, rng)
    assoc = associations(tables["per_realization"], args.bootstrap, rng)
    print("Mean matching and normalization diagnostics", flush=True)
    matched, pairs = mean_matched(tables["per_realization"])
    norm = normalization_differences(tables["per_realization"], args.bootstrap, rng)
    spectral_corr = spectral_associations(tables["per_realization"], tables["spectral_per_seed"], args.bootstrap, rng)
    outputs = dict(comparison=comparison, threshold_curves=curves, associations=assoc,
                   mean_matched=matched, mean_matched_pairs=pairs, normalization_differences=norm,
                   spectral_associations=spectral_corr, data_audit=audit,
                   **{k:v for k,v in tables.items() if k != "threshold_per_seed"})
    for name, frame in outputs.items():
        write_csv(frame, out/"tables"/f"{name}.csv")
    from util.robustness_plots import make_plots
    make_plots(outputs, thresholds, out/"figures")
    from util.robustness_report import write_report
    write_report(outputs, settings, out)
    for source in sources:
        if sha256(ROOT/source["path"]) != source["sha256"]:
            raise RuntimeError(f"Source changed during analysis: {source['path']}")
    (out/"SOURCE_INTEGRITY_VERIFIED.txt").write_text("All nine source CSV SHA256 hashes match their pre-analysis values.\n")
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
