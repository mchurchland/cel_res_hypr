#!/usr/bin/env python3
"""Attribute hyperparameter-grid variability using verified stored surface caches."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-robustness")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/robustness-xdg-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.robustness import PARAMETERS
from util.hyperparameter_variability import factorial_variability

METRICS=("MC","IPC","KR")
FAMILIES={"cel_matched":"C. elegans: unknown signs matched", "cel_removed":"C. elegans: unknown signs removed", "matched_er":"Matched ER"}
NAMES={"rho_target":"Spectral radius","leak":"Leak rate","input_scale":"Input scale","neuron_bias":"Bias range"}
COLORS=["#0072b2","#d55e00","#009e73","#cc79a7"]
MEASURES=("main_pct","involving_pct","allocated_pct","matched_sweep_rms_sd","mean_matched_sweep_range")


def digest(path):
    with open(path,"rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def effect_measures(result):
    return dict(main_pct=100*result["main_fraction"],involving_pct=100*result["total_fraction"],
                allocated_pct=100*result["allocated_fraction"],matched_sweep_rms_sd=result["matched_sweep_rms_sd"],
                mean_matched_sweep_range=result["mean_matched_sweep_range"])


def summarize_boot(meta,measures):
    rows=[]
    for i,param in enumerate(PARAMETERS):
        row=dict(**meta,parameter=param)
        for name,array in measures.items():
            row[name]=float(array[0,i])
            row[f"{name}_ci_low"],row[f"{name}_ci_high"]=np.nanquantile(array[1:,i],[.025,.975])
        rows.append(row)
    return rows


def allocation_table(summary,out,family="cel_matched"):
    """Standalone table figure for one model family; PNG only."""
    labels={"cel_matched":"C. elegans · unknown signs matched", "cel_removed":"C. elegans · unknown signs removed", "matched_er":"Matched ER"}
    if family not in labels:
        raise ValueError(f"Unknown model family: {family}")
    selected=summary[(summary.experiment=="sign")&(summary.family==family)&(summary.normalization=="spectral_radius")]
    values=selected.pivot(index="parameter",columns="metric",values="allocated_pct").reindex(index=PARAMETERS,columns=METRICS)
    if values.isna().any().any():
        raise ValueError(f"Incomplete variance allocation: {family}")
    names={**NAMES,"rho_target":"Spectral-radius target"}
    cells=[[names[p],*[f"{v:.1f}%" for v in values.loc[p]]] for p in PARAMETERS]
    fig,ax=plt.subplots(figsize=(8.6,3.9))
    fig.subplots_adjust(left=.04,right=.96,top=.83,bottom=.03)
    ax.axis("off")
    fig.text(.04,.94,"Hyperparameter contributions to performance variability",fontsize=15,weight="semibold",color="#172b44")
    fig.text(.04,.865,f"{labels[family]} · seed-averaged surfaces",fontsize=10,color="#526176")
    table=ax.table(cellText=cells,colLabels=["Hyperparameter",*METRICS],cellLoc="center",
                   colWidths=[.46,.18,.18,.18],bbox=[0,.25,1,.70])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#dce3eb");cell.set_linewidth(.7)
        if row==0:
            cell.set_facecolor("#223b58");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f2f5f8" if row%2==0 else "white")
        if col==0:
            cell.get_text().set_ha("left");cell.PAD=.06
    for col,metric in enumerate(METRICS,1):
        row=PARAMETERS.index(values[metric].idxmax())+1
        cell=table[row,col]
        cell.set_facecolor("#e1f1e8");cell.get_text().set_weight("bold");cell.get_text().set_color("#155b39")
    fig.text(.04,.115,"Allocated share of within-grid variance across E/I levels; largest share in each metric highlighted.",fontsize=9,color="#526176")
    fig.text(.04,.06,"Interaction variance is split equally among participating hyperparameters.",fontsize=9,color="#526176")
    fig.savefig(out/f"{family}_hyperparameter_variability_table.png",dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def figures(conditions,summary,out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    for family in FAMILIES:
        allocation_table(summary,out,family)
    fig,axes=plt.subplots(3,3,figsize=(12.5,9),layout="constrained")
    for i,(family,label) in enumerate(FAMILIES.items()):
        for j,metric in enumerate(METRICS):
            g=summary[(summary.experiment=="sign")&(summary.family==family)&(summary.metric==metric)].set_index("parameter").reindex(PARAMETERS)
            ax=axes[i,j]
            y=np.arange(4)
            ax.barh(y,g.allocated_pct,color=COLORS,height=.55,alpha=.85)
            ax.errorbar(g.allocated_pct,y,xerr=[np.maximum(0,g.allocated_pct-g.allocated_pct_ci_low),np.maximum(0,g.allocated_pct_ci_high-g.allocated_pct)],fmt="none",ecolor="black",capsize=2,lw=.9)
            for yy,value in zip(y,g.allocated_pct):
                ax.text(min(value+2,91),yy,f"{value:.1f}%",va="center",fontsize=8)
            ax.set(yticks=y,yticklabels=[NAMES[p] for p in PARAMETERS],xlim=(0,100),xlabel="Allocated variance (%)",title=f"{label}\n{metric}")
            ax.invert_yaxis()
    fig.suptitle("Hyperparameter variability across E/I levels\nInteraction variance split equally among participating parameters; 95% seed-bootstrap intervals",fontsize=12)
    fig.savefig(out/"variance_allocation_summary.png",dpi=300,bbox_inches="tight");plt.close(fig)
    for family,label in FAMILIES.items():
        for experiment in ("sign","normalization"):
            fig,axes=plt.subplots(2,3,figsize=(13,6.5),layout="constrained")
            for i in range(2):
                norm="spectral_radius" if experiment=="sign" or i==0 else "original_radius"
                measure=("main_pct" if i==0 else "involving_pct") if experiment=="sign" else "involving_pct"
                desc=("Main effect only" if i==0 else "Including interactions") if experiment=="sign" else ("Own radius" if i==0 else "Fixed reference")
                for j,metric in enumerate(METRICS):
                    g=conditions[(conditions.experiment==experiment)&(conditions.family==family)&(conditions.normalization==norm)&(conditions.metric==metric)]
                    table=g.pivot(index="parameter",columns="negative_percent",values=measure).reindex(PARAMETERS)
                    values=table.to_numpy()
                    ax=axes[i,j]
                    im=ax.imshow(values,vmin=0,vmax=100,cmap="YlGnBu",aspect="auto")
                    for (rr,cc),value in np.ndenumerate(values):
                        ax.text(cc,rr,f"{value:.1f}",ha="center",va="center",fontsize=8,color="white" if value>55 else "black")
                    ax.set_yticks(range(4),[NAMES[p] for p in PARAMETERS])
                    ax.set_xticks(range(len(table.columns)),[f"{x:.1f}" if not np.isclose(x%20,0) else f"{x:.0f}" for x in table.columns])
                    ax.set(xlabel="Inhibitory edges (%)",title=f"{desc} · {metric}")
            fig.colorbar(im,ax=axes,shrink=.8,label="Share of within-grid variance (%)")
            foot="Main effects do not include interactions; total involvement shares overlap." if experiment=="sign" else "Total involvement shares overlap; archival KR remains separate from updated sign sweeps."
            fig.suptitle(f"{label}\n{foot}",fontsize=12)
            fig.savefig(out/f"{family}_{experiment}_variability_by_ei.png",dpi=300,bbox_inches="tight");plt.close(fig)


def report(conditions,summary,out,n_boot):
    main=summary[(summary.experiment=="sign")&(summary.family=="cel_matched")]
    table=main.pivot(index="parameter",columns="metric",values="allocated_pct").reindex(index=PARAMETERS,columns=METRICS)
    lines=["# Variability attributable to the hyperparameter grid\n",
           "The analysis quantifies the spread shown in the E/I dot plots, separately within every E/I level, architecture family, metric and normalization. It uses the original performance values, never jittered display coordinates.\n",
           "## Main connectome result\n\nAllocated share of within-grid variance, summarized across E/I levels (%):\n\n| Parameter | MC | IPC | KR |\n| --- | ---: | ---: | ---: |"]
    for p,row in table.iterrows():
        lines.append(f"| {NAMES[p]} | {row.MC:.1f} | {row.IPC:.1f} | {row.KR:.1f} |")
    lines.append("\nThese percentages split every interaction equally among its participating parameters, so the four shares sum to 100%. This is an explicit allocation convention; a joint effect is not uniquely caused by one parameter. Across E/I levels, component variances are summed and divided by summed total within-level variance. Each E/I level has equal design weight; levels with more grid variance contribute more to the resulting percentage. Variation between E/I levels is excluded.\n")
    lines.append("## What each quantity means\n\n- **Main-effect share:** variance of performance averaged over the other three parameters. It captures an effect that survives averaging; cancellation can hide conditional effects.\n- **Total involvement share:** variance from the main effect plus every interaction involving that parameter. These shares overlap and can sum to more than 100%.\n- **Allocated share:** main effect plus an equal split of each interaction among its participants; the four shares sum to 100%.\n- **Matched-sweep RMS SD:** root mean variance when changing only that parameter, holding the other three fixed, averaged over their settings. It is in the original performance units.\n- **Mean matched-sweep range:** mean max−min performance when changing just that parameter with the others fixed. It depends on extremes.\n")
    lines.append("## Design and uncertainty\n\nThe grid is 4 spectral-radius targets × 3 leak rates × 4 input scales × 2 bias ranges. Levels are equally weighted as discrete choices. An exact factorial decomposition retains all 15 main/interaction components and verifies that they sum to total variance. Results apply to these sampled ranges; they are not intrinsic parameter rankings independent of the chosen grid. Log/linear coordinate distances are relevant to the earlier local-sensitivity analysis, whereas this variance analysis weights the tested levels equally.\n")
    lines.append(f"The primary estimates decompose the seed-averaged surface, matching the dot plots. {n_boot} seed-cluster bootstrap replicates recompute the mean surface and its complete decomposition. Shared numerical seeds are resampled together across E/I levels, metrics and normalization modes with the same seed set; 95% percentile intervals are retained. No configuration is treated as an independent network sample. Per-seed decomposition is also exported as a distinct estimand. There is one observation per seed/configuration, so those per-seed effects can still include interaction with evaluation noise.\n")
    lines.append("Updated sign-only results (200 seeds) and archival normalization controls (100 seeds) remain separate. The requested E/I percentage denotes negative edges, not inhibitory neurons. All cached mean surfaces were checked against the exact plotted means, and their source hashes were checked against the earlier manifests. No new simulations were run.\n")
    lines.append("## Files\n\n- `variance_allocation_summary.png`: comparisons across architecture families and metrics.\n- `cel_matched_sign_variability_by_ei.png`: main versus total involvement for the main connectome at each E/I level; companion PNGs cover other families and normalization modes.\n- `tables/by_ei.csv`: main/total/allocated shares and matched-sweep SD/range with bootstrap intervals.\n- `tables/across_ei_summary.csv`: variance-weighted summaries and intervals.\n- `tables/components.csv`: every main/interaction variance component at each E/I level.\n- `tables/per_seed.csv.gz`: robustness calculated within each individual seed before averaging.\n- `manifest.json`: data/cache provenance, code hashes and analysis settings.\n\nReproduce with `python tools/quantify_hyperparameter_variability.py --out-dir analysis/new_variability`. All figures are PNGs.\n")
    (out/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,default=ROOT/"analysis/robustness_reevaluation_2026-11-16")
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=20261116)
    args=parser.parse_args()
    analysis=args.analysis_dir.resolve()
    out=(args.out_dir or analysis/"hyperparameter_variability").resolve()
    if out.exists() or out.is_relative_to(ROOT/"final_results"):
        raise ValueError("Choose a new output directory outside final_results")
    if args.bootstrap<100:
        raise ValueError("At least 100 bootstrap draws are required")
    dotdir=analysis/"figures/ei_hyperparameter_dots"
    provenance=json.loads((dotdir/"manifest.json").read_text())
    original=json.loads((analysis/"manifest.json").read_text())
    original_hashes={s["path"]:s["sha256"] for s in original["settings"]["sources"]}
    for source in provenance["sources"]:
        assert source["sha256"]==original_hashes[source["path"]]
        assert digest(ROOT/source["path"])==source["sha256"]
    points=pd.read_csv(dotdir/"plotted_points.csv.gz")
    for p in PARAMETERS:
        points[p]=points[p].round(12)
    identities=pd.read_csv(analysis/"tables/per_realization.csv",usecols=["dataset","architecture","normalization"])
    out.mkdir(parents=True);(out/"tables").mkdir()
    rng=np.random.default_rng(args.seed)
    weights={};conditions=[];components=[];per_seed=[];pool={};cache_sources=[]
    datasets=[f"{prefix}_{family}" for prefix in ("sign","norm") for family in FAMILIES]
    for dataset in datasets:
        print(f"Decomposing {dataset}",flush=True)
        experiment="normalization" if dataset.startswith("norm_") else "sign"
        family=dataset.removeprefix("norm_").removeprefix("sign_")
        pairs=identities[identities.dataset==dataset][["architecture","normalization"]].drop_duplicates().sort_values(["architecture","normalization"])
        for group_index,(_,pair) in enumerate(pairs.iterrows()):
            percent=float(re.search(r"(?:sign_test_og_cel|sign_test_er)([0-9.]+)$",pair.architecture).group(1))*100
            for metric in METRICS:
                path=analysis/"cache/matrices"/f"{dataset}_{group_index}_{metric}.npz"
                with np.load(path) as cached:
                    y=cached["values"].copy();seeds=cached["seeds"].copy();grid=cached["grid"].round(12)
                design=pd.DataFrame(grid,columns=PARAMETERS)
                assert np.array_equal(design.sort_values(list(PARAMETERS)).index,np.arange(96))
                levels=tuple(design[p].nunique() for p in PARAMETERS)
                assert levels==(4,3,4,2) and np.isfinite(y).all()
                g=points[(points.experiment==experiment)&(points.family==family)&(points.normalization==pair.normalization)&(points.metric==metric)&np.isclose(points.negative_percent,percent,rtol=0,atol=1e-8)].sort_values(list(PARAMETERS))
                assert len(g)==96
                np.testing.assert_allclose(grid,g[list(PARAMETERS)],rtol=0,atol=1e-11)
                np.testing.assert_allclose(y.mean(axis=0),g.performance,rtol=1e-10,atol=1e-10)
                assert (g.n_seeds==len(seeds)).all()
                key=tuple(seeds.tolist())
                if key not in weights:
                    weights[key]=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=args.bootstrap)/len(seeds)
                mean_surfaces=np.vstack([y.mean(axis=0),weights[key]@y])
                result=factorial_variability(mean_surfaces,levels)
                single=factorial_variability(y,levels)
                meta=dict(experiment=experiment,family=family,normalization=pair.normalization,metric=metric,negative_percent=percent,n_seeds=len(seeds))
                rows=summarize_boot(meta,effect_measures(result))
                for row in rows:
                    row["total_grid_variance"]=result["total_variance"][0]
                    row["interaction_pct"]=100*result["interaction_fraction"][0]
                conditions.extend(rows)
                for i,param in enumerate(PARAMETERS):
                    values=effect_measures(single)
                    per_seed.append(pd.DataFrame(dict(**meta,parameter=param,seed=seeds,**{name:array[:,i] for name,array in values.items()})))
                for j,subset in enumerate(result["subsets"]):
                    components.append(dict(**meta,component=" × ".join(PARAMETERS[i] for i in subset),order=len(subset),variance=result["component_variance"][0,j],fraction_pct=100*result["component_variance"][0,j]/result["total_variance"][0]))
                pool_key=(experiment,family,pair.normalization,metric)
                if pool_key not in pool:
                    pool[pool_key]=dict(variance=np.zeros_like(result["component_variance"]),total=np.zeros_like(result["total_variance"]),ranges=np.zeros_like(result["mean_matched_sweep_range"]),n_levels=0,subsets=result["subsets"],n_seeds=len(seeds))
                pool[pool_key]["variance"]+=result["component_variance"]
                pool[pool_key]["total"]+=result["total_variance"]
                pool[pool_key]["ranges"]+=result["mean_matched_sweep_range"]
                pool[pool_key]["n_levels"]+=1
                cache_sources.append(dict(path=str(path.relative_to(analysis)),sha256=digest(path),dataset=dataset,architecture=pair.architecture,normalization=pair.normalization,metric=metric,verified_against_plotted_surface=True))
    summaries=[]
    for (experiment,family,norm,metric),record in pool.items():
        v,total,subsets=record["variance"],record["total"],record["subsets"]
        main_effect=np.stack([v[:,subsets.index((i,))] for i in range(4)],axis=1)
        involving=np.stack([sum(v[:,j] for j,s in enumerate(subsets) if i in s) for i in range(4)],axis=1)
        allocated=np.stack([sum(v[:,j]/len(s) for j,s in enumerate(subsets) if i in s) for i in range(4)],axis=1)
        measures=dict(main_pct=100*main_effect/total[:,None],involving_pct=100*involving/total[:,None],allocated_pct=100*allocated/total[:,None],matched_sweep_rms_sd=np.sqrt(involving/record["n_levels"]),mean_matched_sweep_range=record["ranges"]/record["n_levels"])
        meta=dict(experiment=experiment,family=family,normalization=norm,metric=metric,n_ei_levels=record["n_levels"],n_seeds=record["n_seeds"])
        rows=summarize_boot(meta,measures)
        for row in rows:
            row["interaction_pct"]=100*(total[0]-main_effect[0].sum())/total[0]
            row["mean_within_ei_grid_variance"]=total[0]/record["n_levels"]
        summaries.extend(rows)
    conditions=pd.DataFrame(conditions);summary=pd.DataFrame(summaries)
    for name,frame in [("by_ei",conditions),("across_ei_summary",summary),("components",pd.DataFrame(components))]:
        frame.to_csv(out/"tables"/f"{name}.csv",index=False,float_format="%.15g")
    pd.concat(per_seed,ignore_index=True).to_csv(out/"tables/per_seed.csv.gz",index=False,float_format="%.15g",compression={"method":"gzip","compresslevel":1})
    for frame,groupkeys in [(conditions,["experiment","family","normalization","metric","negative_percent"]),(summary,["experiment","family","normalization","metric"])]:
        np.testing.assert_allclose(frame.groupby(groupkeys).allocated_pct.sum(),100,atol=1e-9)
        assert (frame.involving_pct>=frame.main_pct-1e-9).all()
        assert frame.involving_pct.between(-1e-9,100+1e-9).all()
    figures(conditions,summary,out)
    report(conditions,summary,out,args.bootstrap)
    manifest=dict(sources=provenance["sources"],cache_sources=cache_sources,bootstrap=args.bootstrap,seed=args.seed,confidence=.95,
        primary_estimand="Variance decomposition of seed-averaged performance surface, within each E/I level",
        seed_estimand="Exact decomposition of each paired seeded surface before averaging",
        weights="Uniform discrete grid levels; equal E/I design weights; across-E/I fractions are ratios of summed within-E/I component variances",
        allocation="All 15 orthogonal factorial components; each interaction split equally among its participants for allocated percentages; total involvement overlaps",
        bootstrap_design="Multinomial seed-cluster resampling; identical seed sets use the same draws across all conditions and metrics",
        grid={p:sorted(points[p].unique().tolist()) for p in PARAMETERS},
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/"util/hyperparameter_variability.py",ROOT/"tests/test_hyperparameter_variability.py"]},
        validation="PASS: all cached surfaces match plotted means; complete grid; exact variance additivity; allocated shares sum to 100%; total involvement bounds; five synthetic unit tests passed",
        output_format="PNG only",n_conditions=len(conditions)//4,n_seed_parameter_rows=sum(len(d) for d in per_seed))
    for source in provenance["sources"]:
        assert digest(ROOT/source["path"])==source["sha256"]
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(summary[(summary.experiment=="sign")&(summary.family=="cel_matched")][["metric","parameter",*MEASURES]].round(3).to_string(index=False))
    print(f"Saved PNGs and variability tables to {out}")


if __name__=="__main__":
    main()
