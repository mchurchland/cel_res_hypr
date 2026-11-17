#!/usr/bin/env python3
"""Compare mean, CV and percentage of hyperparameter scores strictly below one."""
from pathlib import Path
import hashlib
import json
import os
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
from util.robustness import PARAMETERS,cluster_spearman

METRICS=("MC","IPC","KR")


def digest(path):
    with open(path,"rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def figure(stats,out,scope,stem,title):
    d=stats[(stats.dataset=="main")&(stats.scope==scope)]
    cells=[]
    for metric in METRICS:
        group=d[d.metric==metric].set_index("association")
        ref=group.iloc[0]
        values=[]
        for key in ("mean_vs_cv","pct_below_1_vs_cv","pct_below_1_vs_mean"):
            row=group.loc[key]
            values.append(f"{row.rho:.3f}\n[{row.ci_low:.3f}, {row.ci_high:.3f}]" if np.isfinite(row.rho) else "Undefined\n(all percentages = 0)")
        cells.append([metric,*values,f"{int(ref.n_surfaces_with_any_below_1)} / {int(ref.n)}",f"{ref.pct_scores_below_1:.4f}%"])
    fig,ax=plt.subplots(figsize=(13,4.7))
    fig.subplots_adjust(left=.03,right=.97,top=.79,bottom=.03);ax.axis("off")
    fig.text(.03,.94,title,fontsize=16,weight="semibold",color="#172b44")
    fig.text(.03,.86,"Each observation: mean, CV and percentage below 1 across a seeded 96-configuration grid",fontsize=10,color="#526176")
    table=ax.table(cellText=cells,colLabels=["Metric","Mean vs CV\nSpearman ρ [95% CI]","% below 1 vs CV\nSpearman ρ [95% CI]","% below 1 vs mean\nSpearman ρ [95% CI]","Grids containing\na score below 1","All scores\nbelow 1"],cellLoc="center",colWidths=[.07,.2,.23,.23,.15,.12],bbox=[0,.26,1,.71])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#dce3eb");cell.set_linewidth(.7)
        if r==0:
            cell.set_facecolor("#223b58");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f2f5f8" if r%2==0 else "white")
            if "Undefined" in cell.get_text().get_text():
                cell.get_text().set_color("#697586")
    fig.text(.03,.12,"Strict cutoff P < 1; performance equal to 1 is excluded. Intervals resample whole seed clusters (2,000 draws).",fontsize=9,color="#526176")
    fig.text(.03,.065,"A constant below-1 percentage has no defined correlation. The same cutoff represents different scales in MC, IPC and KR.",fontsize=9,color="#526176")
    fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight",facecolor="white");plt.close(fig)


def main():
    base=ROOT/"analysis/robustness_reevaluation_2026-11-16"
    out=base/"below_one_cv_correlations"
    if out.exists():
        raise FileExistsError("Output exists; preserve it and choose a new directory")
    manifest=json.loads((base/"manifest.json").read_text())
    per=pd.read_csv(base/"tables/per_realization.csv")
    parts=[]
    keys=["architecture","normalization","seed"]
    for source in manifest["settings"]["sources"]:
        path=ROOT/source["path"]
        assert digest(path)==source["sha256"]
        raw=pd.read_csv(path,usecols=["mode","normalization","seed",*PARAMETERS,*METRICS])
        raw["architecture"]=raw["mode"].str.replace(r"__norm_.*$","",regex=True)
        assert not raw.duplicated(keys+list(PARAMETERS)).any()
        for metric in METRICS:
            finite=np.isfinite(raw[metric])
            data=raw.assign(is_finite=finite,is_below_one=finite&(raw[metric]<1))
            counts=data.groupby(keys,as_index=False).agg(n_finite_scores=("is_finite","sum"),n_below_1=("is_below_one","sum"))
            counts["pct_below_1"]=100*counts.n_below_1/counts.n_finite_scores
            counts["dataset"],counts["metric"]=source["dataset"],metric
            parts.append(counts)
    counts=pd.concat(parts,ignore_index=True)
    joined=per.merge(counts,on=["dataset","metric",*keys],how="outer",validate="one_to_one",indicator=True)
    assert joined['_merge'].eq('both').all()
    joined=joined.drop(columns="_merge")
    assert joined.n_finite_scores.eq(96).all()
    assert joined.pct_below_1.between(0,100).all()
    rng=np.random.default_rng(20261117)
    results=[]
    groups=[]
    for key,g in joined.groupby(["dataset","normalization","metric"]):
        groups.append((dict(zip(["dataset","normalization","metric"],key),scope="pooled_architectures",architecture="ALL"),g))
        if key[0] in ("main","shuf"):
            for arch,h in g.groupby("architecture"):
                groups.append((dict(zip(["dataset","normalization","metric"],key),scope="within_architecture",architecture=arch),h))
    for meta,g in groups:
        totals=dict(n_scores=int(g.n_finite_scores.sum()),n_scores_below_1=int(g.n_below_1.sum()),
                    n_surfaces_with_any_below_1=int((g.n_below_1>0).sum()),pct_scores_below_1=100*g.n_below_1.sum()/g.n_finite_scores.sum())
        for association,x,y in [("mean_vs_cv","mean","cv"),("pct_below_1_vs_cv","pct_below_1","cv"),("pct_below_1_vs_mean","pct_below_1","mean")]:
            result=cluster_spearman(g,x,y,n_boot=2000,rng=rng)
            results.append(dict(**meta,association=association,**result,**totals,
                                status="estimated" if np.isfinite(result["rho"]) else "undefined: at least one variable is constant"))
    stats=pd.DataFrame(results)
    out.mkdir()
    stats.to_csv(out/"correlations.csv",index=False,float_format="%.12g")
    joined.to_csv(out/"per_realization_with_below_one.csv.gz",index=False,float_format="%.15g",compression={"method":"gzip","compresslevel":1})
    figure(stats,out,"pooled_architectures","main_pooled_below_one_cv","Main experiment: pooled architectures")
    ce=stats[(stats.dataset=="main")&(stats.architecture=="real")].copy()
    figure(ce,out,"within_architecture","ce_only_below_one_cv","C. elegans only: across seeded realizations")
    selected=stats[(stats.dataset=="main")&((stats.scope=="pooled_architectures")|(stats.architecture=="real"))]
    text=["# Mean performance, near-floor frequency and CV\n",
          "Fraction below one is calculated within each seeded 96-configuration surface, using only finite scores and the strict comparison P < 1. The percentage is then correlated with that surface's mean and CV. All correlations are Spearman rank correlations. Constant percentages produce undefined correlations, not zero correlations.\n",
          "| Scope | Metric | Mean–CV ρ | % below 1–CV ρ | % below 1–mean ρ | Grids with any score <1 | Scores <1 |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt=lambda x:f"{x:.3f}" if np.isfinite(x) else "undefined"
    for (scope,metric),g in selected.groupby(["scope","metric"]):
        rows=g.set_index("association");r=g.iloc[0]
        text.append(f"| {'C. elegans only' if scope=='within_architecture' else 'Pooled main'} | {metric} | {fmt(rows.loc['mean_vs_cv','rho'])} | {fmt(rows.loc['pct_below_1_vs_cv','rho'])} | {fmt(rows.loc['pct_below_1_vs_mean','rho'])} | {int(r.n_surfaces_with_any_below_1)}/{int(r.n)} | {int(r.n_scores_below_1)}/{int(r.n_scores)} |")
    text.append("\nThe cutoff is an absolute score of 1 in each metric's own units. It is not an equivalent distance from a performance floor across MC, IPC and KR. In the main experiment it is sparse for MC and absent for IPC/KR. These particular coefficients therefore cannot establish or exclude a general floor mechanism. Correlation with CV is also not independent of its mean denominator.\n")
    text.append("Intervals use 2,000 seed-cluster bootstrap replicates with RNG seed 20261117. Pooled panels retain every architecture sharing a resampled seed as one cluster. All architecture-specific comparisons are also exported. Repeated observations are computational realizations of one anatomical connectome, not independent worms. Only PNG figures are generated.\n")
    (out/"README.md").write_text("\n".join(text))
    meta=dict(sources=manifest["settings"]["sources"],threshold=1,comparison="strictly less than",denominator="finite configurations within each seeded surface",bootstrap=2000,seed=20261117,
              analysis_code_sha256=digest(Path(__file__)),figure_format="PNG only",n_surfaces=len(joined),validation="PASS: unique configuration identities; all surfaces have 96 finite scores; complete merge; percentages bounded; original source hashes verified")
    for source in meta['sources']:
        assert digest(ROOT/source['path'])==source['sha256']
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(selected[["scope","metric","association","rho","ci_low","ci_high","n_surfaces_with_any_below_1","n_scores_below_1","n_scores"]].round(4).to_string(index=False))
    print(f"Saved PNG tables and data to {out}")


if __name__=="__main__":
    main()
