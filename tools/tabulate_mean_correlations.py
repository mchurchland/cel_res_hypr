#!/usr/bin/env python3
"""Export existing mean/robustness correlations as CSV, Markdown and PNG tables."""
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-robustness")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/robustness-xdg-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.robustness import spearman

MEASURES={
    "cv":"CV", "sd":"Standard deviation", "iqr":"Interquartile range", "mad":"Median absolute deviation",
    "minimum":"Minimum performance", "p10":"10th-percentile performance", "bottom10":"Bottom-decile mean",
    "range":"Performance range", "attain_q25":"Threshold attainment: main Q25", "attain_q50":"Threshold attainment: main Q50",
    "attain_q75":"Threshold attainment: main Q75", "retention80":"80% retention", "retention90":"90% retention",
    "sensitivity_mean":"Mean local sensitivity", "low_fraction":"Fraction below main Q25",
    "above_q25_sd":"SD among configurations ≥ main Q25", "feasible_fraction":"Variance / conditional feasible ceiling",
}
METRICS=["MC","IPC","KR"]


def export(frame,out,stem,title,subtitle):
    wide=frame.pivot(index="measure",columns="metric",values="rho").reindex(index=MEASURES,columns=METRICS)
    wide.index=[MEASURES[key] for key in wide.index]
    wide.index.name="Measure"
    wide.to_csv(out/f"{stem}.csv",float_format="%.9f")
    frame.to_csv(out/f"{stem}_details.csv",index=False,float_format="%.12g")
    lines=[f"# {title}\n",subtitle+"\n","| Measure | MC | IPC | KR |","| --- | ---: | ---: | ---: |"]
    for label,row in wide.iterrows():
        lines.append(f"| {label} | "+" | ".join(f"{v:.3f}" if np.isfinite(v) else "undefined" for v in row)+" |")
    (out/f"{stem}.md").write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(10,9.2))
    fig.subplots_adjust(left=.035,right=.965,top=.86,bottom=.045)
    ax.axis("off")
    fig.text(.035,.955,title,fontsize=15,weight="semibold",color="#172b44")
    fig.text(.035,.916,subtitle,fontsize=10,color="#526176")
    cells=[[label,*[f"{v:.3f}" if np.isfinite(v) else "—" for v in row]] for label,row in wide.iterrows()]
    table=ax.table(cellText=cells,colLabels=["Measure","MC","IPC","KR"],cellLoc="center",colWidths=[.61,.13,.13,.13],bbox=[0,.07,1,.93])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#dce3eb");cell.set_linewidth(.6)
        if r==0:
            cell.set_facecolor("#223b58");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f2f5f8" if r%2==0 else "white")
        if c==0:
            cell.get_text().set_ha("left");cell.PAD=.035
    fig.text(.035,.045,"Spearman ρ with mean performance; positive values indicate an increasing association.",fontsize=9,color="#526176")
    fig.text(.035,.019,"Thresholds are metric-specific and exploratory; ceiling-based diagnostics use conditional historical bounds.",fontsize=8.5,color="#526176")
    fig.savefig(out/f"{stem}.png",dpi=300,bbox_inches="tight",facecolor="white");plt.close(fig)
    return wide


def main():
    base=ROOT/"analysis/robustness_reevaluation_2026-11-16"
    out=base/"mean_correlation_tables"
    out.mkdir(exist_ok=True)
    path=base/"tables/associations.csv"
    stored=pd.read_csv(path)
    pooled=stored[(stored.dataset=="main")&(stored.scope=="pooled_realizations")&stored.measure.isin(MEASURES)].copy()
    assert len(pooled)==len(MEASURES)*3
    primary=export(pooled,out,"main_pooled_mean_correlations","Correlation of robustness measures with mean performance",
                   "Main architecture experiment · 10 architectures × 200 paired seeds · each surface has 96 configurations")
    per_path=base/"tables/per_realization.csv"
    per=pd.read_csv(per_path)
    ce=per[(per.dataset=="main")&(per.architecture=="real")]
    rows=[]
    for metric in METRICS:
        g=ce[ce.metric==metric]
        for measure in MEASURES:
            finite=np.isfinite(g["mean"])&np.isfinite(g[measure])
            old=stored[(stored.dataset=="main")&(stored.scope=="within_architecture")&(stored.architecture=="real")&(stored.metric==metric)&(stored.measure==measure)]
            row=dict(metric=metric,measure=measure,rho=spearman(g["mean"],g[measure]),n=int(finite.sum()),n_clusters=int(g.loc[finite,"seed"].nunique()))
            # Retain existing cluster-bootstrap intervals where already computed.
            if len(old)==1:
                row.update(ci_low=old.iloc[0].ci_low,ci_high=old.iloc[0].ci_high)
                assert np.isclose(row['rho'],old.iloc[0].rho,atol=1e-9)
            rows.append(row)
    export(pd.DataFrame(rows),out,"ce_only_mean_correlations","C. elegans only: correlation with mean performance",
           "Main experiment · across 200 seeds within C. elegans · each surface has 96 configurations")
    # Retain all previously estimated experiment/mode panels without mixing them.
    all_pooled=stored[(stored.scope=="pooled_realizations")&stored.measure.isin(MEASURES)]
    all_pooled.to_csv(out/"all_experiment_correlations_details.csv",index=False,float_format="%.12g")
    dimensions=[]
    for dataset in ["main","shuf"]:
        for metric,g in per[per.dataset==dataset].groupby("metric"):
            for name in ["maximum","variance",*[c for c in per if c.startswith("sensitivity_") and c!="sensitivity_mean"]]:
                dimensions.append(dict(dataset=dataset,metric=metric,measure=name,rho=spearman(g["mean"],g[name]),n=int((np.isfinite(g["mean"])&np.isfinite(g[name])).sum())))
    pd.DataFrame(dimensions).to_csv(out/"additional_per_parameter_correlations.csv",index=False,float_format="%.12g")
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,per_path,Path(__file__)]}
    manifest=dict(sources=sources,correlation="Spearman rank correlation; finite pairs only",main_scope="Pooled main architecture/seed observations, not within C. elegans alone",
                  ce_scope="Within C. elegans across seeds; mean and robustness each computed over its 96 configurations",
                  uncertainty="Existing bootstrap intervals retained in detail CSVs; newly tabulated CE-only and per-parameter point estimates do not introduce new intervals",
                  metrics=MEASURES,figure_format="PNG only",n_main_rows=len(primary),validation="PASS: 51 stored pooled coefficients; CE coefficients agree with stored within-architecture results where available")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(primary.to_string(float_format=lambda value:f"{value:.3f}"))
    print(f"Saved tables and PNGs to {out}")


if __name__=="__main__":
    main()
