"""Static, exportable robustness diagnostics using the thesis palette."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-robustness")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from util.robustness import PARAMETERS

METRICS = ("MC", "IPC", "KR")
LABELS = {"cv":"CV (lower dispersion)", "sd":"SD", "iqr":"IQR", "mad":"MAD",
          "bottom10":"Bottom-decile mean", "attain_q50":"Fraction ≥ main median threshold",
          "sensitivity_mean":"Mean local sensitivity", "range":"Performance range",
          "retention90":"90% of own maximum retention", "retention80":"80% retention",
          "minimum":"Minimum", "mean":"Mean performance"}
SHORT = {"real":"C. elegans", "cel+randN":"CE Gaussian", "er+randN":"ER Gaussian",
         "ws_p0.1+randN":"WS Gaussian", "local_sign":"Sign-pres. Gaussian",
         "local_sign+flat":"Sign-pres. uniform", "local_sign+binary":"Sign-pres. ±1",
         "global_sign_pres":"±1 + weight shuffle", "global_sign_pres_real_weight":"Real sign shuffle",
         "binary_base":"Binary", "shuffle":"Weight shuffle", "conn_shuf_only":"Connection shuffle",
         "celW+connShuf":"Connection + weight shuffle", "binary_base_topology_shuffle":"Binary + shuffle",
         "binary+shuffle":"±1 + connection shuffle", "binary+conshuffle+wshuffle":"±1 + conn./wt. shuffle"}


def color(arch):
    # Exactly graph_hist._architecture_sign_balance_color_map's thesis grouping.
    if arch in {"binary_base", "binary_base_topology_shuffle"}:
        return "#1100FF"
    if arch in {"cel+randN", "er+randN", "ws_p0.1+randN"}:
        return "#FF0000"
    return "#DD00FF"


def save(fig, out, stem, legend=None):
    if legend:
        # Reserve a physical gap below axis labels at every figure height.
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.8/fig.get_size_inches()[1]), ncol=min(5, len(legend)), frameon=False, fontsize=8)
    fig.savefig(out/f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out/f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def styles(architectures):
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
    return {a:(color(a), "*" if a == "real" else markers[i%len(markers)]) for i,a in enumerate(architectures)}


def scatter_comparisons(tables, out):
    measures = ["cv", "sd", "iqr", "mad", "bottom10", "attain_q50", "sensitivity_mean", "range"]
    for dataset in ("main", "shuf"):
        d = tables["per_realization"].query("dataset == @dataset")
        st = styles(d.architecture.unique())
        handles = [Line2D([], [], color=c, marker=m, ls="", label=SHORT.get(a,a)) for a,(c,m) in st.items()]
        fig, axes = plt.subplots(3, len(measures), figsize=(23, 8), layout="constrained")
        for i, metric in enumerate(METRICS):
            for j, measure in enumerate(measures):
                ax = axes[i,j]
                for arch, group in d[d.metric == metric].groupby("architecture"):
                    c,m = st[arch]
                    ax.scatter(group["mean"], group[measure], s=8, alpha=.16, color=c, marker=m, rasterized=True, linewidths=0)
                    ax.scatter(group["mean"].mean(), group[measure].mean(), s=50, color=c, marker=m, edgecolor="black", lw=.5)
                row = tables["associations"].query("dataset == @dataset and metric == @metric and measure == @measure and scope == 'pooled_realizations'")
                if len(row):
                    r = row.iloc[0]
                    ax.text(.03, .97, f"ρ={r.rho:.3f}\n95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]", transform=ax.transAxes, va="top", fontsize=7,
                            bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
                ax.set_xlabel(f"Mean {metric}")
                ax.set_ylabel(LABELS[measure])
        fig.suptitle(f"{dataset}: each small point is one seeded 96-configuration surface; large points are architecture means", fontsize=12)
        save(fig,out,f"{dataset}_mean_vs_robustness",handles)


def architecture_comparisons(tables, out):
    for dataset in ("main", "shuf"):
        d = tables["comparison"].query("dataset == @dataset")
        archs = list(d.architecture.unique())
        measures = ["mean", "cv", "sd", "iqr", "mad", "minimum", "bottom10", "attain_q50", "retention90", "sensitivity_mean"]
        fig, axes = plt.subplots(3, len(measures), figsize=(22, 11), layout="constrained")
        for i,metric in enumerate(METRICS):
            g = d[d.metric == metric].set_index("architecture").reindex(archs)
            for j,measure in enumerate(measures):
                ax = axes[i,j]
                x = np.arange(len(archs))
                ax.errorbar(g[measure], x, xerr=[np.maximum(0,g[measure]-g[f"{measure}_ci_low"]),np.maximum(0,g[f"{measure}_ci_high"]-g[measure])],
                            fmt="none", ecolor="#555555", elinewidth=.8)
                ax.scatter(g[measure], x, c=[color(a) for a in archs], s=21)
                ax.set_yticks(x, [SHORT.get(a,a) for a in archs] if j == 0 else [])
                ax.invert_yaxis()
                ax.set_xlabel(f"{metric}: {LABELS[measure]}", fontsize=8)
        fig.suptitle(f"{dataset}: mean per-realization robustness and 95% seed bootstrap intervals")
        save(fig,out,f"{dataset}_architecture_comparison")


def thresholds_and_retention(tables, settings, out):
    for dataset in ("main", "shuf"):
        d = tables["threshold_curves"].query("dataset == @dataset")
        st = styles(d.architecture.unique())
        handles = [Line2D([], [], color=c, marker=m, label=SHORT.get(a,a)) for a,(c,m) in st.items()]
        fig, axes = plt.subplots(1,3,figsize=(13,4),layout="constrained")
        for ax,metric in zip(axes,METRICS):
            for arch,g in d[d.metric == metric].groupby("architecture"):
                c,m=st[arch]
                ax.plot(g.threshold,g.attainment,color=c,marker=m,markevery=12,ms=4,lw=1)
                ax.fill_between(g.threshold,g.ci_low,g.ci_high,color=c,alpha=.05)
            for threshold in settings[metric]["anchors"].values():
                ax.axvline(threshold,ls=":",color="gray",lw=.6)
            ax.set(xlabel=f"Absolute {metric} threshold (exploratory)",ylabel="Fraction of configurations ≥ threshold",ylim=(-.02,1.02))
        fig.suptitle(f"{dataset}: common thresholds within each metric; shaded 95% seed intervals")
        save(fig,out,f"{dataset}_threshold_attainment",handles)
        d = tables["per_realization"].query("dataset == @dataset")
        fig, axes = plt.subplots(2,3,figsize=(12,6),layout="constrained")
        for j,metric in enumerate(METRICS):
            for i,measure in enumerate(("retention80","retention90")):
                for arch,g in d[d.metric == metric].groupby("architecture"):
                    c,m=st[arch]
                    axes[i,j].scatter(g["mean"],g[measure],s=9,alpha=.25,color=c,marker=m,rasterized=True)
                axes[i,j].set(xlabel=f"Mean {metric}",ylabel=LABELS.get(measure,measure),ylim=(-.02,1.02))
        fig.suptitle("Relative retention must be read alongside absolute performance")
        save(fig,out,f"{dataset}_retention_vs_absolute_performance",handles)


def sensitivities(tables,out):
    for dataset in ("main","shuf"):
        d=tables["comparison"].query("dataset == @dataset")
        archs=list(d.architecture.unique())
        fig,axes=plt.subplots(3,2,figsize=(12,11),layout="constrained")
        for i,metric in enumerate(METRICS):
            g=d[d.metric==metric].set_index("architecture").reindex(archs)
            for j,summary in enumerate(("mean","p90")):
                ax=axes[i,j]
                values=g[[f"sensitivity_{p}_{summary}" for p in PARAMETERS]].to_numpy()
                im=ax.imshow(values,aspect="auto",cmap="viridis")
                for (r,c),value in np.ndenumerate(values):
                    ax.text(c,r,f"{value:.2f}",ha="center",va="center",fontsize=7,color="white" if value<np.nanmax(values)*.55 else "black")
                ax.set_yticks(range(len(archs)),[SHORT.get(a,a) for a in archs])
                ax.set_xticks(range(4),["log radius","linear leak","log input","linear bias"])
                ax.set_title(f"{metric}: {summary} local sensitivity")
                fig.colorbar(im,ax=ax,shrink=.8,label="Performance / unit coordinate span")
        fig.suptitle("Only adjacent levels differing in one parameter; each coordinate scaled to [0,1]")
        save(fig,out,f"{dataset}_local_sensitivity")


def floor_and_variance(tables,out):
    per=tables["per_realization"]
    for dataset in ("main","shuf"):
        d=per[per.dataset==dataset]
        fig,axes=plt.subplots(3,4,figsize=(14,10),layout="constrained")
        for i,metric in enumerate(METRICS):
            for arch,g in d[d.metric==metric].groupby("architecture"):
                c=color(arch)
                axes[i,0].scatter(g.distance_lower,g.variance,s=8,alpha=.25,color=c,rasterized=True)
                axes[i,1].scatter(g.distance_upper,g.variance,s=8,alpha=.25,color=c,rasterized=True)
                axes[i,2].scatter(g["mean"],g.feasible_fraction,s=8,alpha=.25,color=c,rasterized=True)
                axes[i,3].scatter(g.low_fraction,g.sd,s=8,alpha=.25,color=c,rasterized=True)
            g=d[d.metric==metric].sort_values("mean")
            axes[i,0].plot(g.distance_lower,g.feasible_variance,color="black",lw=1,label="Feasible ceiling")
            axes[i,0].legend(fontsize=7)
            for ax,x,y in zip(axes[i], ["Distance of mean from lower bound","Distance of mean from upper bound",f"Mean {metric}","Fraction below pooled main Q25"], [f"{metric} variance",f"{metric} variance","Variance / feasible ceiling",f"{metric} SD"]):
                ax.set(xlabel=x,ylabel=y)
        fig.suptitle("Implementation-based bounds (historical protocol conditional); feasible ceiling is not expected variance")
        save(fig,out,f"{dataset}_floor_bounds")
    d=tables["variance_components"].query("dataset == 'main' and status == 'balanced'")
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout="constrained")
    for ax,metric in zip(axes,METRICS):
        g=d[d.metric==metric]
        bottom=np.zeros(len(g))
        for col,c,label in [("configuration_variance","#0072b2","Configuration"),("seed_variance","#e69f00","Seed"),("interaction_noise_variance","#009e73","Interaction + noise")]:
            v=g[col]/g.total_variance
            ax.bar(range(len(g)),v,bottom=bottom,color=c,label=label);bottom+=v.to_numpy()
        ax.set_xticks(range(len(g)),[SHORT.get(a,a) for a in g.architecture],rotation=90,fontsize=8)
        ax.set(title=metric,ylabel="Fraction of total finite-design variance",ylim=(0,1))
    axes[0].legend(fontsize=7)
    save(fig,out,"main_variance_partition")


def spectral(tables,out):
    p=tables["per_realization"].merge(tables["spectral_per_seed"],on=["dataset","architecture","normalization","seed"])
    p=p[p.dataset.str.startswith(("sign_","norm_"))].copy()
    p["negative_fraction"]=p.architecture.str.extract(r"(?:sign_test_og_cel|sign_test_er)([\d.]+)").astype(float)
    family_colors={"cel_matched":"#d55e00","cel_removed":"#0072b2","matched_er":"#009e73"}
    for prefix in ("sign_","norm_"):
        d=p[p.dataset.str.startswith(prefix)]
        fig,axes=plt.subplots(3,7,figsize=(22,9),layout="constrained")
        measures=["raw_rho","scale_factor","mean","sd","iqr","bottom10","retention90"]
        for i,metric in enumerate(METRICS):
            for dataset,g in d[d.metric==metric].groupby("dataset"):
                for norm,h in g.groupby("normalization"):
                    agg=h.groupby("negative_fraction")[measures].mean().sort_index()
                    c=family_colors[dataset.removeprefix(prefix)]
                    ls="-" if norm=="spectral_radius" else "--"
                    for j,measure in enumerate(measures):
                        axes[i,j].plot(agg.index,agg[measure],color=c,ls=ls,marker="o",ms=3,lw=1)
                        axes[i,j].set(xlabel="Negative-edge fraction",ylabel=f"{metric}: {LABELS.get(measure,measure)}")
        handles=[Line2D([],[],color=c,label=a.replace('_',' ')) for a,c in family_colors.items()]
        handles += [Line2D([],[],color="black",ls=ls,label=label) for ls,label in [("-","Own radius"),("--","Fixed reference")]] if prefix=="norm_" else []
        fig.suptitle(f"{prefix.rstrip('_')} experiments: full 96-point surfaces; KR vintage kept separate from main")
        save(fig,out,f"{prefix}spectral_balance_diagnostics",handles)
    d=p[p.dataset.str.startswith("norm_")]
    fig,axes=plt.subplots(3,6,figsize=(18,9),layout="constrained")
    for i,metric in enumerate(METRICS):
        for j,(x,y) in enumerate([(x,y) for x in ("raw_rho","scale_factor") for y in ("sd","bottom10","sensitivity_mean")]):
            ax=axes[i,j]
            for dataset,g in d[d.metric==metric].groupby("dataset"):
                for norm,h in g.groupby("normalization"):
                    c=family_colors[dataset.removeprefix("norm_")]
                    ax.scatter(h[x],h[y],s=4,alpha=.12,color=c,marker="o" if norm=="spectral_radius" else "x",rasterized=True)
            ax.set(xlabel=x,ylabel=f"{metric}: {LABELS.get(y,y)}")
    save(fig,out,"normalization_radius_scaling_robustness")


def make_plots(tables,settings,out):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"axes.grid":False})
    for function in (scatter_comparisons,architecture_comparisons):
        function(tables,out)
    thresholds_and_retention(tables,settings,out)
    for function in (sensitivities,floor_and_variance,spectral):
        function(tables,out)
    key_comparison(tables,out)


def key_comparison(tables,out):
    """A compact figure suitable for the main text; full definitions are supplements."""
    for dataset in ("main","shuf"):
        d=tables["per_realization"].query("dataset == @dataset")
        st=styles(d.architecture.unique())
        handles=[Line2D([],[],color=c,marker=m,ls="",label=SHORT.get(a,a)) for a,(c,m) in st.items()]
        fig,axes=plt.subplots(3,3,figsize=(8.5,7.5),layout="constrained")
        for i,metric in enumerate(METRICS):
            for j,measure in enumerate(("cv","sd","bottom10")):
                ax=axes[i,j]
                for arch,g in d[d.metric==metric].groupby("architecture"):
                    c,m=st[arch]
                    ax.scatter(g["mean"],g[measure],color=c,marker=m,s=5,alpha=.2,linewidths=0,rasterized=True)
                    ax.scatter(g["mean"].mean(),g[measure].mean(),color=c,marker=m,s=38,edgecolor="black",lw=.5)
                ax.set(xlabel=f"Mean {metric}",ylabel=LABELS[measure])
        fig.suptitle("Higher dispersion can coexist with better lower-tail capability",fontsize=11)
        save(fig,out,f"{dataset}_key_comparison",handles)
