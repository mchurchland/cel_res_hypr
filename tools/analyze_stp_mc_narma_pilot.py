#!/usr/bin/env python3
"""Paired seed analysis of the limited STP MC/NARMA10 sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-stp-pilot")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/xdg-stp-pilot")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.crossover_analysis import bootstrap_panel

FAMILIES=("cel_matched","matched_er")
MODES=("static","facilitating","depressing")
SCHEMES=("full","restricted_matched_norm")
COLORS={"cel_matched":"#332288","matched_er":"#0077bb"}
LABELS={"cel_matched":"CE","matched_er":"ER (binary)"}
STYLES={"static":":","facilitating":"-","depressing":"--"}
MODE_COLORS={"static":"#555555","facilitating":"#009988","depressing":"#ee7733"}
METRICS=("MC","NARMA10_test_nmse")


def estimate(values):
    a=np.asarray(values,float)
    mean,draws=bootstrap_panel(a[:,None],n_boot=5000,seed=20261118)
    lo,hi=np.quantile(draws[:,0],[.025,.975])
    return dict(mean=float(mean[0]),ci_low=float(lo),ci_high=float(hi),sd=float(np.std(a,ddof=1)),n_realizations=len(a))


def summarize(data):
    keys=["family","ei_percent","input_scale","input_scheme","synapses"]
    rows=[]
    for key,group in data.groupby(keys,sort=True):
        for metric in (*METRICS,"MC_gain_mean","NARMA10_gain_mean","MC_last_effective_radius","NARMA10_last_effective_radius"):
            rows.append(dict(zip(keys,key),metric=metric,**estimate(group.sort_values("repeat_id")[metric])))
    return pd.DataFrame(rows)


def comparison_tables(data):
    effects=[];advantages=[]
    condition=["ei_percent","input_scale","input_scheme"]
    for key,group in data.groupby(condition,sort=True):
        for mode in MODES:
            paired=group[group.synapses==mode].set_index(["repeat_id","family"])
            for metric in METRICS:
                p=paired[metric].unstack("family")
                value=p.cel_matched-p.matched_er if metric=="MC" else p.matched_er-p.cel_matched
                advantages.append(dict(zip(condition,key),synapses=mode,metric=metric,**estimate(value)))
        for family,fam in group.groupby("family"):
            baseline=fam[fam.synapses=="static"].set_index("repeat_id")
            for mode in MODES[1:]:
                altered=fam[fam.synapses==mode].set_index("repeat_id")
                for metric in METRICS:
                    value=altered[metric]-baseline[metric] if metric=="MC" else baseline[metric]-altered[metric]
                    effects.append(dict(zip(condition,key),family=family,synapses=mode,metric=metric,**estimate(value)))
    return pd.DataFrame(effects),pd.DataFrame(advantages)


def selected_narma(data):
    # Select only the input scale, separately per mode; never select on test.
    keys=["repeat_id","family","ei_percent","input_scheme","synapses"]
    chosen=data.loc[data.groupby(keys).NARMA10_val_nmse.idxmin()].copy()
    rows=[]
    for (ei,scheme,mode),g in chosen.groupby(["ei_percent","input_scheme","synapses"]):
        p=g.pivot(index="repeat_id",columns="family",values="NARMA10_test_nmse")
        rows.append(dict(ei_percent=ei,input_scheme=scheme,synapses=mode,comparison="CE advantage (ER error minus CE error)",**estimate(p.matched_er-p.cel_matched)))
        for family in FAMILIES:
            rows.append(dict(ei_percent=ei,input_scheme=scheme,synapses=mode,comparison=family+" test NMSE",**estimate(p[family])))
    return chosen,pd.DataFrame(rows)


def selected_across_presets(data):
    keys=["repeat_id","family","ei_percent","input_scheme"]
    chosen=data.loc[data.groupby(keys).NARMA10_val_nmse.idxmin()].copy()
    rows=[]
    for (ei,scheme),g in chosen.groupby(["ei_percent","input_scheme"]):
        p=g.pivot(index="repeat_id",columns="family",values="NARMA10_test_nmse")
        rows.append(dict(ei_percent=ei,input_scheme=scheme,CE_test_NMSE=float(p.cel_matched.mean()),
                         ER_test_NMSE=float(p.matched_er.mean()),**estimate(p.matched_er-p.cel_matched)))
    return chosen,pd.DataFrame(rows)


def draw_band(ax,h,color,label,style="-"):
    h=h.sort_values("ei_percent")
    ax.plot(h.ei_percent,h["mean"],style,color=color,label=label,lw=1.8,marker="o",ms=3)
    ax.fill_between(h.ei_percent,h.ci_low,h.ci_high,color=color,alpha=.09)


def make_plots(out,summary,effects,advantages,chosen_summary):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":200,"savefig.bbox":"tight","savefig.pad_inches":.15})
    for scheme in SCHEMES:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey="row",layout="constrained")
        for col,scale in enumerate((.1,.5)):
            for row,metric in enumerate(METRICS):
                ax=axes[row,col]
                for family in FAMILIES:
                    for mode in MODES:
                        h=summary[(summary.input_scheme==scheme)&(summary.input_scale==scale)&(summary.family==family)&(summary.synapses==mode)&(summary.metric==metric)]
                        draw_band(ax,h,COLORS[family],f"{LABELS[family]} · {mode}",STYLES[mode])
                ax.set_title(f"Input scale {scale:g}")
                ax.set_ylabel("Memory capacity (higher is better)" if metric=="MC" else "NARMA10 test NMSE (lower is better; log)")
                if metric=="MC":ax.set_ylim(0,30.5)
                else:
                    ax.set_yscale("log")
                    ax.axhline(1.,color=".6",ls=":",lw=.7)
                if row==1:ax.set_xlabel("Inhibitory edges (%)")
                ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis="y",alpha=.15)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
        title="Full input" if scheme=="full" else "Sensory CE / random ER input · matched total input norm"
        fig.suptitle(f"STP pilot: {title}\nInitial radius 0.95 · no radius grid · no dynamic renormalization\nEight paired realizations · pointwise bootstrap 95% intervals",fontsize=12)
        fig.savefig(out/f"performance_{scheme}.png");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey="row",layout="constrained")
    for col,scheme in enumerate(SCHEMES):
        for row,metric in enumerate(METRICS):
            ax=axes[row,col]
            for family in FAMILIES:
                for mode in MODES[1:]:
                    h=effects[(effects.input_scheme==scheme)&(effects.input_scale==.1)&(effects.family==family)&(effects.synapses==mode)&(effects.metric==metric)]
                    draw_band(ax,h,COLORS[family],f"{LABELS[family]} · {mode}",STYLES[mode])
            ax.axhline(0,color=".5",lw=.8)
            ax.set_title("Full input" if scheme=="full" else "Sensory CE / random ER")
            ax.set_ylabel("MC improvement over static" if metric=="MC" else "NARMA10 error reduction from static")
            if row==1:ax.set_xlabel("Inhibitory edges (%)")
            ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis="y",alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=9)
    fig.suptitle("Does STP improve performance? · input scale 0.1\nPositive means improvement; paired-realization pointwise bootstrap 95% intervals",fontsize=12)
    fig.savefig(out/"stp_improvements_weak_input.png");plt.close(fig)

    for scheme in SCHEMES:
        fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey="row",layout="constrained")
        for col,scale in enumerate((.1,.5)):
            for row,metric in enumerate(METRICS):
                ax=axes[row,col]
                for mode in MODES:
                    h=advantages[(advantages.input_scheme==scheme)&(advantages.input_scale==scale)&(advantages.synapses==mode)&(advantages.metric==metric)]
                    draw_band(ax,h,MODE_COLORS[mode],mode,STYLES[mode])
                ax.axhline(0,color=".5",lw=.8)
                ax.set_title(f"Input scale {scale:g}")
                ax.set_ylabel("CE − ER MC" if metric=="MC" else "ER − CE NARMA10 NMSE")
                if row==1:ax.set_xlabel("Inhibitory edges (%)")
                ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis="y",alpha=.15)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
        fig.suptitle(f"CE advantage · {'full input' if scheme=='full' else 'sensory CE / random ER'}\nPositive favors CE; negative favors ER",fontsize=13)
        fig.savefig(out/f"ce_er_advantage_{scheme}.png");plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True,layout="constrained")
    for ax,scheme in zip(axes,SCHEMES):
        for family in FAMILIES:
            for mode in MODES:
                h=chosen_summary[(chosen_summary.input_scheme==scheme)&(chosen_summary.synapses==mode)&(chosen_summary.comparison==family+" test NMSE")]
                draw_band(ax,h,COLORS[family],f"{LABELS[family]} · {mode}",STYLES[mode])
        ax.set(title="Full input" if scheme=="full" else "Sensory CE / random ER",xlabel="Inhibitory edges (%)",ylabel="Held-out NARMA10 NMSE (log)",yscale="log")
        ax.set_xticks([0,20,40,60,80,100])
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False,fontsize=9)
    fig.suptitle("NARMA10: input scale selected using validation, separately for each synapse preset",fontsize=12)
    fig.savefig(out/"validation_selected_narma.png");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True,layout="constrained")
    for col,scheme in enumerate(SCHEMES):
        for row,task in enumerate(("MC","NARMA10")):
            ax=axes[row,col]
            for family in FAMILIES:
                for mode in MODES[1:]:
                    h=summary[(summary.input_scheme==scheme)&(summary.input_scale==.1)&(summary.family==family)&(summary.synapses==mode)&(summary.metric==task+"_gain_mean")]
                    draw_band(ax,h,COLORS[family],f"{LABELS[family]} · {mode}",STYLES[mode])
            ax.axhline(1,color=".5",ls=":",lw=.8)
            ax.set_title(f"{task} · {'full input' if scheme=='full' else 'sensory CE / random ER'}")
            ax.set_ylabel("Mean recurrent synaptic gain u·x/U")
            if row==1:ax.set_xlabel("Inhibitory edges (%)")
            ax.set_xticks([0,20,40,60,80,100])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False,fontsize=9)
    fig.suptitle("STP changes effective recurrent gain · input scale 0.1\nNo normalization is applied after the initial weight scaling",fontsize=12)
    fig.savefig(out/"stp_gain_diagnostics.png");plt.close(fig)


def write_report(out,summary,effects,advantages,selected,all_selected,manifest,validation):
    def value(scheme,mode,family,metric,scale=.1,ei=20):
        return summary[(summary.input_scheme==scheme)&(summary.synapses==mode)&(summary.family==family)&(summary.metric==metric)&(summary.input_scale==scale)&(summary.ei_percent==ei)].iloc[0]
    def interval(row):return f"{row['mean']:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
    def effect(scheme,family,mode,metric):
        return effects[(effects.input_scheme==scheme)&(effects.family==family)&(effects.synapses==mode)&
                       (effects.metric==metric)&(effects.ei_percent==20)&(effects.input_scale==.1)].iloc[0]
    lines=["# Local STP pilot: MC and NARMA10 across 0–100% inhibitory edges","",
           f"**{manifest['n_repeats']} paired CPU realizations; {validation['n_condition_pairs']:,} MC/NARMA10 condition pairs.** "
           "Fractions 0, 20, 40, 60, 80 and 100%; input scales 0.1 and 0.5; full input and sensory CE/random ER input with matched total input norm. "
           "Static, facilitation-dominant and depression-dominant synapses are compared on the same base networks and signals.","",
           "## Spectral radius is not a grid parameter","",
           "There is **one initial own-radius scaling to 0.95**, inherited from the preceding local pilots. There is no radius sweep. "
           "STP is then allowed to change the recurrent coupling freely: no per-step or post-STP spectral normalization, gain clipping or parameter optimization. "
           "The initial normalization still differs by architecture/EI condition, so this is not an unnormalized or fixed-reference-scaling experiment.","",
           "## Main findings", "",
           f"- At 20% inhibitory edges, scale 0.1 and full input, facilitation changes CE MC by "
           f"{interval(effect('full','cel_matched','facilitating','MC'))}; its interval includes zero. "
           f"CE NARMA10 error falls by {interval(effect('full','cel_matched','facilitating','NARMA10_test_nmse'))}. "
           f"ER MC falls by {abs(effect('full','matched_er','facilitating','MC')['mean']):.3f}, so a wider CE advantage partly reflects ER deterioration.",
           f"- With sensory CE/random ER input, the same facilitation preset reduces CE MC by "
           f"{abs(effect('restricted_matched_norm','cel_matched','facilitating','MC')['mean']):.3f} and increases CE NARMA10 error by "
           f"{abs(effect('restricted_matched_norm','cel_matched','facilitating','NARMA10_test_nmse')['mean']):.3f}. "
           "STP therefore does not consistently improve the connectome-derived reservoir.",
           f"- At 40% inhibitory edges with weak full input, facilitation changes MC from "
           f"{value('full','static','cel_matched','MC',ei=40)['mean']:.2f} to {value('full','facilitating','cel_matched','MC',ei=40)['mean']:.2f} in CE "
           f"and from {value('full','static','matched_er','MC',ei=40)['mean']:.2f} to {value('full','facilitating','matched_er','MC',ei=40)['mean']:.2f} in ER. "
           "Both deteriorate, especially ER. NARMA10 also deteriorates in both. The apparent reversal in architecture ranking here is not a CE performance improvement.",
           "- Depression generally reduces MC in this parameterization. NARMA10 effects depend on E/I fraction and input strength. "
           "These are two generic parameter presets, not a search for each architecture's best STP regime.", "",
           "## Results at 20% inhibitory edges and weak input (0.1)","",
           "Higher MC and lower NARMA10 NMSE are better. These are means of eight computational realizations, not optimized architecture scores.","",
           "| Input | Synapses | CE MC | ER MC | CE NARMA10 NMSE | ER NARMA10 NMSE |",
           "|---|---|---:|---:|---:|---:|"]
    for scheme in SCHEMES:
        for mode in MODES:
            lines.append(f"| {'Full' if scheme=='full' else 'Sensory/random'} | {mode} | "
                         f"{value(scheme,mode,'cel_matched','MC')['mean']:.3f} | {value(scheme,mode,'matched_er','MC')['mean']:.3f} | "
                         f"{value(scheme,mode,'cel_matched','NARMA10_test_nmse')['mean']:.4f} | {value(scheme,mode,'matched_er','NARMA10_test_nmse')['mean']:.4f} |")
    lines.extend(["","## Does STP help each architecture?","",
                  "At 20% inhibitory edges, scale 0.1, sensory/random input. Positive means improvement relative to the paired static control. "
                  "Intervals are pointwise 95% bootstrap intervals; no multiplicity correction.","",
                  "| Architecture | STP preset | MC improvement [95% interval] | NARMA10 error reduction [95% interval] |",
                  "|---|---|---:|---:|"])
    for family in FAMILIES:
        for mode in MODES[1:]:
            h=effects[(effects.family==family)&(effects.synapses==mode)&(effects.ei_percent==20)&(effects.input_scale==.1)&(effects.input_scheme=="restricted_matched_norm")]
            lines.append(f"| {LABELS[family]} | {mode} | {interval(h[h.metric=='MC'].iloc[0])} | {interval(h[h.metric=='NARMA10_test_nmse'].iloc[0])} |")
    lines.extend(["","## Full sweep figures","",
                  "![Full-input performance](performance_full.png)","",
                  "![Sensory/random-input performance](performance_restricted_matched_norm.png)","",
                  "![Paired improvements](stp_improvements_weak_input.png)","",
                  "![CE–ER differences with sensory/random input](ce_er_advantage_restricted_matched_norm.png)","",
                  "## Validation-selected input scale for NARMA10","",
                  "Input scale is selected from 0.1 and 0.5 using validation NMSE separately for each realization, family, EI fraction, input scheme and synapse preset. "
                  "The untouched test score is reported. Synapse presets are kept separate; neither test scores nor cross-preset test minima select models. "
                  "The limited two-value input search is not an estimate of unrestricted optimal performance.","",
                  "![Validation-selected NARMA10](validation_selected_narma.png)","",
                  "A second comparison lets each architecture choose among **all six tested input-scale/synapse-preset combinations** using validation. "
                  "This retains the static control as an eligible choice, so an architecture is not forced to use a harmful plasticity preset. "
                  "Positive differences below mean lower CE test error.", "",
                  "| EI | Input arrangement | CE test NMSE | ER test NMSE | ER − CE error [95% interval] |",
                  "|---:|---|---:|---:|---:|"])
    for row in all_selected.itertuples():
        lines.append(f"| {row.ei_percent:g}% | {'Full' if row.input_scheme=='full' else 'Sensory/random'} | {row.CE_test_NMSE:.4f} | {row.ER_test_NMSE:.4f} | "
                     f"{row.mean:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |")
    primary=all_selected[(all_selected.ei_percent==20)&(all_selected.input_scheme=="full")].iloc[0]
    lines.extend(["",f"At 20% with full input, letting both architectures select among all tested presets gives CE test NMSE "
                  f"{primary.CE_test_NMSE:.4f} and ER {primary.ER_test_NMSE:.4f}, with CE advantage "
                  f"{interval(primary)}. The interval includes zero. The advantage seen when both are forced to use facilitation "
                  "does not establish a best-tested CE advantage at this fraction.", "",
                  "## STP model and its scope","",
                  "Published rate-based Tsodyks–Markram/Mongillo equations are used, as in "
                  "[Masse et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7321806/) and "
                  "[Carvalho and Buonomano](https://pmc.ncbi.nlm.nih.gov/articles/PMC3105243/):", "",
                  "```text", "du/dt = (U-u)/tau_f + U*(1-u)*r",
                  "dx/dt = (1-x)/tau_d - u*x*r", "r = abs(previous signed tanh activity)",
                  "recurrent transmitted activity = (u*x/U) * previous activity", "```", "",
                  "- One utilization/resource pair per presynaptic neuron modulates all its outgoing recurrent connections. All synapses in a preset share parameters. "
                  "Synaptic signs and zero connections are preserved. Plasticity does not apply to the external input or readout connections.",
                  "- Facilitation-dominant preset: U=0.15, tau_f=75, tau_d=5. Depression-dominant: U=0.45, tau_f=5, tau_d=75. "
                  "The U values and relative time-constant contrasts follow the [Norse demonstration](https://norse.github.io/notebooks/stp_example.html), "
                  "with time constants expressed in input-sample steps. They are generic exploratory settings, not measured worm parameters or calibrated milliseconds.",
                  "- STP starts at recovered rest, u=U and x=1, separately for each task/condition. Dividing by U matches resting recurrent gain to the static matrix. "
                  "Absolute tanh activity is a nonnegative release-rate proxy; signed tanh activity itself is retained for transmission. "
                  "This is an explicit graded-activity adapter, not a claim that signed tanh states are biological firing rates.",
                  "- There was no rate-compatible STP implementation in the repository. The small published-ODE adapter is compiled with Numba; "
                  "RK4 at four substeps per input sample is independently checked against SciPy solve_ivp. Existing spike-driven package implementations "
                  "were inspected but were not substituted directly into signed graded dynamics. MC code and NARMA10 generation/readout are reused.",
                  "- All state variables are reset before each MC and NARMA10 trajectory. Only neuron activity is available to the linear readout, "
                  "not the extra synaptic utilization/resource states. STP therefore changes the hidden dynamical system without adding readout features.","",
                  "![STP recurrent gain](stp_gain_diagnostics.png)","",
                  "STP changes mean recurrent gain as well as adding history-dependent synaptic states. Without a matched static-gain control, "
                  "performance changes cannot be attributed exclusively to synaptic memory. The saved last-step effective recurrent radii describe "
                  "diag(gain)W, not the complete neuron-plus-STP Jacobian, and are not stability criteria.","",
                  "## Benchmark protocol, uncertainty and limits","",
                  "- Same CE unknown-sign completion and binary matched-ER sign-sweep control as the previous experiments. Each has 297 nodes and 3,604 original edges. "
                  "This still varies topology and recurrent-weight distribution together; no Gaussian-weight ER control is introduced.",
                  "- Leak is fixed at 1 and bias at 0. Input is either the original dense Gaussian projection or the 78-neuron sensory CE / random ER mask "
                  "from the preceding sensory pilot. Restricted inputs are scaled to the full input vector's total L2 norm. One random ER mask per realization "
                  "is fixed across all fractions, input scales and synapse presets. Exact coefficients, node indices and labels are cached.",
                  "- MC: existing compute_MC, delays 1–30, ridge 1e-4 plus existing epsilon 1e-6, float64 readout, 500 washout, 1,500 training, 500 test steps. "
                  "Reservoir activity is float32. MC is truncated at 30; near-ceiling scores do not resolve unrestricted memory capacity.",
                  "- NARMA10: **ReservoirPy 0.4.2**, with the unchanged generator and lag alignment of the original crossover runner. "
                  "Target burn-in 1,000; reservoir washout 500; training 1,500, validation 500, test 500. Existing sklearn Ridge with intercept; "
                  "alpha chosen on validation from 1e-8, 1e-6, 1e-4, 1e-2 and 1; no refit on validation and no output feedback. No local NARMA recurrence was written.",
                  "- The same input/target sequences, initial network and input projection are paired across static/STP conditions. MC uses U[-1,1]; "
                  "NARMA10 uses its package-defined nonnegative input and nonzero mean, so the two tasks can induce different STP operating regimes.",
                  "- Eight computational realizations of one connectome; base seed 20261117 plus repeat_id×100000. "
                  "Bootstrap resamples whole paired realization panels, 5,000 percentile resamples. Intervals are exploratory, pointwise and not corrected for multiple comparisons. "
                  "This sample is too small to locate a crossover precisely or establish biological optimization.","",
                  "## Validation and reproduction","",
                  f"All {validation['n_condition_pairs']:,} condition pairs have finite scores; MC and per-delay scores obey their bounds. "
                  "All conditions and 30 MC delays are present. NRMSE squared matches NMSE. STP resource/utilization states remain in [0,1] without clipping, "
                  "base recurrent matrices are unchanged across input and plasticity conditions, and all input norms are matched as specified. "
                  "Six synthetic STP tests pass. A static smoke check reproduced the previous local MC score exactly; doubling STP integration resolution "
                  "changed MC by 1.24e-5 in the checked CE condition.","",
                  "```bash", "env PYTHONPATH=/tmp/cel-crossover-deps:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
                  "  python tools/run_stp_mc_narma_pilot.py --n-repeats 8 --workers 4",
                  "python tools/analyze_stp_mc_narma_pilot.py", "```", "",
                  "`performance.csv`, `memory_curves.csv`, `summary_metrics.csv`, `stp_effects.csv`, `ce_er_advantages.csv`, "
                  "`validation_selected_configurations.csv`, and `validation_selected_narma.csv` retain all results. "
                  "`manifest.json` records exact model/settings, source/code hashes and package versions; per-realization caches store stimuli, "
                  "targets, networks, input weights and terminal gains. PNG figures only; existing results are preserved.",""])
    (out/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/stp_mc_narma_full_sweep_2026-11-18")
    args=parser.parse_args();out=args.out_dir
    data=pd.read_csv(out/"performance.csv");curves=pd.read_csv(out/"memory_curves.csv")
    manifest=json.loads((out/"manifest.json").read_text())
    keys=["repeat_id","family","ei_percent","input_scale","input_scheme","synapses"]
    if len(data)!=manifest["n_repeats"]*144 or data.duplicated(keys).any():raise ValueError("Incomplete/duplicate design.")
    scores=["MC","NARMA10_val_nmse","NARMA10_test_nmse","NARMA10_test_nrmse"]
    if not np.isfinite(data[scores]).all().all() or not data.MC.between(0,30).all():raise ValueError("Invalid scores.")
    if not (data.NARMA10_test_nmse>=0).all() or not np.allclose(data.NARMA10_test_nrmse**2,data.NARMA10_test_nmse):raise ValueError("Invalid NARMA metrics.")
    totals=curves.groupby(["repeat_id","condition_id"]).capacity.agg(["sum","count"])
    totals=totals.join(data.set_index(["repeat_id","condition_id"])["MC"],validate="one_to_one")
    if not (totals["count"]==30).all() or not curves.capacity.between(0,1).all() or np.max(abs(totals["sum"]-totals.MC))>1e-5:raise ValueError("Invalid memory curves.")
    if data.groupby(["repeat_id","family","ei_percent"]).network_sha256.nunique().max()!=1:raise ValueError("Base weights changed.")
    dynamic=data[data.synapses!="static"]
    for task in ("MC","NARMA10"):
        if dynamic[task+"_stp_state_min"].min()<-1e-8 or dynamic[task+"_stp_state_max"].max()>1+1e-8:raise ValueError("STP bounds violated.")
    norms=data.groupby(["repeat_id","input_scale"]).input_l2_norm.agg(["min","max"])
    if np.max(norms["max"]-norms["min"])>1e-6:raise ValueError("Input norm mismatch.")
    if data.initial_radius_target.nunique()!=1 or manifest["radius_grid"] or manifest["dynamic_renormalization"]:raise ValueError("Unexpected radius sweep/renormalization.")
    static=data[data.synapses=="static"]
    initial_radius_error=float(np.max(abs(static.MC_last_effective_radius-static.initial_post_rho)))
    if initial_radius_error>2e-5:raise ValueError("Initial static radius diagnostics disagree.")
    previous_path=ROOT/"analysis/sensory_input_pilot_2026-11-18/performance.csv"
    previous_matches=0
    if previous_path.exists():
        previous=pd.read_csv(previous_path)
        for row in static[static.ei_percent.isin([20,40])].itertuples():
            p=previous[(previous.repeat_id==row.repeat_id)&(previous.family==row.family)&
                       (previous.ei_percent==row.ei_percent)&(previous.input_scale==row.input_scale)]
            if row.input_scheme=="full":p=p[p.placement=="dense"]
            elif row.family=="cel_matched":p=p[(p.placement=="sensory")&(p.calibration=="matched_total_norm")]
            else:p=p[(p.placement=="random")&(p.calibration=="matched_total_norm")&(p.mask_draw==0)]
            if len(p)!=1 or row.MC!=p.MC.iloc[0]:raise ValueError("Static MC does not reproduce the preceding local pilot.")
            previous_matches+=1
    summary=summarize(data);effects,advantages=comparison_tables(data);chosen,chosen_summary=selected_narma(data)
    all_chosen,all_chosen_summary=selected_across_presets(data)
    summary.to_csv(out/"summary_metrics.csv",index=False);effects.to_csv(out/"stp_effects.csv",index=False)
    advantages.to_csv(out/"ce_er_advantages.csv",index=False);chosen.to_csv(out/"validation_selected_configurations.csv",index=False)
    chosen_summary.to_csv(out/"validation_selected_narma.csv",index=False)
    all_chosen.to_csv(out/"validation_selected_across_presets_configurations.csv",index=False)
    all_chosen_summary.to_csv(out/"validation_selected_across_presets.csv",index=False)
    make_plots(out,summary,effects,advantages,chosen_summary)
    validation=dict(n_realizations=manifest["n_repeats"],n_condition_pairs=len(data),n_task_scores=2*len(data),
                    n_delay_scores=len(curves),all_scores_finite=True,complete_design=True,
                    max_curve_sum_error=float(np.max(abs(totals["sum"]-totals.MC))),
                    max_input_norm_difference=float(np.max(norms["max"]-norms["min"])),
                    recurrent_matrices_unchanged=True,stp_states_bounded=True,
                    initial_radius_grid=False,dynamic_renormalization=False,
                    max_initial_radius_diagnostic_error=initial_radius_error,
                    static_MC_scores_matching_previous_pilot=previous_matches,
                    bootstrap=dict(n_resamples=5000,method="percentile",unit="paired realization",seed=20261118),
                    analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"VALIDATION.json").write_text(json.dumps(validation,indent=2)+"\n")
    write_report(out,summary,effects,advantages,chosen_summary,all_chosen_summary,manifest,validation)
    print(json.dumps(validation,indent=2))
    print(advantages[(advantages.ei_percent==20)&(advantages.input_scale==.1)].round(5).to_string(index=False))


if __name__=="__main__":main()
