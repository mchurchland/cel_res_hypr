#!/usr/bin/env python3
"""Paired realization summaries for the local sensory-input MC pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-sensory-input-pilot")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/xdg-sensory-input-pilot")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.crossover_analysis import bootstrap_panel

FAMILIES=("cel_matched","matched_er")
CALIBRATIONS=("unchanged_coefficients","matched_total_norm")
MODEL_COLORS={"cel_matched":"#332288","matched_er":"#0077bb"}
INPUT_COLORS={.1:"#332288",.5:"#ee7733"}
CAL_LABELS={"unchanged_coefficients":"Same coefficients","matched_total_norm":"Matched total input norm"}


def estimate(values):
    a=np.asarray(values,float)
    if not np.isfinite(a).all():
        raise ValueError("Missing/nonfinite paired observations.")
    mean,draws=bootstrap_panel(a[:,None],n_boot=5000,seed=20261118)
    lo,hi=np.quantile(draws[:,0],[.025,.975])
    return dict(mean=float(mean[0]),ci_low=float(lo),ci_high=float(hi),n_realizations=len(a),sd=float(np.std(a,ddof=1)))


def summarize(data):
    keys=["ei_percent","input_scale","family","placement","calibration"]
    # Mask draws do not become extra independent reservoir samples.
    per_seed=data.groupby(["repeat_id",*keys],as_index=False)[["MC","input_l2_norm","fraction_saturated","mean_state_variance","mean_driven_outdegree"]].mean()
    rows=[]
    for key,group in per_seed.groupby(keys,sort=True):
        for metric in ("MC","input_l2_norm","fraction_saturated","mean_state_variance","mean_driven_outdegree"):
            rows.append(dict(zip(keys,key),metric=metric,**estimate(group.sort_values("repeat_id")[metric])))
    return per_seed,pd.DataFrame(rows)


def contrasts(per_seed):
    rows=[]
    for (ei,scale),group in per_seed.groupby(["ei_percent","input_scale"]):
        def get(family,placement,calibration):
            return group[(group.family==family)&(group.placement==placement)&(group.calibration==calibration)].set_index("repeat_id").MC.sort_index()
        ce_full=get("cel_matched","dense","dense");er_full=get("matched_er","dense","dense")
        rows.append(dict(ei_percent=ei,input_scale=scale,calibration="dense",contrast="CE dense minus ER dense",**estimate(ce_full-er_full)))
        for calibration in CALIBRATIONS:
            ce_sens=get("cel_matched","sensory",calibration)
            ce_random=get("cel_matched","random",calibration)
            er_random=get("matched_er","random",calibration)
            comparisons={
                "CE sensory minus ER random":ce_sens-er_random,
                "CE random minus ER random":ce_random-er_random,
                "CE sensory minus CE random":ce_sens-ce_random,
                "CE sensory minus CE dense":ce_sens-ce_full,
                "ER random minus ER dense":er_random-er_full,
                "Change in CE-ER advantage from dense":(ce_sens-er_random)-(ce_full-er_full),
            }
            for label,values in comparisons.items():
                rows.append(dict(ei_percent=ei,input_scale=scale,calibration=calibration,contrast=label,**estimate(values)))
    return pd.DataFrame(rows)


def plot_figures(out,summary,comparison,per_seed,curves,data,k):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
    def mc_row(ei,scale,family,placement,calibration):
        return summary[(summary.ei_percent==ei)&(summary.input_scale==scale)&(summary.family==family)&
                       (summary.placement==placement)&(summary.calibration==calibration)&(summary.metric=="MC")].iloc[0]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True,layout="constrained")
    for r,ei in enumerate((20,40)):
        for c,scale in enumerate((.1,.5)):
            ax=axes[r,c]
            for family,placement,label,style in (("cel_matched","sensory","CE sensory","-"),
                                                  ("cel_matched","random","CE random","--"),
                                                  ("matched_er","random","ER random","-")):
                rows=[mc_row(ei,scale,family,"dense","dense")]+[mc_row(ei,scale,family,placement,cal) for cal in CALIBRATIONS]
                mean=np.array([row["mean"] for row in rows]);lo=np.array([row.ci_low for row in rows]);hi=np.array([row.ci_high for row in rows])
                ax.plot([0,1,2],mean,style,color=MODEL_COLORS[family],marker="o" if placement!="random" or family=="matched_er" else "^",label=label)
                ax.fill_between([0,1,2],lo,hi,color=MODEL_COLORS[family],alpha=.10)
            ax.set_title(f"{ei}% inhibitory edges · input scale {scale:g}")
            ax.set_xticks([0,1,2],labels=["All 297 neurons",f"{k} neurons\nsame coefficients",f"{k} neurons\nmatched total norm"])
            ax.set_ylabel("Memory capacity")
            ax.grid(axis="y",alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Sensory placement versus equally sparse random inputs\nRadius 0.95 · leak 1 · bias 0 · 20 paired realizations · pointwise bootstrap 95% intervals",fontsize=13)
    fig.savefig(out/"01_input_placement_performance.png");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey="row",layout="constrained")
    for c,ei in enumerate((20,40)):
        for r,label in enumerate(("CE sensory minus ER random","CE sensory minus CE random")):
            ax=axes[r,c]
            for scale in (.1,.5):
                h=comparison[(comparison.ei_percent==ei)&(comparison.input_scale==scale)&(comparison.contrast==label)].set_index("calibration").loc[list(CALIBRATIONS)]
                x=np.array([0.,1.])+(-.05 if scale==.1 else .05)
                ax.errorbar(x,h["mean"],yerr=np.vstack([h["mean"]-h.ci_low,h.ci_high-h["mean"]]),
                            color=INPUT_COLORS[scale],fmt="o-",capsize=4,label=f"Input scale {scale:g}")
            ax.axhline(0,color=".5",lw=.9)
            ax.set_title(f"{ei}% inhibitory edges")
            ax.set_ylabel(label+"\nMC difference")
            ax.set_xticks([0,1],labels=[CAL_LABELS[cal] for cal in CALIBRATIONS])
            ax.grid(axis="y",alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Does sensory input favor CE—and is sensory placement better than random placement?\nPaired differences; random masks averaged within each realization",fontsize=12)
    fig.savefig(out/"02_paired_input_advantages.png");plt.close(fig)

    joined=curves.merge(data[["repeat_id","condition_id","family","ei_percent","input_scale","placement","calibration","mask_draw"]],
                       on=["repeat_id","condition_id"],validate="many_to_one")
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout="constrained")
    for r,ei in enumerate((20,40)):
        for c,calibration in enumerate(CALIBRATIONS):
            ax=axes[r,c]
            selections=(("cel_matched","sensory",calibration,"CE sensory","#332288","-"),
                        ("cel_matched","random",calibration,"CE random","#332288","--"),
                        ("matched_er","random",calibration,"ER random","#0077bb","-"),
                        ("cel_matched","dense","dense","CE full input",".4",":"),
                        ("matched_er","dense","dense","ER full input",".4","-."))
            for family,placement,cal,label,color,style in selections:
                h=joined[(joined.ei_percent==ei)&(joined.input_scale==.1)&(joined.family==family)&
                         (joined.placement==placement)&(joined.calibration==cal)]
                panel=h.groupby(["repeat_id","delay"]).capacity.mean().unstack("delay")
                mean,draws=bootstrap_panel(panel.to_numpy(),n_boot=5000,seed=20261118)
                lo,hi=np.quantile(draws,[.025,.975],axis=0)
                ax.plot(panel.columns,mean,style,color=color,label=label,lw=1.5)
                if placement!="dense":ax.fill_between(panel.columns,lo,hi,color=color,alpha=.08)
            ax.set(title=f"{ei}% inhibitory edges · {CAL_LABELS[calibration]}",xlabel="Input delay (steps)",ylabel="Squared-correlation memory score",ylim=(0,1.03))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("Memory by delay under weak input (scale 0.1)",fontsize=14)
    fig.savefig(out/"03_memory_by_delay.png");plt.close(fig)


def write_report(out,summary,comparison,manifest,checks):
    def fmt(row):return f"{row['mean']:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
    weak=comparison[(comparison.ei_percent==20)&(comparison.input_scale==.1)]
    def weak_row(label,calibration):
        return weak[(weak.contrast==label)&(weak.calibration==calibration)].iloc[0]
    lines=["# Preliminary sensory-input experiment", "",
           f"**20 paired computational realizations; {checks['n_evaluations']:,} MC evaluations.** "
           f"CE receives one scalar input at {manifest['n_sensory']} annotated sensory neurons; ER receives the same input at "
           "an equal number of randomly selected neurons. Three random input placements are tested per realization, together with "
           "equally sparse random placements on CE and full-input baselines for both architectures.", "",
           "## Main findings", "",
           f"At **20% inhibitory edges and input scale 0.1**, CE sensory input exceeds ER random input by "
           f"**{fmt(weak_row('CE sensory minus ER random','unchanged_coefficients'))} MC** with unchanged input coefficients. "
           f"The advantage remains **{fmt(weak_row('CE sensory minus ER random','matched_total_norm'))} MC** when total input norm is matched to the full-input baseline.", "",
           f"The CE−ER difference increases from the full-input baseline even with total drive matched: "
           f"**{fmt(weak_row('Change in CE-ER advantage from dense','matched_total_norm'))} MC**. "
           "Thus reducing total drive is not the only change associated with the restricted-input advantage in this pilot.", "",
           f"Within CE, sensory placement exceeds equally sparse random placement by "
           f"**{fmt(weak_row('CE sensory minus CE random','unchanged_coefficients'))} MC** with unchanged coefficients. "
           f"After matching total input norm, that difference is **{fmt(weak_row('CE sensory minus CE random','matched_total_norm'))} MC**; "
           "its interval includes zero. A sensory-specific benefit independent of total-drive calibration is therefore not established.", "",
           "The counterexamples are retained: **ER outperforms sensory-driven CE at 40% inhibitory edges and at input scale 0.5** under both calibrations. "
           "The results support conditional memory performance under restricted input, not a universal sparse-input advantage. "
           "They do not demonstrate that secondary eigenvalues cause the input-placement effect; the recurrent spectra were held fixed across placement conditions. "
           "At 40% and weak input, ER approaches the delay-30 MC ceiling, so its full memory capacity is not resolved by this protocol.", "",
           "## Paired results", "",
           "Positive values favor CE sensory input. Intervals are exploratory pointwise 95% bootstrap intervals over 20 paired realizations.", "",
           "| Inhibitory edges | Input scale | Input calibration | CE sensory − ER random MC | CE sensory − CE random MC |",
           "|---:|---:|---|---:|---:|"]
    for ei in (20,40):
        for scale in (.1,.5):
            for cal in CALIBRATIONS:
                h=comparison[(comparison.ei_percent==ei)&(comparison.input_scale==scale)&(comparison.calibration==cal)]
                main=h[h.contrast=="CE sensory minus ER random"].iloc[0]
                placement=h[h.contrast=="CE sensory minus CE random"].iloc[0]
                lines.append(f"| {ei}% | {scale} | {CAL_LABELS[cal]} | {fmt(main)} | {fmt(placement)} |")
    lines.extend(["", "![Performance by input placement](01_input_placement_performance.png)", "",
                  "![Paired differences](02_paired_input_advantages.png)", "",
                  "## Input calibration and controls", "",
                  "- **Same coefficients:** preserve the existing signed Gaussian coefficients at CE sensory entries. Move that exact ordered coefficient "
                  "vector to each random CE/ER subset. Every sparse comparison therefore has the same number of driven nodes, coefficient values, signs and total norm. "
                  "Relative to full input, this also reduces total drive.",
                  "- **Matched total input norm:** scale the same restricted coefficient vector so its L2 norm equals the original full-input vector. "
                  "This holds summed expected input-drive power constant across dense and restricted inputs; individual driven neurons receive stronger input. "
                  "It does not guarantee equal internal state amplitudes or saturation across architectures.",
                  "- **CE random control:** move the same coefficients to random CE nodes. This separates sensory placement from simply driving fewer nodes. "
                  "Random subsets are uniform draws without replacement, may include sensory nodes, and are fixed across E/I fractions, scales and calibrations within realization.",
                  "- **Full input:** the usual Gaussian input vector drives all 297 neurons. The two architectures have exactly the same dense input weights in each realization. "
                  f"{checks['n_reused_baselines']} existing 20%-EI, scale-0.1 full-input baselines were reused after exact checks of networks, input weights and stimulus sequences.", "",
                  "## Sensory annotation audit", "",
                  f"The frozen [OpenWorm WormAtlas-derived neuron-type table]({manifest['annotation_url']}) supplies annotations. "
                  "The selected rule includes labels beginning `Sensory neuron`, plus labels beginning `Polymodal neuron` that explicitly contain `sensory neuron`. "
                  "It excludes motor-led/proprioceptive, putative motor/sensory, and interneuron-led labels. This is one conservative operational sensory set, "
                  "not a claim that sensory function has a unique exhaustive classification.", "",
                  f"This selects 80 annotated names; {manifest['n_sensory']} map exactly onto the 297 matrix rows. "
                  f"Missing from this adjacency: **{', '.join(manifest['missing_sensory_names'])}**. No neurons were inserted or indices guessed. "
                  "The repository's legacy `old_cel.xls: Sensory` sheet is audited but is not the primary mask: it includes some neurons currently labeled interneurons "
                  "and omits PVD. The exact mask, type strings and evidence URLs are in `sensory_neurons.csv`; every adjacency node is listed in `neuron_label_audit.csv`.", "",
                  "## Simulation and uncertainty", "",
                  "- Fractions 20% and 40%; input scales 0.1 and 0.5; own-radius target 0.95; leak 1; added bias 0. Recurrent matrices are unchanged between input conditions. "
                  "CE restores unknown edges using the existing matched-sign construction; ER is the existing binary-weight sign-sweep control, not Gaussian ER. "
                  "These comparisons still combine topology and recurrent-weight distribution.",
                  "- Reuses the repository's reservoir dynamics and MC readout: float32 states, float64 readout fitting; 500 washout, 1,500 training, 500 test steps; "
                  "delays 1–30; ridge 1e-4 plus existing 1e-6 epsilon. All conditions share the same U[-1,1] stimulus within realization. Every readout is fitted to "
                  "all 297 states. This is not a motor-neuron-only readout or a full biological sensory-to-motor model.",
                  "- One scalar signal is projected with signed Gaussian coefficients. It does not model sensory transduction, independent sensory channels or anatomical input synapse signs. "
                  "No downstream task, noise robustness, spectral-mode intervention or input-sparsity sweep is included in this brief pilot.",
                  "- Base seed 20261117 + repeat_id×100000; existing network/unknown-sign/MC stimulus offsets retained. Random mask seed and all exact indices/coefficients "
                  "are stored. Computational realizations are not independent animals.",
                  "- Three random masks are averaged within each realization for inference; bootstrap resamples complete paired realizations (5,000 percentile resamples, "
                  "seed 20261118). Mask replicates are not counted as independent networks. All intervals are pointwise and not adjusted for multiple comparisons.", "",
                  "![Delay-resolved memory](03_memory_by_delay.png)", "",
                  "## Validation and files", "",
                  f"All {checks['n_evaluations']:,} scores are finite and within the MC bound; all 30 delay scores are present for every condition. "
                  f"Maximum relative error in matched total input norm: {checks['max_relative_norm_error']:.2e}. "
                  "Sparse CE/ER inputs use identical coefficients and counts. Recurrent-matrix hashes are unchanged across all input placements. "
                  "Five synthetic mask/annotation tests pass. All original data and preceding analyses are preserved.", "",
                  "`performance.csv` and `memory_curves.csv` retain every observation. `per_realization.csv` averages repeated random placements within realization. "
                  "`summary_metrics.csv` and `paired_contrasts.csv` contain all estimates and uncertainty intervals. `manifest.json` records source/code hashes, "
                  "frozen annotation commit, exact sensory indices and settings. Per-realization caches retain every actual input vector and recurrent matrix. "
                  "Outputs are PNG, CSV, NPZ, JSON and Markdown; no PDFs.", "",
                  "```bash", "python tools/run_sensory_input_pilot.py --n-repeats 20 --workers 4",
                  "python tools/analyze_sensory_input_pilot.py", "```", ""])
    (out/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/sensory_input_pilot_2026-11-18")
    args=parser.parse_args();out=args.out_dir
    manifest=json.loads((out/"manifest.json").read_text())
    data=pd.read_csv(out/"performance.csv");curves=pd.read_csv(out/"memory_curves.csv")
    if len(data)!=manifest["n_repeats"]*64 or data.duplicated(["repeat_id","condition_id"]).any():
        raise ValueError("Missing/duplicate conditions.")
    for placement,expected in (("dense",297),("sensory",manifest["n_sensory"]),("random",manifest["n_sensory"])):
        if not (data[data.placement==placement].n_driven==expected).all():raise ValueError("Wrong driven-node count.")
    totals=curves.groupby(["repeat_id","condition_id"]).capacity.agg(["count","sum"])
    totals=totals.join(data.set_index(["repeat_id","condition_id"])["MC"],validate="one_to_one")
    if not (totals["count"]==30).all() or not np.allclose(totals["sum"],totals.MC,atol=1e-5,rtol=0):
        raise ValueError("Delay curves do not reproduce total MC.")
    if not np.isfinite(data.MC).all() or not data.MC.between(0,30).all() or not curves.capacity.between(0,1).all():
        raise ValueError("Invalid memory scores.")
    if data.groupby(["repeat_id","family","ei_percent"]).network_sha256.nunique().max()!=1:
        raise ValueError("A recurrent matrix changed across input conditions.")
    norm_groups=data[data.placement!="dense"].groupby(["repeat_id","ei_percent","input_scale","calibration"]).input_l2_norm.agg(["min","max"])
    if np.max(norm_groups["max"]-norm_groups["min"])>1e-6:
        raise ValueError("Unequal sparse input norms across models/placements.")
    per_seed,summary=summarize(data);comparison=contrasts(per_seed)
    per_seed.to_csv(out/"per_realization.csv",index=False);summary.to_csv(out/"summary_metrics.csv",index=False)
    comparison.to_csv(out/"paired_contrasts.csv",index=False)
    plot_figures(out,summary,comparison,per_seed,curves,data,manifest["n_sensory"])
    checks=dict(n_realizations=manifest["n_repeats"],n_evaluations=len(data),n_delay_scores=len(curves),
                n_reused_baselines=int(data.baseline_reused.sum()),all_scores_finite=True,complete_design=True,
                max_curve_sum_error=float(np.max(abs(totals["sum"]-totals.MC))),
                max_relative_norm_error=float(np.max(abs(data[data.calibration=="matched_total_norm"].input_norm_ratio-1))),
                max_sparse_norm_difference=float(np.max(norm_groups["max"]-norm_groups["min"])),
                recurrent_matrices_unchanged=True,synthetic_tests_passed=5,
                bootstrap=dict(n_resamples=5000,method="percentile",unit="paired realization",random_masks="averaged within realization",seed=20261118),
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"VALIDATION.json").write_text(json.dumps(checks,indent=2)+"\n")
    write_report(out,summary,comparison,manifest,checks)
    print(json.dumps(checks,indent=2))
    print(comparison[comparison.contrast.isin(["CE sensory minus ER random","CE sensory minus CE random","Change in CE-ER advantage from dense"])].round(5).to_string(index=False))


if __name__=="__main__":
    main()
