#!/usr/bin/env python3
"""Summarize the local spectral intervention pilot with paired seed bootstrap."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-spectral-memory-pilot")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-spectral-memory-pilot")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.crossover_analysis import bootstrap_panel
from util.spectral_intervention import attenuate_blocks, real_schur_blocks, secondary_blocks

FAMILIES = ("cel_matched", "matched_er")
LABELS = {"cel_matched":"C. elegans", "matched_er":"ER (binary weights)"}
COLORS = {"cel_matched":"#332288", "matched_er":"#0077bb"}
STYLES = {"largest":"-", "random":"--", "smallest":":"}


def estimate(values):
    x = np.asarray(values, float)
    mean, draws = bootstrap_panel(x[:, None], n_boot=5000, seed=20261118)
    lo, hi = np.quantile(draws[:, 0], [.025,.975])
    return dict(mean=float(mean[0]), ci_low=float(lo), ci_high=float(hi),
                n_realizations=len(x), sd=float(np.std(x, ddof=1)))


def seed_means(group, column):
    # Random controls are repeated perturbations, not additional network samples.
    return group.groupby("repeat_id", sort=True)[column].mean()


def summarize(data):
    rows = []
    d = data.copy()
    d["dose"] = np.where(d.protocol=="equal_fraction", 1-d.multiplier,
                         np.where(d.protocol=="equal_frobenius", d.budget_fraction, 0.))
    keys = ["protocol", "family", "strategy", "requested_dimension", "dose"]
    for key, group in d.groupby(keys, sort=True):
        identifiers = dict(zip(keys,key))
        for metric in ("MC", "MC_loss", "delta_frobenius", "relative_delta_frobenius", "multiplier"):
            rows.append(dict(**identifiers, metric=metric, **estimate(seed_means(group,metric))))
    return d, pd.DataFrame(rows)


def contrast_tables(d):
    rows=[]
    baseline=d[d.protocol=="baseline"].pivot(index="repeat_id",columns="family",values="MC")
    rows.append(dict(contrast="CE minus ER MC", protocol="baseline", family="paired", strategy="baseline",
                     requested_dimension=0,dose=0.,**estimate(baseline.cel_matched-baseline.matched_er)))
    for (protocol,k,dose),group in d[d.protocol!="baseline"].groupby(["protocol","requested_dimension","dose"]):
        for family, famgroup in group.groupby("family"):
            target=seed_means(famgroup[famgroup.strategy=="largest"],"MC_loss")
            for strategy in ("random","smallest"):
                control=seed_means(famgroup[famgroup.strategy==strategy],"MC_loss")
                if len(control):
                    values=(target-control).dropna()
                    rows.append(dict(contrast="target loss minus control loss",protocol=protocol,family=family,
                                     strategy=strategy,requested_dimension=k,dose=dose,**estimate(values)))
        for strategy, sub in group.groupby("strategy"):
            byseed=sub.groupby(["repeat_id","family"]).MC.mean().unstack("family")
            rows.append(dict(contrast="CE minus ER MC",protocol=protocol,family="paired",strategy=strategy,
                             requested_dimension=k,dose=dose,**estimate(byseed.cel_matched-byseed.matched_er)))
            loss=sub.groupby(["repeat_id","family"]).MC_loss.mean().unstack("family")
            rows.append(dict(contrast="CE loss minus ER loss",protocol=protocol,family="paired",strategy=strategy,
                             requested_dimension=k,dose=dose,**estimate(loss.cel_matched-loss.matched_er)))
    return pd.DataFrame(rows)


def delay_band_losses(d,curves):
    joined=curves.merge(d[["condition_id","repeat_id","family","protocol","strategy","requested_dimension","dose"]],
                         on=["condition_id","repeat_id","family"],validate="many_to_one")
    rows=[]
    for low,high in ((1,5),(6,10),(11,20),(21,30)):
        section=joined[joined.delay.between(low,high)]
        baseline=section[section.protocol=="baseline"].groupby(["repeat_id","family"]).capacity.sum()
        target=section[(section.protocol=="equal_fraction")&(section.strategy=="largest")&(section.dose==1)]
        for (family,k),group in target.groupby(["family","requested_dimension"]):
            modified=group.groupby("repeat_id").capacity.sum()
            loss=baseline.xs(family,level="family")-modified
            rows.append(dict(family=family,requested_dimension=k,delay_low=low,delay_high=high,**estimate(loss)))
    return pd.DataFrame(rows)


def band(ax,x,sub,color,label,style="-"):
    h=sub.sort_values("dose")
    ax.plot(x,h["mean"],style,color=color,label=label,lw=1.8,marker="o",ms=4)
    ax.fill_between(x,h.ci_low,h.ci_high,color=color,alpha=.10)


def plots(out,d,summary,curves,contrasts):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"savefig.dpi":200})
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey="row",layout="constrained")
    for col,k in enumerate((2,6,12)):
        for row,metric in enumerate(("MC","MC_loss")):
            ax=axes[row,col]
            for family in FAMILIES:
                for strategy in ("largest","random","smallest"):
                    h=summary[(summary.protocol=="equal_fraction")&(summary.family==family)&
                              (summary.strategy==strategy)&(summary.requested_dimension==k)&(summary.metric==metric)]
                    b=summary[(summary.protocol=="baseline")&(summary.family==family)&(summary.metric==metric)].iloc[0].copy()
                    b["dose"]=0.
                    h=pd.concat([pd.DataFrame([b]),h]).sort_values("dose")
                    band(ax,h.dose.to_numpy()*100,h,COLORS[family],f"{LABELS[family]} · {strategy}",STYLES[strategy])
            if row==1:
                ax.axhline(0,color=".6",lw=.6)
            else:
                mc_summary=summary[summary.metric=="MC"]
                ax.set_ylim(mc_summary.ci_low.min()-.2,mc_summary.ci_high.max()+.2)
            ax.set_ylabel("Memory capacity" if row==0 else "MC loss from own baseline")
            if row==0:
                actual=d[(d.protocol=="equal_fraction")&(d.requested_dimension==k)].actual_dimension
                dimension=f"{actual.min()}" if actual.min()==actual.max() else f"{actual.min()}–{actual.max()}"
                ax.set_title(f"Target ≈{k} secondary modes ({dimension} dimensions)")
            else:
                ax.set_xlabel("Eigenvalue attenuation (%)")
            ax.set_xticks([0,25,50,100])
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=9)
    fig.suptitle("20% inhibitory edges · leading eigenvalue retained · readouts refitted\nBands: paired-realization bootstrap 95% intervals",fontsize=14)
    fig.savefig(out/"01_mode_suppression.png")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True,layout="constrained")
    for ax,family in zip(axes,FAMILIES):
        b=curves.merge(d[["condition_id","repeat_id","family","protocol","strategy","requested_dimension","dose"]],
                       on=["condition_id","repeat_id","family"],validate="many_to_one")
        selections=[("Baseline",b[(b.family==family)&(b.protocol=="baseline")],"#222222")]
        for k,color in zip((2,6,12),("#009988","#ee7733","#cc3311")):
            selections.append((f"Largest ≈{k} modes → 0",b[(b.family==family)&(b.protocol=="equal_fraction")&
                               (b.strategy=="largest")&(b.requested_dimension==k)&(b.dose==1.)],color))
        for label,h,color in selections:
            panel=h.pivot(index="repeat_id",columns="delay",values="capacity")
            mean,draws=bootstrap_panel(panel.to_numpy(),n_boot=5000,seed=20261118)
            lo,hi=np.quantile(draws,[.025,.975],axis=0)
            ax.plot(panel.columns,mean,label=label,color=color)
            ax.fill_between(panel.columns,lo,hi,color=color,alpha=.12)
        ax.set(title=LABELS[family],xlabel="Input delay (steps)",ylabel="Squared-correlation memory score",ylim=(0,1.03))
        ax.legend(fontsize=9)
    fig.suptitle("Which delays lose memory after secondary-mode suppression?",fontsize=14)
    fig.savefig(out/"02_memory_by_delay.png")
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout="constrained")
    axes[1].sharey(axes[0])
    for ax,family in zip(axes[:2],FAMILIES):
        for strategy in ("largest","random"):
            h=summary[(summary.protocol=="equal_frobenius")&(summary.family==family)&
                      (summary.strategy==strategy)&(summary.metric=="MC_loss")].sort_values("dose")
            band(ax,h.dose.to_numpy()*100,h,COLORS[family],strategy,STYLES[strategy])
        ax.axhline(0,color=".5",lw=.7)
        ax.set(title=LABELS[family],xlabel="Fraction of shared feasible perturbation budget (%)",ylabel="MC loss")
        ax.legend()
    ax=axes[2]
    for family in FAMILIES:
        h=contrasts[(contrasts.protocol=="equal_frobenius")&(contrasts.family==family)&
                    (contrasts.contrast=="target loss minus control loss")].sort_values("dose")
        band(ax,h.dose.to_numpy()*100,h,COLORS[family],LABELS[family])
    ax.axhline(0,color=".5",lw=.7)
    ax.set(title="Selective cost of targeting large modes",xlabel="Shared budget (%)",ylabel="Target MC loss − random-mode MC loss")
    ax.legend(fontsize=9)
    fig.suptitle("≈6 secondary modes · target and random control have equal ||ΔW||F within each pair\nRandom controls averaged within realization; bands resample realizations",fontsize=12)
    fig.savefig(out/"03_size_matched_controls.png")
    plt.close(fig)

    cache=np.load(out/"cache/repeat_000_networks.npz")
    fig,axes=plt.subplots(1,2,figsize=(10,5.2),sharex=True,sharey=True,layout="constrained")
    for family,ax in zip(FAMILIES,axes):
        a=cache[f"{family}_weights"].T.astype(float)
        t,z,blocks=real_schur_blocks(a)
        chosen=secondary_blocks(blocks,6)
        e=np.linalg.eigvals(a)
        changed=np.linalg.eigvals(attenuate_blocks(a,t,z,blocks,chosen,0))
        ax.scatter(e.real,e.imag,s=16,facecolors="none",edgecolors=COLORS[family],alpha=.7,label="Baseline")
        ax.scatter(changed.real,changed.imag,s=9,color="#cc3311",alpha=.5,label="Largest ≈6 secondary modes → 0")
        circle=plt.Circle((0,0),.95,fill=False,color=".7",ls="--")
        ax.add_patch(circle)
        ax.axhline(0,color=".8",lw=.5); ax.axvline(0,color=".8",lw=.5)
        ax.set(xlim=(-1,1),ylim=(-1,1),xlabel="Real eigenvalue component",ylabel="Imaginary component",title=LABELS[family],aspect="equal")
        ax.legend(fontsize=8,loc="lower center")
    fig.suptitle("Illustrative realization 0 · largest eigenvalue remains at radius 0.95",fontsize=12)
    fig.savefig(out/"04_example_spectra.png")
    plt.close(fig)


def write_report(out,d,summary,contrasts,checks):
    def entry(family,metric,protocol="baseline",strategy="baseline",k=0,dose=0):
        return summary[(summary.family==family)&(summary.metric==metric)&(summary.protocol==protocol)&
                       (summary.strategy==strategy)&(summary.requested_dimension==k)&(summary.dose==dose)].iloc[0]
    def interval(row):
        return f"{row['mean']:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
    def contrast(protocol,kind,family="paired",strategy="largest",k=6,dose=1):
        return contrasts[(contrasts.protocol==protocol)&(contrasts.contrast==kind)&(contrasts.family==family)&
                         (contrasts.strategy==strategy)&(contrasts.requested_dimension==k)&(contrasts.dose==dose)].iloc[0]
    baseline=contrast("baseline","CE minus ER MC",strategy="baseline",k=0,dose=0)
    baseline_statement=("The paired baseline interval includes zero, so this 20-realization pilot does not establish a positive CE advantage by itself. "
                        "It cannot demonstrate that an independently established advantage disappears.") if baseline.ci_low<=0<=baseline.ci_high else (
                        "The baseline difference is conditional on this input, leak, radius and sign fraction; it does not establish a universal architecture advantage.")
    lines=["# Local spectral-mode intervention at 20% inhibitory edges", "",
           f"Completed {checks['n_realizations']} paired CPU realizations, {checks['n_evaluations']:,} MC evaluations, and {checks['n_delay_scores']:,} delay scores. "
           "All readouts were refitted after each intervention. Existing results were preserved.", "",
           "## Main results", "",
           f"Original mean MC: CE **{entry('cel_matched','MC')['mean']:.3f}**, ER **{entry('matched_er','MC')['mean']:.3f}**. "
           f"Paired CE−ER difference: **{interval(baseline)}** (pointwise bootstrap 95% interval). {baseline_statement}", "",
           "The table below sets the selected secondary eigenvalues to zero while leaving the leading block unchanged. "
           "This shortens their persistence; it does not delete every transient pathway through those coordinates.", "",
           "| Requested secondary dimensions | Model | MC after suppression | MC loss [95% interval] | Random-mode MC loss | Smaller-mode MC loss |",
           "|---:|---|---:|---:|---:|---:|"]
    for k in (2,6,12):
        for family in FAMILIES:
            target=entry(family,"MC_loss","equal_fraction","largest",k,1)
            lines.append(f"| {k} | {LABELS[family]} | {entry(family,'MC','equal_fraction','largest',k,1)['mean']:.3f} | {interval(target)} | "
                         f"{entry(family,'MC_loss','equal_fraction','random',k,1)['mean']:.3f} | "
                         f"{entry(family,'MC_loss','equal_fraction','smallest',k,1)['mean']:.3f} |")
    differential=contrast("equal_fraction","CE loss minus ER loss")
    after=contrast("equal_fraction","CE minus ER MC")
    lines.extend(["",f"For ≈6 secondary dimensions, CE loses **{interval(differential)}** more MC than ER under complete attenuation. "
                  f"The resulting CE−ER MC difference is **{interval(after)}**. These equal-fraction interventions have different weight-change sizes between architectures.","",
                  "![MC and loss under gradual suppression](01_mode_suppression.png)","",
                  "## Does the effect survive matching the size of the weight change?", "",
                  "For the ≈6-mode comparison, each target/random pair has the same actual number of dimensions, the same counts of real and conjugate-pair blocks, "
                  "and the same Frobenius weight-change norm. Two random controls are averaged within each realization. The full feasible budget is the smaller "
                  "of the two selected-block norms; both half and full budgets are tested. The attenuation multipliers consequently differ between target and control.","",
                  "| Model | Target MC loss | Random MC loss | Target loss − random loss [95% interval] | Mean shared weight-change norm |",
                  "|---|---:|---:|---:|---:|"])
    for family in FAMILIES:
        selective=contrast("equal_frobenius","target loss minus control loss",family,"random")
        lines.append(f"| {LABELS[family]} | {entry(family,'MC_loss','equal_frobenius','largest',6,1)['mean']:.3f} | "
                     f"{entry(family,'MC_loss','equal_frobenius','random',6,1)['mean']:.3f} | {interval(selective)} | "
                     f"{entry(family,'delta_frobenius','equal_frobenius','largest',6,1)['mean']:.3f} |")
    lines.extend(["", "Budgets are matched **within** each architecture/target/control pair. They are not identical across CE and ER, "
                  "so their raw loss difference here is not a comparison at a common cross-architecture perturbation budget.","",
                  "![Size-matched comparisons](03_size_matched_controls.png)","",
                  "## Interpretation", ""])
    for family in FAMILIES:
        selective=contrast("equal_frobenius","target loss minus control loss",family,"random")
        if selective.ci_low>0:
            lines.append(f"- In {LABELS[family]}, targeting large secondary blocks costs more memory than random blocks even after matching weight-change size: "
                         f"{interval(selective)}. This supports a contribution of the targeted persistence structure in this model intervention.")
        else:
            lines.append(f"- In {LABELS[family]}, the size-matched selectivity interval includes or lies below zero ({interval(selective)}); "
                         "a selective cost of targeting large blocks is not established.")
    retention={family:entry(family,"MC","equal_fraction","largest",6,1)["mean"]/entry(family,"MC")["mean"] for family in FAMILIES}
    delay_summary=pd.read_csv(out/"delay_band_losses.csv")
    ce_delays=delay_summary[(delay_summary.family=="cel_matched")&(delay_summary.requested_dimension==6)]
    mid_delay=ce_delays[ce_delays.delay_low==11].iloc[0]
    early_delay=ce_delays[ce_delays.delay_low==1].iloc[0]
    lines.extend([f"- After ≈6-mode complete attenuation, mean MC retains {retention['cel_matched']:.1%} of its baseline in CE and {retention['matched_er']:.1%} in ER. "
                  "The intervention does not collapse memory; it does not support saying that a handful of modes do all the computational work.",
                  f"- CE's loss is concentrated at later delays: delays 11–20 account for {interval(mid_delay)} MC loss, "
                  f"while delays 1–5 change by {interval(early_delay)}. This is consistent with shortening temporal persistence.",
                  "- This is evidence about the consequences of a specified weight-matrix intervention. It does not uniquely isolate eigenvalue magnitudes "
                  "from changes to eigenvectors, transient interactions or nonlinear operating regime, and does not explain the entire E/I crossover.","",
                  "![Memory scores by input delay](02_memory_by_delay.png)","",
                  "## Methods and limits", "",
                  "- Pre-intervention sign fraction 20%; own-radius target 0.95; leak 1; input scale 0.1; added bias 0. CE restores unknown edges using the existing "
                  "matched-sign construction. ER is the existing **binary-weight** sign-sweep control, not Gaussian ER. Both have 297 nodes and 3,604 original edges.",
                  "- The actual repository dynamics use `W.T @ state`. SciPy real Schur decomposition is applied to `A = W.T`; selected 1×1/2×2 diagonal blocks "
                  "are multiplied by 0.75, 0.5 or 0. Off-block Schur entries are retained. No global renormalization follows. The leading block is excluded from all selections.",
                  "- Requested dimensions 2, 6 and 12 are reached or slightly exceeded to preserve conjugate pairs. Actual counts are in `interventions.csv`; "
                  "all controls match the target's actual singleton/pair counts within each realization. Random selections may overlap target selections.",
                  "- MC reuses `network_stats.run_one.run_reservoir_states` and `network_stats.stats.compute_MC`: float32 dynamics, float64 readout fits, "
                  "500 washout, 1,500 training, 500 test steps, delays 1–30, ridge 1e-4 plus existing 1e-6 epsilon. Same input stream and input weights "
                  "are used for every intervention within a realization. There is no selection on test scores and no new benchmark implementation.",
                  "- Base seed 20261117 + repeat_id×100000; the existing network, unknown-sign and MC-input offsets are retained. All baselines are rerun on CPU. "
                  "CPU and CUDA input-weight draws can differ at the same seed, so these scores are not exact reproductions of the earlier GPU observations.",
                  "- Bootstrap: 5,000 paired-realization resamples, percentile 95% intervals, seed 20261118. Random-control draws are averaged within realization; "
                  "configurations and perturbations are not treated as independent networks. Intervals are exploratory and pointwise, without multiplicity adjustment.",
                  "- Reconstruction generally creates dense connections and changes edge signs. **20% refers to the original networks**; the modified matrices "
                  "do not preserve topology or inhibitory-edge fraction. This is a mechanistic model diagnostic, not a biological sign-preserving intervention.",
                  "- One E/I fraction, one hyperparameter combination, one input stream per realization, and 20 realizations limit generality. "
                  "The Schur intervention depends on its chosen basis and retains nonnormal off-block couplings; zero eigenvalues can still support transient memory.","",
                  "![Example spectra before and after attenuation](04_example_spectra.png)","",
                  "## Validation and reproducibility", "",
                  f"All {checks['n_evaluations']:,} scores are finite and within the MC bounds. Maximum leading-radius change: {checks['max_radius_change']:.2e}. "
                  f"Maximum target/control Frobenius mismatch: {checks['max_frobenius_pair_mismatch']:.2e}. "
                  "Six synthetic intervention tests and six existing bootstrap/analysis tests pass. Source matrices are hashed and checked for changes; "
                  "exact settings, package versions and code hashes are in `manifest.json`. `VALIDATION.json` records numerical checks. "
                  "Per-realization baseline matrices, input weights, inputs and Schur decompositions are cached for reconstruction.","",
                  "```bash", "python tools/run_spectral_memory_pilot.py --n-repeats 20 --workers 4",
                  "python tools/analyze_spectral_memory_pilot.py", "```", "",
                  "Outputs: `interventions.csv`, `memory_curves.csv`, `summary_metrics.csv`, `paired_contrasts.csv`, `delay_band_losses.csv`, "
                  "four PNG figures, provenance JSON and per-realization caches. Existing completed realization files are reused only when the run manifest matches. "
                  "No PDFs are generated.", ""])
    (out/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/spectral_memory_intervention_ei20_2026-11-18")
    args=parser.parse_args(); out=args.out_dir
    data=pd.read_csv(out/"interventions.csv")
    curves=pd.read_csv(out/"memory_curves.csv")
    manifest=json.loads((out/"manifest.json").read_text())
    expected=manifest["n_repeats"]*90
    if len(data)!=expected or data.duplicated(["repeat_id","condition_id"]).any():
        raise ValueError("Missing or duplicate intervention evaluations.")
    totals=curves.groupby(["repeat_id","condition_id"]).capacity.agg(["sum","count"])
    totals=totals.join(data.set_index(["repeat_id","condition_id"])["MC"],validate="one_to_one")
    if not (totals["count"]==30).all() or np.max(abs(totals["sum"]-totals.MC))>1e-5:
        raise ValueError("Memory curves do not reproduce total MC.")
    if not curves.capacity.between(0,1).all() or not data.MC.between(0,30).all():
        raise ValueError("Scores outside squared-correlation MC bounds.")
    d,summary=summarize(data)
    contrasts=contrast_tables(d)
    summary.to_csv(out/"summary_metrics.csv",index=False)
    contrasts.to_csv(out/"paired_contrasts.csv",index=False)
    delay_band_losses(d,curves).to_csv(out/"delay_band_losses.csv",index=False)
    plots(out,d,summary,curves,contrasts)
    # Numerical checks are separate from statistical interpretation.
    matched=d[d.protocol=="equal_frobenius"].pivot(index=["repeat_id","family","draw","budget_fraction"],columns="strategy",values="delta_frobenius")
    checks=dict(n_realizations=int(d.repeat_id.nunique()),n_evaluations=len(d),n_delay_scores=len(curves),
                complete_design=bool(len(data)==expected),
                max_curve_sum_error=float(np.max(abs(totals["sum"]-totals.MC))),
                finite_scores=bool(np.isfinite(d[["MC","MC_loss"]]).all().all()),
                max_radius_change=float(np.max(abs(d.post_rho-d.baseline_rho))),
                max_frobenius_pair_mismatch=float(np.max(abs(matched.largest-matched.random))),
                max_frobenius_prediction_error=float(np.max(abs(d.delta_frobenius-d.expected_delta_frobenius))),
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bootstrap=dict(n_resamples=5000,method="percentile",unit="paired realization",seed=20261118),
                intervals="Pointwise exploratory 95% intervals; no multiplicity correction.")
    if checks["max_radius_change"]>2e-5 or checks["max_frobenius_pair_mismatch"]>2e-5:
        raise ValueError("Intervention audit failed.")
    (out/"VALIDATION.json").write_text(json.dumps(checks,indent=2)+"\n")
    write_report(out,d,summary,contrasts,checks)
    print(json.dumps(checks,indent=2))
    print(contrasts[(contrasts.protocol=="baseline")|((contrasts.protocol=="equal_fraction")&(contrasts.requested_dimension==6)&(contrasts.dose==1))|(contrasts.protocol=="equal_frobenius")].to_string(index=False))


if __name__=="__main__":
    main()
