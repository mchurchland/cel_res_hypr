#!/usr/bin/env python3
"""Compare mixed STP with the three saved control conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpl-mixed-stp-pilot")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/xdg-mixed-stp-pilot")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import analyze_stp_mc_narma_pilot as plots

# Configure the unchanged shared plotting/summary utilities for a fourth mode.
plots.MODES=("static","facilitating","depressing","mixed")
plots.STYLES={**plots.STYLES,"mixed":"-."}
plots.MODE_COLORS={**plots.MODE_COLORS,"mixed":"#aa3377"}


def report(out,summary,effects,advantages,selected,manifest,checks):
    def cell(scheme,mode,family,metric,ei=20,scale=.1):
        return summary[(summary.input_scheme==scheme)&(summary.synapses==mode)&(summary.family==family)&
                       (summary.metric==metric)&(summary.ei_percent==ei)&(summary.input_scale==scale)].iloc[0]
    def interval(row):return f"{row['mean']:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
    lines=["# Mixed facilitating/depressing STP: local MC/NARMA10 follow-up","",
           "Added **148 facilitating and 149 depressing presynaptic neurons** to each reservoir. "
           "The assignment is randomized per realization, shared by CE and ER at the same node indices, independent of edge signs, "
           "and held fixed across EI fractions, input strengths, input arrangements and tasks. "
           "Each neuron's outgoing connections use its assigned preset.","",
           f"**{checks['new_condition_pairs']} new MC/NARMA condition pairs** were run across eight paired realizations. "
           f"All {checks['reused_condition_pairs']} preceding static/facilitating/depressing condition pairs were reused exactly. "
           "No spectral-radius grid was added: the same initial radius 0.95 is retained, and there is no dynamic renormalization.","",
           "The two earlier presets already contain both utilization and resource-depletion dynamics. "
           "The new condition mixes the two preset types within the same network rather than adding a second process to an otherwise one-process synapse.","",
           "## Main preliminary result", "",
           "At 20% inhibitory edges and input scale 0.1, the mixed condition favors ER on both MC and NARMA10, "
           "with full input and with sensory/random input. Compared with the paired static controls, ER improves on both tasks while CE deteriorates. "
           "Thus the CE advantage under the homogeneous facilitating preset does not persist under this mixed assignment.", "",
           "Around 40–60%, CE remains ahead of mixed ER under weak drive, but ER also deteriorates sharply relative to its static control there. "
           "The full sweep and paired changes retain these counterexamples. No general CE or ER superiority follows from this eight-realization, "
           "one-mixture-fraction pilot.", "",
           "## At 20% inhibitory edges and weak input (0.1)","",
           "| Input arrangement | Synapses | CE MC | ER MC | CE NARMA10 NMSE | ER NARMA10 NMSE |",
           "|---|---|---:|---:|---:|---:|"]
    for scheme in plots.SCHEMES:
        for mode in plots.MODES:
            lines.append(f"| {'Full' if scheme=='full' else 'Sensory/random'} | {mode} | "
                         f"{cell(scheme,mode,'cel_matched','MC')['mean']:.3f} | {cell(scheme,mode,'matched_er','MC')['mean']:.3f} | "
                         f"{cell(scheme,mode,'cel_matched','NARMA10_test_nmse')['mean']:.4f} | {cell(scheme,mode,'matched_er','NARMA10_test_nmse')['mean']:.4f} |")
    lines.extend(["","## Does the mixed condition improve each architecture?","",
                  "At 20% and input scale 0.1, positive values indicate improvement relative to each realization's static control. "
                  "Intervals are paired-realization bootstrap 95% intervals, exploratory and uncorrected for multiplicity.","",
                  "| Input | Model | MC improvement [95% interval] | NARMA10 error reduction [95% interval] |",
                  "|---|---|---:|---:|"])
    for scheme in plots.SCHEMES:
        for family in plots.FAMILIES:
            h=effects[(effects.input_scheme==scheme)&(effects.family==family)&(effects.synapses=="mixed")&
                      (effects.ei_percent==20)&(effects.input_scale==.1)]
            lines.append(f"| {'Full' if scheme=='full' else 'Sensory/random'} | {plots.LABELS[family]} | "
                         f"{interval(h[h.metric=='MC'].iloc[0])} | {interval(h[h.metric=='NARMA10_test_nmse'].iloc[0])} |")
    lines.extend(["","## Full sweep","",
                  "![Full input](performance_full.png)","",
                  "![Sensory/random input](performance_restricted_matched_norm.png)","",
                  "![Changes from static synapses](mixed_changes_from_static.png)","",
                  "## Validation-selected NARMA10","",
                  "Input scale is selected from 0.1 and 0.5 using validation, separately for each synapse condition. "
                  "The second selection analysis allows each architecture to choose among all eight input-scale/synapse combinations "
                  "(static plus three STP conditions), again using validation only. Test outcomes do not choose models.","",
                  "| EI | Input | CE test NMSE | ER test NMSE | CE advantage [95% interval] |",
                  "|---:|---|---:|---:|---:|"])
    for row in selected.itertuples():
        lines.append(f"| {row.ei_percent:g}% | {'Full' if row.input_scheme=='full' else 'Sensory/random'} | "
                     f"{row.CE_test_NMSE:.4f} | {row.ER_test_NMSE:.4f} | {row.mean:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |")
    lines.extend(["","![Validation-selected performance](validation_selected_narma.png)","",
                  "## Model and interpretation","",
                  "- Identical rate-based STP equations, four RK4 substeps, absolute tanh-activity rate proxy, recovered-rest initialization "
                  "and gain normalization u*x/U as the previous pilot. Facilitation preset: U=0.15, tau_f=75, tau_d=5; depression preset: U=0.45, tau_f=5, tau_d=75. "
                  "Time constants are in model steps and are not fitted worm physiology.",
                  "- A neuron receives one preset; it retains both resource and utilization dynamics. Both excitatory and inhibitory edges "
                  "can be facilitating or depressing. Assignment is not linked to neuron function or inhibitory-edge percentage. "
                  "Approximately half the neurons are assigned each type; fractions of affected edges and absolute recurrent weight need not be exactly half, "
                  "especially for heterogeneous CE. Those fractions are saved per condition.",
                  "- Reuses the original cached recurrent matrices, exact input coefficients, stimuli and ReservoirPy 0.4.2 NARMA10 targets. "
                  "The readout pipeline is unchanged: MC delays 1–30, float64 ridge fitting, and NARMA ridge chosen using validation with an untouched test segment. "
                  "No benchmark generator was rewritten and no original conditions were rerun.",
                  "- Both full input and 78-neuron sensory CE/random ER input are retained. Restricted input matches the dense vector's total L2 norm. "
                  "Input scales 0.1 and 0.5, six fractions 0–100%, unit leak and zero added bias remain fixed as in the preceding pilot.",
                  "- The mixed assignment is one random draw per computational realization. Eight seed clusters jointly vary network/sign realization "
                  "and mixed assignment; this design does not estimate assignment variability separately. These are not eight independent animals.",
                  "- STP changes mean gain as well as adding history dependence. A larger CE–ER gap need not mean that CE itself improved; "
                  "use the paired effects table and static controls. No matched static-gain control or biological calibration is included.","",
                  "## Validation and provenance","",
                  "Four mixed-backend tests pass, including **bitwise equality** with the original simulation when all neurons use one preset. "
                  "Silence, resets, balanced reproducible assignment and bounded resource variables are checked. "
                  "All original condition values are retained exactly, all mixed task scores are finite, all 30 MC delays are present, "
                  "and original source/result hashes are unchanged. Old outputs remain in their original directory.","",
                  "All estimates and intervals are in `summary_metrics.csv`, `stp_effects.csv`, `ce_er_advantages.csv` and the validation-selection tables. "
                  "`mixed_assignment.csv` records each neuron's type in every realization; `mixed_assignment_by_ei_sign.csv` audits independence from edge signs. "
                  "`manifest.json` hashes the reused datasets and simulation code. New caches store exact assignments and terminal STP states. "
                  "The original pilot report supplies the full inherited protocol. PNG only; no PDFs.","",
                  "```bash","python tools/run_mixed_stp_mc_narma_pilot.py --workers 4",
                  "python tools/analyze_mixed_stp_mc_narma_pilot.py","```",""])
    (out/"REPORT.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/stp_mc_narma_mixed_sweep_2026-11-18")
    args=parser.parse_args();out=args.out_dir
    manifest=json.loads((out/"manifest.json").read_text());parent=ROOT/manifest["source_dir"]
    data=pd.read_csv(out/"performance.csv",float_precision="round_trip")
    curves=pd.read_csv(out/"memory_curves.csv",float_precision="round_trip")
    original=pd.read_csv(parent/"performance.csv",float_precision="round_trip")
    reused=data[data.synapses!="mixed"][original.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(reused,original,check_exact=True,check_dtype=False)
    expected=manifest["n_repeats"]*192
    keys=["repeat_id","family","ei_percent","input_scale","input_scheme","synapses"]
    if len(data)!=expected or data.duplicated(keys).any():raise ValueError("Incomplete or duplicate mixed design.")
    mixed=data[data.synapses=="mixed"]
    if len(mixed)!=manifest["new_conditions"]:raise ValueError("Wrong mixed count.")
    if not np.isfinite(data[["MC","NARMA10_test_nmse","NARMA10_val_nmse"]]).all().all():raise ValueError("Nonfinite scores.")
    if not data.MC.between(0,30).all() or not curves.capacity.between(0,1).all():raise ValueError("MC bounds violated.")
    if not np.allclose(data.NARMA10_test_nmse,data.NARMA10_test_nrmse**2):raise ValueError("Inconsistent NARMA metrics.")
    totals=curves.groupby(["repeat_id","condition_id"]).capacity.agg(["sum","count"])
    totals=totals.join(data.set_index(["repeat_id","condition_id"])["MC"],validate="one_to_one")
    if not (totals["count"]==30).all() or np.max(abs(totals["sum"]-totals.MC))>1e-5:raise ValueError("Invalid memory curves.")
    if data.groupby(["repeat_id","family","ei_percent"]).network_sha256.nunique().max()!=1:raise ValueError("Different base matrices.")
    for task in ("MC","NARMA10"):
        if mixed[task+"_stp_state_min"].min()<-1e-8 or mixed[task+"_stp_state_max"].max()>1+1e-8:raise ValueError("STP states out of bounds.")
    node_names=(ROOT/"Connectome/ce_nodes.txt").read_text().splitlines();assignments=[];sign_audit=[]
    for repeat in range(manifest["n_repeats"]):
        with np.load(out/"cache"/f"mixed_repeat_{repeat:03d}_states.npz") as z:mask=z["facilitating_neurons"]
        if len(mask)!=297 or mask.sum()!=148:raise ValueError("Wrong assignment counts.")
        assignments.extend(dict(repeat_id=repeat,node_index=i,neuron=name,synapse_type="facilitating" if mask[i] else "depressing") for i,name in enumerate(node_names))
        with np.load(parent/"cache"/f"repeat_{repeat:03d}_data_and_networks.npz") as z:
            for family in plots.FAMILIES:
                for ei in (0,20,40,60,80,100):
                    w=z[f"{family}_ei{ei:03d}_weights"]
                    for flag,label in ((True,"facilitating"),(False,"depressing")):
                        entries=w[mask==flag];nz=entries[entries!=0]
                        sign_audit.append(dict(repeat_id=repeat,family=family,ei_percent=ei,synapse_type=label,
                                               n_presynaptic=int(np.sum(mask==flag)),n_edges=len(nz),n_negative=int(np.sum(nz<0)),n_positive=int(np.sum(nz>0))))
    pd.DataFrame(assignments).to_csv(out/"mixed_assignment.csv",index=False)
    pd.DataFrame(sign_audit).to_csv(out/"mixed_assignment_by_ei_sign.csv",index=False)
    summary=plots.summarize(data);effects,advantages=plots.comparison_tables(data)
    chosen,chosen_summary=plots.selected_narma(data);all_chosen,all_summary=plots.selected_across_presets(data)
    for name,frame in (("summary_metrics",summary),("stp_effects",effects),("ce_er_advantages",advantages),
                       ("validation_selected_configurations",chosen),("validation_selected_narma",chosen_summary),
                       ("validation_selected_across_presets_configurations",all_chosen),("validation_selected_across_presets",all_summary)):
        frame.to_csv(out/(name+".csv"),index=False)
    plots.make_plots(out,summary,effects,advantages,chosen_summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey="row",layout="constrained")
    for col,scheme in enumerate(plots.SCHEMES):
        for row,metric in enumerate(plots.METRICS):
            ax=axes[row,col]
            for family in plots.FAMILIES:
                h=effects[(effects.input_scheme==scheme)&(effects.input_scale==.1)&(effects.family==family)&(effects.synapses=="mixed")&(effects.metric==metric)]
                plots.draw_band(ax,h,plots.COLORS[family],plots.LABELS[family])
            ax.axhline(0,color=".5",lw=.8)
            ax.set_title("Full input" if scheme=="full" else "Sensory CE / random ER")
            ax.set_ylabel("MC change from static" if metric=="MC" else "NARMA10 error reduction from static")
            if row==1:ax.set_xlabel("Inhibitory edges (%)")
            ax.set_xticks([0,20,40,60,80,100])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Mixed STP versus static synapses · input scale 0.1\n148 facilitating / 149 depressing neurons; positive means improvement",fontsize=12)
    fig.savefig(out/"mixed_changes_from_static.png");plt.close(fig)
    checks=dict(n_realizations=manifest["n_repeats"],total_condition_pairs=len(data),new_condition_pairs=len(mixed),
                reused_condition_pairs=len(reused),original_scores_reused_exactly=True,all_scores_finite=True,
                complete_design=True,max_memory_curve_error=float(np.max(abs(totals["sum"]-totals.MC))),
                stp_states_bounded=True,n_facilitating=148,n_depressing=149,initial_radius_grid=False,dynamic_renormalization=False,
                mixed_tests_passed=4,shared_analysis_code_sha256=hashlib.sha256(Path(plots.__file__).read_bytes()).hexdigest(),
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bootstrap=dict(n_resamples=5000,method="percentile",unit="paired realization",seed=20261118))
    for filename,expected_hash in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=expected_hash:raise ValueError("Original source changed.")
    (out/"VALIDATION.json").write_text(json.dumps(checks,indent=2)+"\n")
    report(out,summary,effects,advantages,all_summary,manifest,checks)
    print(json.dumps(checks,indent=2))
    print(effects[(effects.synapses=="mixed")&(effects.ei_percent==20)&(effects.input_scale==.1)].round(5).to_string(index=False))


if __name__=="__main__":main()
