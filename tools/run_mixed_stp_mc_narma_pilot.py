#!/usr/bin/env python3
"""Add only a mixed STP condition to the completed local MC/NARMA pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[variable]="1"
import numpy as np
import pandas as pd
import torch

from network_stats.mixed_short_term_plasticity import mixed_assignment,run_mixed_stp_states
from network_stats.temporal_tasks import TemporalTaskSettings,fit_narma_readout
from network_stats.stats import compute_MC

PARENT=ROOT/"analysis/stp_mc_narma_full_sweep_2026-11-18"
ASSIGNMENT_SEED=20261119


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_mixed(weights,input_weights,data,task,facilitating):
    metrics={};diagnostics={};last={};curve=None
    for task_name,key in (("MC","mc_input"),("NARMA10","narma_input")):
        states,details,u,x,U=run_mixed_stp_states(weights,input_weights,data[key].numpy(),facilitating,task.washout)
        h=torch.from_numpy(states);start=task.washout;end=start+task.train
        if task_name=="MC":
            metrics["MC"],curve=compute_MC(h[start:end].double(),h[end:].double(),data[key][start:end].double(),
                                           data[key][end:].double(),task.mc_max_delay,task.mc_ridge_alpha,torch.device("cpu"))
        else:
            val_end=end+task.validation;y=data["narma_target"]
            metrics.update(fit_narma_readout(h[start:end],y[start:end],h[end:val_end],y[end:val_end],
                                            h[val_end:],y[val_end:],task.narma_ridge_alphas))
        gain=u*x/U
        diagnostics.update({task_name+"_"+key:value for key,value in details.items()})
        diagnostics[task_name+"_saturation_fraction"]=float(np.mean(abs(states[start:])>.9))
        diagnostics[task_name+"_mean_state_variance"]=float(np.mean(np.var(states[start:].astype(float),axis=0)))
        diagnostics[task_name+"_last_effective_radius"]=float(np.max(abs(np.linalg.eigvals(weights.astype(float)*gain[:,None]))))
        last[task_name+"_last_gain"]=gain;last[task_name+"_last_utilization"]=u;last[task_name+"_last_resources"]=x
    if not np.isfinite(list(metrics.values())).all():raise ValueError("Nonfinite mixed STP performance.")
    return metrics,curve,diagnostics,last


def run_repeat(job):
    repeat,out,parent,base_settings=job
    out=Path(out);parent=Path(parent);complete=out/"cache"/f"mixed_repeat_{repeat:03d}.csv"
    if complete.exists():return repeat,len(pd.read_csv(complete)),"cached"
    torch.set_num_threads(1);start=time.monotonic()
    source=pd.read_csv(parent/"performance.csv",float_precision="round_trip")
    source=source[source.repeat_id==repeat]
    static=source[source.synapses=="static"].sort_values("condition_id")
    if len(static)!=48:raise ValueError("Missing static source conditions.")
    with np.load(parent/"cache"/f"repeat_{repeat:03d}_data_and_networks.npz") as z:
        original={key:z[key].copy() for key in z.files}
    n=len(original["cel_matched_ei000_weights"])
    assignment=mixed_assignment(n,ASSIGNMENT_SEED+repeat*100000)
    data={key:torch.from_numpy(original[key]) for key in ("mc_input","narma_input","narma_target")}
    task=TemporalTaskSettings(**base_settings)
    rows=[];curves=[];cache={"facilitating_neurons":assignment}
    offset=int(source.condition_id.max())+1
    for i,(_,row) in enumerate(static.iterrows()):
        weights=original[f"{row.family}_ei{round(row.ei_percent):03d}_weights"]
        input_weights=original[f"condition_{int(row.condition_id):03d}_input_weights"]
        if hashlib.sha256(weights.tobytes()).hexdigest()!=row.network_sha256:raise ValueError("Source matrix hash mismatch.")
        metrics,curve,details,last=evaluate_mixed(weights,input_weights,data,task,assignment)
        condition_id=offset+i
        record=row.to_dict()
        record.update(condition_id=condition_id,synapses="mixed",**metrics,**details,
                      n_facilitating=int(assignment.sum()),n_depressing=int((~assignment).sum()),
                      facilitating_edge_fraction=float(np.count_nonzero(weights[assignment])/np.count_nonzero(weights)),
                      facilitating_abs_weight_fraction=float(np.sum(abs(weights[assignment]))/np.sum(abs(weights))))
        rows.append(record)
        curves.extend(dict(condition_id=condition_id,repeat_id=repeat,delay=d+1,capacity=float(score)) for d,score in enumerate(curve))
        for key,value in last.items():cache[f"condition_{condition_id:03d}_{key}"]=value
    pd.DataFrame(curves).to_csv(out/"cache"/f"mixed_repeat_{repeat:03d}_curves.csv",index=False,float_format="%.17g")
    np.savez_compressed(out/"cache"/f"mixed_repeat_{repeat:03d}_states.npz",**cache)
    pd.DataFrame(rows).to_csv(complete,index=False,float_format="%.17g")
    return repeat,len(rows),f"{time.monotonic()-start:.1f}s"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=PARENT)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/stp_mc_narma_mixed_sweep_2026-11-18")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args();parent=args.source_dir.resolve();out=args.out_dir.resolve()
    if args.workers<1:parser.error("Workers must be positive.")
    (out/"cache").mkdir(parents=True,exist_ok=True)
    original_manifest=json.loads((parent/"manifest.json").read_text());n_repeats=original_manifest["n_repeats"]
    for category in ("code_hashes","source_hashes"):
        for name,expected in original_manifest[category].items():
            if sha(ROOT/name)!=expected:raise ValueError(f"Original source/code changed: {name}")
    source_files=([parent/"performance.csv",parent/"memory_curves.csv",parent/"manifest.json"]+
                  [parent/"cache"/f"repeat_{repeat:03d}_data_and_networks.npz" for repeat in range(n_repeats)])
    code=["tools/run_mixed_stp_mc_narma_pilot.py","network_stats/mixed_short_term_plasticity.py",
          "network_stats/short_term_plasticity.py","network_stats/temporal_tasks.py","network_stats/stats.py","network_stats/stats_util.py"]
    manifest=dict(settings=original_manifest["settings"],n_repeats=n_repeats,task=original_manifest["task"],
                  stp_presets=original_manifest["stp_presets"],packages=original_manifest["packages"],
                  source_dir=str(parent.relative_to(ROOT)),assignment_seed=ASSIGNMENT_SEED,
                  assignment="148 facilitating / 149 depressing presynaptic neurons; same randomized index assignment in CE and ER; independent of E/I signs; fixed across all conditions within realization.",
                  source_hashes={str(path.relative_to(ROOT)):sha(path) for path in source_files},
                  code_hashes={name:sha(ROOT/name) for name in code},
                  inherited_radius_grid=False,dynamic_renormalization=False,new_conditions_per_realization=48,
                  reused_conditions=n_repeats*144,new_conditions=n_repeats*48)
    manifest=json.loads(json.dumps(manifest));path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError("Incompatible mixed run; choose a new directory.")
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Reusing {n_repeats*144} original condition pairs; running {n_repeats*48} mixed pairs. No spectral-radius grid.",flush=True)
    run_mixed_stp_states(np.zeros((4,4),np.float32),np.ones((4,1),np.float32),np.zeros((5,1),np.float32),mixed_assignment(4,0),1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(repeat,str(out),str(parent),manifest["task"]) for repeat in range(n_repeats)]
        for future in as_completed([pool.submit(run_repeat,job) for job in jobs]):
            repeat,count,duration=future.result();print(f"[repeat {repeat:02d}] {count} new MC/NARMA pairs; {duration}",flush=True)
    original=pd.read_csv(parent/"performance.csv",float_precision="round_trip")
    additions=pd.concat([pd.read_csv(out/"cache"/f"mixed_repeat_{repeat:03d}.csv",float_precision="round_trip") for repeat in range(n_repeats)],ignore_index=True)
    data=pd.concat([original,additions],ignore_index=True)
    pd.testing.assert_frame_equal(data.iloc[:len(original)][original.columns].reset_index(drop=True),original,check_dtype=False,check_exact=True)
    old_curves=pd.read_csv(parent/"memory_curves.csv",float_precision="round_trip")
    new_curves=pd.concat([pd.read_csv(out/"cache"/f"mixed_repeat_{repeat:03d}_curves.csv",float_precision="round_trip") for repeat in range(n_repeats)],ignore_index=True)
    data.to_csv(out/"performance.csv",index=False,float_format="%.17g")
    pd.concat([old_curves,new_curves],ignore_index=True).to_csv(out/"memory_curves.csv",index=False,float_format="%.17g")
    if manifest["source_hashes"]!={str(path.relative_to(ROOT)):sha(path) for path in source_files}:raise ValueError("Original results changed during mixed run.")
    print(f"Saved {len(data)} total condition pairs; {len(additions)} newly simulated. {out}",flush=True)


if __name__=="__main__":main()
