#!/usr/bin/env python3
"""Limited local 0–100% EI pilot with dynamic synapses; no radius grid."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import pandas as pd
import torch

from mc_narma_crossover import build_condition,load_completed_connectome
from network_stats.temporal_tasks import TemporalTaskSettings,make_task_data,fit_narma_readout
from network_stats.run_one import run_reservoir_states
from network_stats.stats import compute_MC
from network_stats.short_term_plasticity import PRESETS,run_stp_states
from util.sensory_input import place_input_weights

FAMILIES=("cel_matched","matched_er")
MODES=("static","facilitating","depressing")
SETTINGS=dict(fractions=[0.,.2,.4,.6,.8,1.],input_scales=[.1,.5],
              input_schemes=["full","restricted_matched_norm"],
              initial_radius=.95,normalization="initial_own_radius_only",leak=1.,bias=0.,
              seed=20261117,repeat_stride=100000,mask_seed=20261118,
              stp_rate="abs(previous signed tanh activity)",stp_gain="utilization*resources/U",
              stp_time_units="input sample steps; not calibrated milliseconds")
SENSORY_SOURCE=ROOT/"analysis/sensory_input_pilot_2026-11-18/sensory_neurons.csv"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(weights,input_weights,data,task,mode):
    metrics={};curves=None;diagnostics={};states_to_cache={}
    for task_name,key in (("MC","mc_input"),("NARMA10","narma_input")):
        signal=data[key].numpy()
        if mode=="static":
            states=run_reservoir_states(torch.from_numpy(weights),torch.from_numpy(input_weights),data[key],1.).numpy()
            details=dict(gain_mean=1.,gain_sd=0.,gain_min=1.,gain_max=1.,utilization_mean=np.nan,
                         resources_mean=1.,stp_state_min=np.nan,stp_state_max=np.nan)
            gain_last=np.ones(len(weights))
        else:
            states,details,u_last,x_last=run_stp_states(weights,input_weights,signal,PRESETS[mode],task.washout)
            gain_last=u_last*x_last/PRESETS[mode].U
        x=torch.from_numpy(states)
        start,end=task.washout,task.washout+task.train
        if task_name=="MC":
            total,curves=compute_MC(x[start:end].double(),x[end:].double(),data[key][start:end].double(),
                                    data[key][end:].double(),task.mc_max_delay,task.mc_ridge_alpha,torch.device("cpu"))
            metrics["MC"]=total
        else:
            val_end=end+task.validation
            y=data["narma_target"]
            metrics.update(fit_narma_readout(x[start:end],y[start:end],x[end:val_end],y[end:val_end],
                                            x[val_end:],y[val_end:],task.narma_ridge_alphas))
        diagnostics.update({f"{task_name}_{name}":value for name,value in details.items()})
        diagnostics[f"{task_name}_saturation_fraction"]=float(np.mean(np.abs(states[start:])>.9))
        diagnostics[f"{task_name}_mean_state_variance"]=float(np.mean(np.var(states[start:].astype(float),axis=0)))
        # Diagnostic only: W_eff=diag(gain)*W, not the full STP-system Jacobian.
        effective=weights.astype(float)*gain_last[:,None]
        diagnostics[f"{task_name}_last_effective_radius"]=float(np.max(np.abs(np.linalg.eigvals(effective))))
        states_to_cache[f"{task_name}_last_gain"]=gain_last
    if not np.isfinite(list(metrics.values())).all():raise ValueError("Nonfinite task score.")
    return metrics,curves,diagnostics,states_to_cache


def run_repeat(job):
    repeat,out,sensory,settings=job
    out=Path(out);complete=out/"cache"/f"repeat_{repeat:03d}.csv"
    if complete.exists():return repeat,len(pd.read_csv(complete)),"cached"
    torch.set_num_threads(1);started=time.monotonic()
    task=TemporalTaskSettings()
    seed=settings["seed"]+repeat*settings["repeat_stride"]
    completed,_=load_completed_connectome(ROOT/"Connectome/ce_adj.npy",ROOT/"Connectome/ce_unknown_sign_weights.npy",seed)
    data=make_task_data(task,seed)
    sensory=np.asarray(sensory,dtype=int)
    # Same random ER mask as draw 0 of the preceding local sensory pilot.
    er_mask=np.random.default_rng(settings["mask_seed"]+repeat*10000+1000).choice(len(completed),len(sensory),replace=False)
    rows=[];delay_rows=[];cache={k:v.numpy() for k,v in data.items()}
    common_win=None
    for fraction in settings["fractions"]:
        for family in FAMILIES:
            base,win,info=build_condition(completed,family,fraction,"spectral_radius",seed,torch.device("cpu"))
            weights=(base*settings["initial_radius"]).numpy()
            original=weights.copy();network_hash=hashlib.sha256(weights.tobytes()).hexdigest()
            cache[f"{family}_ei{round(fraction*100):03d}_weights"]=weights
            if common_win is None:common_win=win.numpy().copy()
            np.testing.assert_array_equal(win.numpy(),common_win)
            for scale in settings["input_scales"]:
                dense=(win*scale).numpy()
                for scheme in settings["input_schemes"]:
                    indices=np.arange(len(weights)) if scheme=="full" else sensory if family=="cel_matched" else er_mask
                    input_weights=dense.copy() if scheme=="full" else place_input_weights(dense,sensory,indices,"matched_total_norm")
                    for mode in MODES:
                        metrics,curve,details,last=evaluate(weights,input_weights,data,task,mode)
                        condition_id=len(rows)
                        rows.append(dict(condition_id=condition_id,repeat_id=repeat,seed=seed,family=family,ei_percent=fraction*100,
                                         input_scale=scale,input_scheme=scheme,synapses=mode,n_driven=len(indices),
                                         input_l2_norm=float(np.linalg.norm(input_weights.astype(float))),
                                         node_indices=json.dumps(indices.tolist()),
                                         initial_radius_target=settings["initial_radius"],raw_rho=info["raw_rho"],
                                         initial_post_rho=info["post_rho"]*settings["initial_radius"],
                                         applied_initial_scale=info["scale_factor"]*settings["initial_radius"],
                                         network_sha256=network_hash,**metrics,**details))
                        delay_rows.extend(dict(condition_id=condition_id,repeat_id=repeat,delay=d+1,capacity=float(score)) for d,score in enumerate(curve))
                        for key,value in last.items():cache[f"condition_{condition_id:03d}_{key}"]=value
                        cache[f"condition_{condition_id:03d}_input_weights"]=input_weights
            np.testing.assert_array_equal(weights,original)
    pd.DataFrame(delay_rows).to_csv(out/"cache"/f"repeat_{repeat:03d}_curves.csv",index=False)
    np.savez_compressed(out/"cache"/f"repeat_{repeat:03d}_data_and_networks.npz",**cache)
    pd.DataFrame(rows).to_csv(complete,index=False)
    return repeat,len(rows),f"{time.monotonic()-started:.1f}s"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,default=ROOT/"analysis/stp_mc_narma_full_sweep_2026-11-18")
    p.add_argument("--n-repeats",type=int,default=8)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if args.n_repeats<2 or args.workers<1:p.error("Require >=2 repeats and positive workers.")
    if version("reservoirpy")!="0.4.2":raise ValueError("Requires the original ReservoirPy 0.4.2 NARMA protocol.")
    out=args.out_dir.resolve();(out/"cache").mkdir(parents=True,exist_ok=True)
    sensory=pd.read_csv(SENSORY_SOURCE).node_index.to_numpy()
    sources=[ROOT/"Connectome/ce_adj.npy",ROOT/"Connectome/ce_unknown_sign_weights.npy",ROOT/"Connectome/ce_nodes.txt",SENSORY_SOURCE]
    code=["tools/run_stp_mc_narma_pilot.py","network_stats/short_term_plasticity.py","network_stats/temporal_tasks.py",
          "network_stats/run_one.py","network_stats/stats.py","network_stats/stats_util.py","util/sensory_input.py","util/util.py","mc_narma_crossover.py"]
    manifest=dict(settings=SETTINGS,n_repeats=args.n_repeats,task=asdict(TemporalTaskSettings()),stp_presets={k:asdict(v) for k,v in PRESETS.items()},
                  source_hashes={str(path.relative_to(ROOT)):sha(path) for path in sources},code_hashes={name:sha(ROOT/name) for name in code},
                  packages={name:version(name) for name in ("numpy","torch","scipy","numba","reservoirpy","scikit-learn")},
                  device="cpu",sensory_indices=sensory.tolist(),
                  radius_grid=False,dynamic_renormalization=False,
                  model_sources=["https://pmc.ncbi.nlm.nih.gov/articles/PMC7321806/","https://pmc.ncbi.nlm.nih.gov/articles/PMC3105243/"],
                  model_adapter="Nonnegative rate proxy abs(signed tanh activity); per-presynaptic STP state; recovered-rest gain normalization; RK4 step 1/4; generic parameters.",
                  scope="Exploratory eight-seed pilot; no parameter optimization; full input and matched-total-drive sensory CE/random ER.")
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError("Incompatible run; choose a new directory.")
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"No spectral-radius grid. {args.n_repeats*144} paired MC/NARMA condition evaluations; {len(sensory)} sensory inputs.",flush=True)
    # Warm the Numba implementation once before launching worker processes.
    run_stp_states(np.zeros((3,3),np.float32),np.ones((3,1),np.float32),np.zeros((5,1),np.float32),PRESETS["facilitating"],1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(repeat,str(out),sensory.tolist(),SETTINGS) for repeat in range(args.n_repeats)]
        for future in as_completed([pool.submit(run_repeat,job) for job in jobs]):
            repeat,count,duration=future.result();print(f"[repeat {repeat:02d}] {count} MC/NARMA pairs; {duration}",flush=True)
    data=pd.concat([pd.read_csv(out/"cache"/f"repeat_{r:03d}.csv") for r in range(args.n_repeats)],ignore_index=True)
    curves=pd.concat([pd.read_csv(out/"cache"/f"repeat_{r:03d}_curves.csv") for r in range(args.n_repeats)],ignore_index=True)
    data.to_csv(out/"performance.csv",index=False);curves.to_csv(out/"memory_curves.csv",index=False)
    if manifest["source_hashes"]!={str(path.relative_to(ROOT)):sha(path) for path in sources}:raise ValueError("Sources changed.")
    print(f"Completed {len(data)} condition pairs. Saved {out}",flush=True)


if __name__=="__main__":main()
