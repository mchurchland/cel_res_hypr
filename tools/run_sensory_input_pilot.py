#!/usr/bin/env python3
"""Small local MC pilot: anatomical CE sensory inputs versus random ER inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
import numpy as np
import pandas as pd
import torch

from mc_narma_crossover import build_condition, load_completed_connectome
from network_stats.run_one import run_reservoir_states
from network_stats.stats import compute_MC
from util.sensory_input import place_input_weights, sensory_annotation_map

FAMILIES=("cel_matched","matched_er")
SETTINGS=dict(fractions=[.2,.4],input_scales=[.1,.5],rho=.95,leak=1.,bias=0.,
              normalization="spectral_radius",random_masks=3,washout=500,train=1500,test=500,
              max_delay=30,ridge_alpha=1e-4,seed=20261117,repeat_stride=100000,mask_seed=20261118,
              calibrations=["unchanged_coefficients","matched_total_norm"])
ANNOTATION_URL="https://raw.githubusercontent.com/openworm/owmeta/945f7172f0dff1d022ce0574f3c630ee53297386/OpenWormData/aux_data/Modified%20celegans%20db%20dump.csv"
PREVIOUS=ROOT/"analysis/spectral_memory_intervention_ei20_2026-11-18"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_repeat(job):
    repeat,out,sensory,settings=job
    out=Path(out)
    done=out/"cache"/f"repeat_{repeat:03d}.csv"
    if done.exists():
        return repeat,len(pd.read_csv(done)),"cached"
    torch.set_num_threads(1)
    start_time=time.monotonic()
    seed=settings["seed"]+repeat*settings["repeat_stride"]
    completed,_=load_completed_connectome(ROOT/"Connectome/ce_adj.npy",ROOT/"Connectome/ce_unknown_sign_weights.npy",seed)
    n=len(completed); sensory=np.asarray(sensory,dtype=int); k=len(sensory)
    u=np.random.default_rng(seed+31_000_003).uniform(-1,1,settings["washout"]+settings["train"]+settings["test"]).astype(np.float32)[:,None]
    u_tensor=torch.from_numpy(u)
    rows,curves=[],[]
    arrays={"input":u}
    masks={}
    for family_index,family in enumerate(FAMILIES):
        masks[family]=[np.random.default_rng(settings["mask_seed"]+repeat*10000+family_index*1000+draw).choice(n,k,replace=False)
                       for draw in range(settings["random_masks"])]
    previous_scores=pd.read_csv(PREVIOUS/"cache"/f"repeat_{repeat:03d}.csv")
    previous_curves=pd.read_csv(PREVIOUS/"cache"/f"repeat_{repeat:03d}_curves.csv")
    with np.load(PREVIOUS/"cache"/f"repeat_{repeat:03d}_networks.npz") as previous:
        previous_arrays={key:previous[key].copy() for key in previous.files}
    common_input=None
    for fraction in settings["fractions"]:
        for family in FAMILIES:
            base,win,info=build_condition(completed,family,fraction,settings["normalization"],seed,torch.device("cpu"))
            w=base*settings["rho"]
            original=w.numpy().copy()
            if common_input is None:
                common_input=win.numpy().copy()
                arrays["unit_dense_input_weights"]=common_input
            np.testing.assert_array_equal(win.numpy(),common_input)
            arrays[f"{family}_ei{round(fraction*100):03d}_weights"]=original
            network_hash=hashlib.sha256(original.tobytes()).hexdigest()
            for scale in settings["input_scales"]:
                dense=(win*scale).numpy()

                def evaluate(input_weights,placement,calibration,draw,indices):
                    cached=(fraction==.2 and scale==.1 and placement=="dense")
                    if cached:
                        np.testing.assert_array_equal(original,previous_arrays[f"{family}_weights"])
                        np.testing.assert_array_equal(dense,previous_arrays[f"{family}_input_weights"])
                        np.testing.assert_array_equal(u,previous_arrays["input"])
                        old=previous_scores[(previous_scores.family==family)&(previous_scores.protocol=="baseline")].iloc[0]
                        total=float(old.MC)
                        curve=previous_curves[(previous_curves.family==family)&(previous_curves.condition_id==old.condition_id)].sort_values("delay").capacity.to_numpy()
                        saturation=float(old.saturation_fraction)
                        state_variance=float(old.mean_state_variance)
                    else:
                        x=run_reservoir_states(w,torch.from_numpy(input_weights),u_tensor,settings["leak"])
                        start=settings["washout"]; end=start+settings["train"]
                        total,curve=compute_MC(x[start:end].double(),x[end:].double(),u_tensor[start:end].double(),u_tensor[end:].double(),
                                               settings["max_delay"],settings["ridge_alpha"],torch.device("cpu"))
                        saturation=float((x[start:].abs()>.9).float().mean())
                        state_variance=float(x[start:].var(dim=0,unbiased=False).mean())
                    if not np.isfinite(total) or not np.isfinite(curve).all():
                        raise ValueError("Nonfinite MC.")
                    if len(indices)!=np.count_nonzero(input_weights):
                        raise ValueError("Input mask does not match the nonzero drive count.")
                    norm=float(np.linalg.norm(input_weights.astype(float)))
                    full_norm=float(np.linalg.norm(dense.astype(float)))
                    if calibration=="matched_total_norm" and not np.isclose(norm,full_norm,rtol=1e-6):
                        raise ValueError("Input norm matching failed.")
                    condition_id=len(rows)
                    rows.append(dict(condition_id=condition_id,repeat_id=repeat,seed=seed,family=family,ei_percent=fraction*100,
                                     input_scale=scale,rho_target=settings["rho"],leak=settings["leak"],placement=placement,
                                     calibration=calibration,mask_draw=draw,n_driven=len(indices),node_indices=json.dumps(np.asarray(indices).tolist()),
                                     MC=total,input_l2_norm=norm,dense_input_l2_norm=full_norm,input_norm_ratio=norm/full_norm,
                                     active_input_rms=float(np.sqrt(np.mean(input_weights[np.asarray(indices)]**2))),
                                     fraction_saturated=saturation,mean_state_variance=state_variance,
                                     mean_driven_outdegree=float(np.count_nonzero(original,axis=1)[indices].mean()),
                                     network_sha256=network_hash,raw_rho=info["raw_rho"],baseline_reused=cached))
                    curves.extend(dict(condition_id=condition_id,repeat_id=repeat,delay=i+1,capacity=float(score)) for i,score in enumerate(curve))
                    arrays[f"input_weights_{condition_id:03d}"]=input_weights.copy()

                evaluate(dense,"dense","dense",-1,np.arange(n))
                for calibration in settings["calibrations"]:
                    if family=="cel_matched":
                        targeted=place_input_weights(dense,sensory,sensory,calibration)
                        evaluate(targeted,"sensory",calibration,-1,sensory)
                    for draw,indices in enumerate(masks[family]):
                        placed=place_input_weights(dense,sensory,indices,calibration)
                        evaluate(placed,"random",calibration,draw,indices)
            np.testing.assert_array_equal(w.numpy(),original)
    pd.DataFrame(curves).to_csv(out/"cache"/f"repeat_{repeat:03d}_curves.csv",index=False)
    np.savez_compressed(out/"cache"/f"repeat_{repeat:03d}_inputs_and_networks.npz",**arrays)
    pd.DataFrame(rows).to_csv(done,index=False)
    return repeat,len(rows),f"{time.monotonic()-start_time:.1f}s"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/sensory_input_pilot_2026-11-18")
    parser.add_argument("--annotation-csv",type=Path,help="Downloaded frozen OpenWorm CSV; copied into this run's sources folder.")
    parser.add_argument("--n-repeats",type=int,default=20)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if not 2<=args.n_repeats<=20 or args.workers<1:
        parser.error("This pilot reuses up to 20 existing paired baselines; require 2–20 repeats and positive workers.")
    out=args.out_dir.resolve()
    (out/"sources").mkdir(parents=True,exist_ok=True);(out/"cache").mkdir(exist_ok=True)
    annotation_path=out/"sources/openworm_types_945f717.csv"
    if not annotation_path.exists():
        if args.annotation_csv is None:
            parser.error(f"Download {ANNOTATION_URL} and pass --annotation-csv on the first run.")
        shutil.copyfile(args.annotation_csv,annotation_path)
    elif args.annotation_csv is not None and sha(annotation_path)!=sha(args.annotation_csv):
        raise ValueError("Annotation source differs from the existing cached source.")
    names=(ROOT/"Connectome/ce_nodes.txt").read_text().splitlines()
    if len(names)!=np.load(ROOT/"Connectome/ce_adj.npy").shape[0]:
        raise ValueError("Node-name count differs from adjacency dimension.")
    metadata=pd.read_csv(annotation_path)
    sensory,missing=sensory_annotation_map(metadata,names)
    sensory.to_csv(out/"sensory_neurons.csv",index=False)
    legacy=pd.read_excel(ROOT/"old_cel.xls",sheet_name="Sensory")
    legacy_names=set(legacy.Neuron)
    types=metadata[metadata.Relationship.eq("type")].groupby("Entity1").Entity2.agg(lambda x:" | ".join(sorted(set(x))))
    pd.DataFrame(dict(node_index=np.arange(len(names)),neuron=names,
                      selected_sensory=[name in set(sensory.neuron) for name in names],
                      legacy_sensory_sheet=[name in legacy_names for name in names],
                      type_annotation=[types.get(name,"") for name in names])).to_csv(out/"neuron_label_audit.csv",index=False)
    source_files=[ROOT/"Connectome/ce_adj.npy",ROOT/"Connectome/ce_unknown_sign_weights.npy",ROOT/"Connectome/ce_nodes.txt",ROOT/"old_cel.xls",annotation_path]
    code=["tools/run_sensory_input_pilot.py","util/sensory_input.py","mc_narma_crossover.py","util/util.py",
          "network_stats/run_one.py","network_stats/stats.py","network_stats/stats_util.py"]
    reused_files=[]
    for repeat in range(args.n_repeats):
        reused_files.extend(PREVIOUS/"cache"/f"repeat_{repeat:03d}{suffix}" for suffix in (".csv","_curves.csv","_networks.npz"))
    manifest=dict(settings=SETTINGS,n_repeats=args.n_repeats,device="cpu",n_nodes=len(names),n_sensory=len(sensory),
                  sensory_indices=sensory.node_index.tolist(),missing_sensory_names=missing,annotation_url=ANNOTATION_URL,
                  annotation_policy="Type starts 'Sensory neuron' or starts 'Polymodal neuron' and contains 'sensory neuron'; exact node-name mapping.",
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  reused_baseline_hashes={str(p.relative_to(ROOT)):sha(p) for p in reused_files},
                  code_hashes={name:sha(ROOT/name) for name in code},packages=dict(numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__),
                  experimental_unit="One computational realization; random input masks averaged within realization for inference.",
                  input_protocol="Same scalar signal; identical ordered Gaussian input coefficients on all equal-size masks; full-state MC readout.")
    path=out/"manifest.json"
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError("Run manifest differs. Use a new output folder.")
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Mapped {len(sensory)} sensory neurons of {len(names)}; missing {missing}. Expected evaluations: {args.n_repeats*64}",flush=True)
    jobs=[(repeat,str(out),sensory.node_index.tolist(),SETTINGS) for repeat in range(args.n_repeats)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_repeat,job) for job in jobs]):
            repeat,count,duration=future.result()
            print(f"[repeat {repeat:02d}] {count} conditions; {duration}",flush=True)
    results=pd.concat([pd.read_csv(out/"cache"/f"repeat_{repeat:03d}.csv") for repeat in range(args.n_repeats)],ignore_index=True)
    curves=pd.concat([pd.read_csv(out/"cache"/f"repeat_{repeat:03d}_curves.csv") for repeat in range(args.n_repeats)],ignore_index=True)
    results.to_csv(out/"performance.csv",index=False);curves.to_csv(out/"memory_curves.csv",index=False)
    if manifest["source_hashes"]!={str(p.relative_to(ROOT)):sha(p) for p in source_files}:
        raise ValueError("Input sources changed during the experiment.")
    print(f"Completed {len(results)} MC evaluations ({results.baseline_reused.sum()} existing baselines reused). Saved {out}",flush=True)


if __name__=="__main__":
    main()
