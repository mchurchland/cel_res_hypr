#!/usr/bin/env python3
"""Plot the saved video realization's spectra; no new reservoir simulations."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-eigenvalue-magnitude-sweep")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-eigenvalue-magnitude-sweep")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=("cel_matched","matched_er")
LABELS=("C. elegans","ER (binary weights)")
COLORS=("#332288","#0077bb")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"analysis/ei_network_mode_video")
    parser.add_argument("--out-dir",type=Path,default=ROOT/"analysis/eigenvalue_magnitude_full_sweep_2026-11-18")
    args=parser.parse_args()
    source=args.source_dir
    files=[source/"networks_and_trajectories.npz",source/"manifest.json",source/"mode_diagnostics.csv"]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    manifest=json.loads(files[1].read_text())
    with np.load(files[0]) as cache:
        eig=cache["eigenvalues"].copy()
    fractions=np.asarray(manifest["fractions"])
    rho=manifest["settings"]["rho"]
    if eig.shape[:2]!=(2,len(fractions)) or eig.shape[-1]!=297 or not np.isfinite(eig).all():
        raise ValueError("Unexpected or nonfinite cached spectra.")
    radii=np.max(np.abs(eig),axis=-1)
    diagnostics=pd.read_csv(files[2])
    for f,family in enumerate(FAMILIES):
        expected=diagnostics[diagnostics.family==family].sort_values("ei_percent").post_rho.to_numpy()
        np.testing.assert_allclose(radii[f],expected,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(radii,rho,rtol=2e-5,atol=2e-5)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    out=args.out_dir
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
    bins=np.linspace(0,1,41)
    summaries,values=[],[]
    for f,family in enumerate(FAMILIES):
        for c,fraction in enumerate(fractions):
            magnitudes=np.sort(np.abs(eig[f,c]))[::-1]
            summaries.append(dict(family=family,ei_percent=round(fraction*100,8),n_eigenvalues=len(magnitudes),
                                  largest=magnitudes[0],second_largest=magnitudes[1],mean=magnitudes.mean(),
                                  median=np.median(magnitudes),q10=np.quantile(magnitudes,.1),q90=np.quantile(magnitudes,.9),
                                  count_above_05=int(np.sum(magnitudes>.5)),
                                  count_above_08=int(np.sum(magnitudes>.8))))
            for rank,value in enumerate(magnitudes,1):
                values.append(dict(family=family,ei_percent=round(fraction*100,8),descending_rank=rank,magnitude=value))
    pd.DataFrame(summaries).to_csv(out/"spectral_summary.csv",index=False)
    pd.DataFrame(values).to_csv(out/"eigenvalue_magnitudes.csv",index=False)
    subtitle=f"Own-radius normalization · target {rho:g} · same illustrative network pair as the video"
    for kind in ("histograms","ecdfs","ranked"):
        fig,axes=plt.subplots(2,3,figsize=(12,7.3),sharex=True,sharey=True,layout="constrained")
        for c,(ax,fraction) in enumerate(zip(axes.flat,fractions)):
            for f,(label,color) in enumerate(zip(LABELS,COLORS)):
                x=np.sort(np.abs(eig[f,c]))
                if kind=="histograms":
                    counts,_=np.histogram(x,bins)
                    ax.stairs(counts/len(x),bins,color=color,lw=1.8,label=label)
                    ax.stairs(counts/len(x),bins,color=color,fill=True,alpha=.10)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                elif kind=="ecdfs":
                    ax.step(np.r_[0,x,1],np.r_[0,np.arange(1,len(x)+1)/len(x),1],where="post",color=color,lw=1.8,label=label)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                else:
                    ax.plot(np.arange(1,len(x)+1),x[::-1],color=color,lw=1.8,label=label)
            ax.set_title(f"{fraction:.0%} inhibitory edges",fontsize=12)
            ax.grid(axis="y",alpha=.15)
            if kind=="ranked":
                ax.set_xscale("log")
                ax.set_xlim(1,297)
                ax.set_xticks([1,3,10,30,100,297],labels=[1,3,10,30,100,297])
                ax.set_ylim(0,1)
                ax.axhline(rho,ls=":",color=".6",lw=.8)
            else:
                ax.set_xlim(0,1)
                ax.axvline(rho,ls=":",color=".6",lw=.8)
            if kind=="ecdfs":
                ax.set_ylim(0,1.02)
            if c>=3:
                ax.set_xlabel("Descending eigenvalue rank (log scale)" if kind=="ranked" else "Eigenvalue magnitude |λ|")
            if c%3==0:
                ax.set_ylabel({"histograms":"Fraction of eigenvalues per bin", "ecdfs":"Fraction with magnitude ≤ |λ|", "ranked":"Eigenvalue magnitude |λ|"}[kind])
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
        title={"histograms":"Eigenvalue-magnitude distributions across the full E/I sweep",
               "ecdfs":"Cumulative eigenvalue-magnitude distributions",
               "ranked":"Largest modes and the full eigenvalue-magnitude spectrum"}[kind]
        fig.suptitle(title+"\n"+subtitle,fontsize=13)
        fig.savefig(out/f"eigenvalue_magnitude_{kind}.png")
        plt.close(fig)
    validation=dict(n_models=2,n_fractions=len(fractions),n_eigenvalues_per_matrix=297,
                    total_eigenvalues=int(eig.size),all_finite=bool(np.isfinite(eig).all()),
                    cached_radii_match_diagnostics=True,max_radius_target_error=float(np.max(abs(radii-rho))),
                    histogram_bin_edges=bins.tolist(),sources=hashes,
                    settings=manifest["settings"],fractions=fractions.tolist(),
                    plotting_code_sha256=sha(Path(__file__)),
                    interpretation="One paired illustrative realization; eigenvalues are not independent network replicates.",
                    simulations_run=0)
    if hashes!={str(p.relative_to(ROOT)):sha(p) for p in files}:
        raise ValueError("Source changed while plotting.")
    (out/"manifest.json").write_text(json.dumps(validation,indent=2)+"\n")
    (out/"README.md").write_text(
        "# Eigenvalue-magnitude distributions across the full broad E/I sweep\n\n"
        "Uses all six saved broad-sweep fractions: 0, 20, 40, 60, 80 and 100% inhibitory edges. "
        "These are the same CE/ER matrix pairs used in the network video; no new trajectories or simulations are run.\n\n"
        "All figures use identical axes across architectures and fractions. Histograms use common width-0.025 bins "
        "and show fractions of each matrix's 297 eigenvalues. ECDFs avoid bin choices. Ranked curves use a log rank axis "
        "to make the leading modes visible while retaining the full spectrum. The dotted reference marks radius 0.95.\n\n"
        "These are post-normalization spectra at an own-radius target of 0.95. The largest magnitude is fixed by "
        "normalization; differences concern the remaining spectrum. Input scale 0.1, leak 1 and bias 0 describe the "
        "associated video dynamics, but the recurrent-weight spectra themselves do not depend on input scale or bias.\n\n"
        "CE retains empirical magnitudes with unknown edges restored; ER is the existing binary-weight matched "
        "sign-sweep control. One illustrative realization (base seed 20261117), not uncertainty over seeds or "
        "evidence of an MC association. The finer 20–40% crossover experiment is a separate grid.\n\n"
        "Exact sorted values: `eigenvalue_magnitudes.csv`. Descriptive quantities: `spectral_summary.csv`. "
        "Sources, hashes and settings: `manifest.json`. PNG only; existing outputs are unchanged.\n")
    print(pd.DataFrame(summaries).round(4).to_string(index=False))
    print(f"Saved three PNG figures and spectral tables to {out}")


if __name__=="__main__":
    main()
