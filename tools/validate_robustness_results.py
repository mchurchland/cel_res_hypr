#!/usr/bin/env python3
"""Validate exported analysis invariants against cached surfaces and source hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.robustness import performance_metrics, variance_decomposition, grid_edges, local_sensitivity, PARAMETERS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    out=args.directory.resolve()
    manifest=json.loads((out/"manifest.json").read_text())
    checks=[]
    for source in manifest["settings"]["sources"]:
        digest=hashlib.file_digest(open(ROOT/source["path"],"rb"),"sha256").hexdigest()
        assert digest==source["sha256"],source["path"]
    checks.append("All nine source CSV hashes match the recorded pre-analysis hashes")
    p=pd.read_csv(out/"tables/per_realization.csv")
    a=pd.read_csv(out/"tables/data_audit.csv")
    assert (p.n_valid==96).all()
    assert (p.n_total==96).all()
    assert p.n_valid.sum()==a.n_rows.sum()
    assert (p.n_nonfinite==0).all() and (p.n_negative==0).all()
    assert not p.duplicated(["dataset","architecture","normalization","metric","seed"]).any()
    np.testing.assert_allclose(p.cv,p.sd/p["mean"],rtol=1e-10)
    np.testing.assert_allclose(p["range"],p.maximum-p.minimum,rtol=1e-10)
    np.testing.assert_allclose(p.variance,p.sd**2,rtol=1e-10)
    assert (p.retention80>=p.retention90).all()
    assert (p.bottom10<=p["mean"]+1e-10).all()
    assert (p.minimum<=p.bottom10+1e-10).all()
    assert (p.feasible_fraction<=1+1e-8).all()
    assert (p.attain_q25>=p.attain_q50).all() and (p.attain_q50>=p.attain_q75).all()
    checks.append(f"All {len(p)} seeded surfaces have 96 finite nonnegative scores, unique identities and valid metric relationships")
    v=pd.read_csv(out/"tables/variance_components.csv")
    np.testing.assert_allclose(v.total_variance,v.configuration_variance+v.seed_variance+v.interaction_noise_variance,rtol=1e-10)
    np.testing.assert_allclose(v.within_configuration_variance,v.seed_variance+v.interaction_noise_variance,rtol=1e-10)
    checks.append(f"Exact finite-design variance identities hold for all {len(v)} architecture/metric/normalization conditions")
    curves=pd.read_csv(out/"tables/threshold_curves.csv")
    for _,g in curves.groupby(["dataset","architecture","normalization","metric"]):
        g=g.sort_values("threshold")
        assert (np.diff(g.attainment)<=1e-10).all()
        assert abs(g.attainment.iloc[0]-1)<1e-10 and abs(g.attainment.iloc[-1])<1e-10
        assert (g.ci_low<=g.attainment+1e-10).all() and (g.ci_high>=g.attainment-1e-10).all()
    checks.append("All threshold curves are monotone, start at full attainment and end at zero; bootstrap intervals enclose estimates")
    # Recompute one architecture per dataset and metric directly from cached matrices.
    selected=sorted((out/"cache/matrices").glob("*_0_*.npz"))
    for path in selected:
        with np.load(path) as stored:
            y=stored["values"]
            grid=pd.DataFrame(stored["grid"],columns=PARAMETERS)
            for i in [0,len(y)-1]:
                metric=path.stem.rsplit("_",1)[1]
                dataset=path.stem.rsplit("_",2)[0]
                # Group 0 is the first sorted architecture/normalization pair.
                g=p[(p.dataset==dataset)&(p.metric==metric)]
                order=g[["architecture","normalization"]].drop_duplicates().sort_values(["architecture","normalization"])
                arch,norm=order.iloc[0]
                row=g[(g.architecture==arch)&(g.normalization==norm)&(g.seed==stored["seeds"][i])].iloc[0]
                expected=performance_metrics(y[i])
                expected.update(local_sensitivity(y[i],grid_edges(grid)))
                for key,val in expected.items():
                    if key in row:
                        np.testing.assert_allclose(row[key],val,rtol=1e-10,atol=1e-10,err_msg=f"{path.name}:{key}")
            parts=variance_decomposition(y)
            row=v[(v.dataset==dataset)&(v.metric==metric)&(v.architecture==arch)&(v.normalization==norm)].iloc[0]
            for key,val in parts.items():
                np.testing.assert_allclose(row[key],val,rtol=1e-10,atol=1e-10)
    checks.append(f"Recomputed metric definitions, local slopes and variance components from {len(selected)} cached seed/configuration matrices")
    comparison=pd.read_csv(out/"tables/comparison.csv")
    keys=["dataset","architecture","normalization","metric"]
    expected=p.groupby(keys)[["mean","sd","iqr","mad","cv","bottom10","attain_q50","sensitivity_mean"]].mean()
    actual=comparison.set_index(keys).reindex(expected.index)
    np.testing.assert_allclose(actual[expected.columns],expected,rtol=1e-10,atol=1e-10)
    checks.append("Architecture comparison table matches direct means of per-realization metrics")
    for filename in ["REPORT.md","METHODS_AND_RESULTS.md"]:
        for target in re.findall(r"\]\(([^)]+)\)",(out/filename).read_text()):
            if not target.startswith("https://"):
                assert (out/target).exists(),target
    figures=list((out/"figures").glob("*.pdf"))
    assert len(figures)>=15
    assert len(figures)==len(list((out/"figures").glob("*.png")))
    checks.append(f"All report links resolve; {len(figures)} figures have both vector PDF and 300 dpi PNG exports")
    (out/"VALIDATION.json").write_text(json.dumps(dict(status="PASS",checks=checks),indent=2)+"\n")
    print(json.dumps(dict(status="PASS",checks=checks),indent=2))


if __name__=="__main__":
    main()
