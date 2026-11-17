"""Restore recorded threshold coordinates after a 12-digit CSV export.

This changes only serialization of the generated analysis cache, never input
results or attainment estimates. Original extraction stores each seed's curve
in ascending coordinate order, which is checked before restoring full precision.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parents[1]
manifest=json.loads((out/"manifest.json").read_text())
keys=["dataset","architecture","normalization","metric"]
path=out/"cache/threshold_per_seed.csv.gz"
d=pd.read_csv(path,dtype={key:"category" for key in keys})
for metric,spec in manifest["settings"]["thresholds"].items():
    mask=d.metric==metric
    group=d.loc[mask]
    levels=np.asarray(spec["curve"])
    size=group.groupby(keys+["seed"],observed=True).size()
    assert (size==len(levels)).all()
    position=group.groupby(keys+["seed"],sort=False,observed=True).cumcount().to_numpy()
    expected=levels[position]
    np.testing.assert_allclose(group.threshold,expected,rtol=1e-10,atol=1e-10)
    d.loc[mask,"threshold"]=expected
assert not d.duplicated(keys+["seed","threshold"]).any()
d.to_csv(path,index=False,float_format="%.17g",compression={"method":"gzip","compresslevel":1})
check=pd.read_csv(path,dtype={key:"category" for key in keys})
assert not check.duplicated(keys+["seed","threshold"]).any()
manifest["export_review"]["cache_serialization_repair"]="code_snapshot/repair_threshold_cache.py; all attainment values unchanged; full recorded coordinates restored and CSV round-trip uniqueness verified"
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Restored and verified {len(d)} cached threshold records")
