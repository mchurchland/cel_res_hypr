#!/usr/bin/env python3
"""PNG comparison of C. elegans minus matched ER within normalization controls."""
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-robustness")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/robustness-xdg-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.robustness import bootstrap_mean


def sha(path):
    with open(path,"rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def main():
    base=ROOT/"analysis/robustness_reevaluation_2026-11-16"
    out=base/"normalization_model_comparison"
    if out.exists():
        raise FileExistsError("Preserve existing output; choose a new directory")
    path=base/"tables/per_realization.csv"
    data=pd.read_csv(path)
    data=data[data.dataset.isin(["norm_cel_matched","norm_matched_er"])].copy()
    data['negative_percent']=pd.to_numeric(data.architecture.str.extract(r"(?:sign_test_og_cel|sign_test_er)([0-9.]+)$",expand=False))*100
    metadata=json.loads((base/"manifest.json").read_text())
    sources=[s for s in metadata['settings']['sources'] if s['dataset'] in ['norm_cel_matched','norm_matched_er']]
    for source in sources:
        assert sha(ROOT/source['path'])==source['sha256']
    ce=data[data.dataset=='norm_cel_matched'];er=data[data.dataset=='norm_matched_er']
    keys=['negative_percent','normalization','metric','seed']
    pairs=ce.merge(er,on=keys,how='inner',suffixes=('_ce','_er'),validate='one_to_one')
    pairs['difference']=pairs['mean_ce']-pairs['mean_er']
    rng=np.random.default_rng(20261117)
    rows=[]
    for (percent,norm,metric),g in pairs.groupby(['negative_percent','normalization','metric']):
        assert len(g)==100 and g.n_valid_ce.eq(96).all() and g.n_valid_er.eq(96).all()
        estimate,interval=bootstrap_mean(g.difference,2000,rng)
        rows.append(dict(negative_percent=percent,normalization=norm,metric=metric,
                         ce_mean=g.mean_ce.mean(),er_mean=g.mean_er.mean(),ce_minus_er=float(estimate[0]),
                         ci_low=float(interval[0,0]),ci_high=float(interval[1,0]),n_paired_seeds=len(g),n_configurations=96))
    results=pd.DataFrame(rows)
    assert len(results)==36
    columns=pd.MultiIndex.from_product([['MC','IPC','KR'],['spectral_radius','original_radius']])
    table=results.pivot(index='negative_percent',columns=['metric','normalization'],values='ce_minus_er').reindex(columns=columns)
    assert table.notna().all().all() and list(table.index)==[0,20,40,60,80,100]
    out.mkdir()
    results.to_csv(out/'normalization_model_comparison_details.csv',index=False,float_format='%.12g')
    table.to_csv(out/'normalization_model_comparison.csv',float_format='%.12g')
    cells=[[f'{percent:.0f}%',*[f'{v:+.2f}' for v in row]] for percent,row in table.iterrows()]
    fig,ax=plt.subplots(figsize=(11.7,5.8))
    fig.subplots_adjust(left=.035,right=.965,top=.80,bottom=.04);ax.axis('off')
    fig.text(.035,.95,'C. elegans − matched ER: effect of normalization',fontsize=17,weight='semibold',color='#172b44')
    fig.text(.035,.88,'Mean across 96 configurations and 100 seeds · positive = C. elegans higher; negative = ER higher',fontsize=10.5,color='#526176')
    headers=['Inhibitory\nedges',*[f'{m}\n{label}' for m in ['MC','IPC','KR'] for label in ['Own radius','Fixed reference']]]
    rendered=ax.table(cellText=cells,colLabels=headers,cellLoc='center',colWidths=[.13]+[.145]*6,bbox=[0,.24,1,.73])
    rendered.auto_set_font_size(False);rendered.set_fontsize(12)
    for (r,c),cell in rendered.get_celld().items():
        cell.set_edgecolor('#dce3eb');cell.set_linewidth(.7)
        if r==0:
            cell.set_facecolor('#223b58');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        elif c==0:
            cell.set_facecolor('#f2f5f8');cell.get_text().set_weight('bold')
        else:
            value=table.iloc[r-1,c-1]
            cell.set_facecolor('#e6f2eb' if value>0 else '#f9eee6')
            cell.get_text().set_color('#205b3c' if value>0 else '#8b4423')
    fig.text(.035,.135,'Both scaling modes use the same archival normalization-control dataset; KR is separate from the updated 200-seed sign sweep.',fontsize=9,color='#526176')
    fig.text(.035,.085,'Only shared E/I levels are compared. No ER measurement exists at 24.25%; no interpolation is used.',fontsize=9,color='#526176')
    fig.text(.035,.035,'Colors indicate direction, not statistical significance. Paired seed-bootstrap 95% intervals are saved in the detail CSV.',fontsize=9,color='#526176')
    fig.savefig(out/'normalization_model_comparison.png',dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(sources=sources,summary_source=str(path.relative_to(ROOT)),summary_source_sha256=sha(path),code_sha256=sha(Path(__file__)),bootstrap=2000,bootstrap_seed=20261117,comparison='C. elegans (unknown signs matched) minus matched ER; paired seeds within each archival normalization mode',aggregation='Mean over 96 configurations within seed, followed by mean across 100 seeds',excluded='24.250832%: no shared matched-ER condition; no interpolation',validation='PASS: 36 comparisons, 100 paired seeds per comparison, 96 configurations per seed, shared sign levels only; source hashes verified',figure_format='PNG only'),indent=2)+'\n')
    print(table.to_string(float_format=lambda x:f'{x:+.3f}'))
    print(results[(results.negative_percent==0)&(results.metric=='KR')][['normalization','ce_minus_er','ci_low','ci_high']].to_string(index=False))
    print(out/'normalization_model_comparison.png')


if __name__=='__main__':
    main()
