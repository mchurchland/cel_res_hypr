#!/usr/bin/env python3
"""Adjust one IPC mean figure; retain outliers in an inset and archive originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/architecture-sweep-xdg-cache')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from tools.analyze_architecture_mc_narma import LABELS,COLORS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--y-min',type=float,default=-60)
    p.add_argument('--y-max',type=float,default=70)
    args=p.parse_args();base=args.analysis_dir.resolve()
    if not np.isfinite([args.y_min,args.y_max]).all() or args.y_min>=args.y_max:raise ValueError('Invalid y limits')
    target=base/'mean_IPC_own_radius.png';backup=base/'mean_IPC_own_radius_full_range.png'
    meta=base/'mean_IPC_own_radius_zoom.json';manifest=base/'manifest.json'
    old_manifest=base/'manifest.before_ipc_ylim.json'
    if any(f.exists() for f in [backup,meta,old_manifest]):raise FileExistsError('Figure already adjusted; preserve its recorded provenance')
    original=json.loads(manifest.read_text());source=base/'mean_performance.csv'
    if sha(source)!=original['outputs'][source.name] or sha(target)!=original['outputs'][target.name]:raise ValueError('Source/figure differs from recorded hashes')
    all_data=pd.read_csv(source)
    d=all_data[(all_data.metric=='IPC')&(all_data.normalization=='own_radius')].copy()
    if not np.isfinite(d[['mean','ci_low','ci_high']].to_numpy()).all():raise ValueError('Nonfinite figure values')
    families=d.family.unique()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':8})
    fig,ax=plt.subplots(figsize=(9,5.8))
    inset=ax.inset_axes([.66,.055,.31,.29]);inset.set_facecolor('white');inset.set_zorder(8)
    clipped=[]
    for family in families:
        g=d[d.family==family].sort_values('negative_fraction');x=g.negative_fraction*100;color=COLORS[family]
        ax.plot(x,g['mean'],marker='o',ms=3,color=color,label=LABELS[family])
        ax.fill_between(x,g.ci_low,g.ci_high,color=color,alpha=.12)
        inset.plot(x,g['mean'],color=color,lw=.8)
        inset.fill_between(x,g.ci_low,g.ci_high,color=color,alpha=.08,lw=0)
        for r in g.itertuples():
            if r.mean<args.y_min or r.mean>args.y_max:
                boundary=args.y_min if r.mean<args.y_min else args.y_max
                ax.plot(r.negative_fraction*100,boundary,marker='v' if r.mean<args.y_min else '^',color=color,ms=6,clip_on=False)
                if family=='ws_signed_empirical':
                    ax.annotate(f'WS at {100*r.negative_fraction:g}%: mean {r.mean:.1f}',xy=(100*r.negative_fraction,boundary),
                        xytext=(100*r.negative_fraction-13,args.y_min+15),fontsize=8,color=color,
                        arrowprops=dict(arrowstyle='->',color=color,lw=.8))
                clipped.append(dict(family=family,negative_fraction=r.negative_fraction,mean=r.mean,ci_low=r.ci_low,ci_high=r.ci_high))
            elif r.ci_low<args.y_min:
                ax.plot(r.negative_fraction*100,args.y_min,marker='v',ms=5,color=color,clip_on=False)
                ax.annotate(f'Lower CI: {r.ci_low:.1f}',xy=(100*r.negative_fraction,args.y_min),
                    xytext=(100*r.negative_fraction+2,args.y_min+8),fontsize=7,color=color,
                    arrowprops=dict(arrowstyle='->',color=color,lw=.7))
                clipped.append(dict(family=family,negative_fraction=r.negative_fraction,mean=r.mean,ci_low=r.ci_low,ci_high=r.ci_high))
    ax.axhline(0,color='.5',lw=.8)
    ax.set(xlabel='Inhibitory edges (%)',ylabel='Held-out IPC estimate (higher is better)',
        title='Own-radius scaling: mean IPC across the grid',ylim=(args.y_min,args.y_max),xlim=(-2,102),xticks=np.arange(0,101,10))
    ax.grid(axis='y',alpha=.15)
    inset.axhspan(args.y_min,args.y_max,color='.5',alpha=.12)
    inset.set_title('Full range: same data',fontsize=7,pad=3)
    inset.tick_params(labelsize=6,length=2);inset.set_xticks([0,50,100]);inset.yaxis.set_major_locator(MaxNLocator(3))
    inset.set_xlim(-2,102);inset.grid(alpha=.1)
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False)
    fig.text(.5,-.025,'Pointwise 95% paired-realization bootstrap intervals. Zoomed main axis; full-range inset retains outliers. No data removed.',ha='center',fontsize=7.5)
    fig.tight_layout()
    temporary=base/'mean_IPC_own_radius.zoom.tmp.png'
    fig.savefig(temporary,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    shutil.copyfile(target,backup);shutil.copyfile(manifest,old_manifest)
    temporary.replace(target)
    record=dict(source=source.name,source_sha256=sha(source),y_limits=[args.y_min,args.y_max],full_range_inset=True,
        observations_removed=0,clipped_main_panel_values=clipped,original_figure=backup.name,original_sha256=sha(backup),
        updated_figure_sha256=sha(target),script=str(Path(__file__).relative_to(ROOT)),script_sha256=sha(Path(__file__)))
    meta.write_text(json.dumps(record,indent=2)+'\n')
    original['outputs'][target.name]=sha(target)
    original.setdefault('figure_adjustments',[]).append(dict(target=target.name,record=meta.name,record_sha256=sha(meta),original_manifest=old_manifest.name))
    manifest.write_text(json.dumps(original,indent=2)+'\n')
    assert sha(backup)==json.loads(old_manifest.read_text())['outputs'][target.name]
    assert sha(source)==record['source_sha256']
    print(target);print(f'Y limits: {args.y_min:g} to {args.y_max:g}; original preserved as {backup.name}')

if __name__=='__main__':main()
