#!/usr/bin/env python3
"""Zoom performance-AUC axes without changing scores or dropping outliers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tools.plot_performance_auc import plt,performance_panel,short_label,save,COLORS,METRIC_NAMES
from tools.plot_architecture_duels_by_model import MODES
from architecture_mc_narma import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--auc-dir',type=Path,required=True)
    p.add_argument('--metric',choices=['MC','IPC','NARMA10'],required=True)
    p.add_argument('--x-min',type=float,required=True);p.add_argument('--x-max',type=float,required=True);a=p.parse_args()
    if not np.isfinite([a.x_min,a.x_max]).all() or a.x_min>=a.x_max:raise ValueError('Invalid display bounds')
    metric=next(k for k,v in METRIC_NAMES.items() if v==a.metric);base=a.auc_dir.resolve()
    prefix=f'performance_auc_{a.metric}';target=base/f'{prefix}.png';backup=base/f'{prefix}_full_range.png'
    record_path=base/f'{prefix}_xlim.json';source=base/'performance_curve_auc_summary.csv'
    manifest=base/'performance_auc_manifest.json';old=json.loads(manifest.read_text())
    main_path=base/'manifest.json';parent=json.loads(main_path.read_text())
    if backup.exists() or record_path.exists():raise FileExistsError('Bounds already adjusted; preserve original records')
    for path in [source,target]:
        if file_digest(path)!=old['outputs'][path.name]:raise ValueError(f'Source changed: {path}')
    c=json.loads((Path(old['source_analysis'])/'manifest.json').read_text())['config']
    d=pd.read_csv(source,dtype={'input_scale':str});d=d[(d.metric==metric)&(d.input_scale=='ALL')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True,sharex=True,layout='constrained');off_scale=[]
    for ax,(norm,(label,_,_)) in zip(axes,MODES.items()):
        group=d[d.normalization==norm];performance_panel(ax,group,c,metric)
        ax.set_xscale('linear');ax.xaxis.set_major_locator(MaxNLocator(nbins=8));ax.set_xlim(a.x_min,a.x_max);ax.set_title(label)
        for y,family in enumerate(c['families']):
            row=group[group.family==family].iloc[0];lo=float(row.ci_low);hi=float(row.ci_high);mean=float(row.performance_auc)
            if lo>=a.x_min and hi<=a.x_max and a.x_min<=mean<=a.x_max:continue
            off_scale.append(dict(normalization=norm,family=family,mean=mean,ci_low=lo,ci_high=hi))
            if mean<a.x_min or mean>a.x_max:
                boundary=a.x_min if mean<a.x_min else a.x_max
                ax.plot(boundary,y,marker='<' if mean<a.x_min else '>',ms=8,color=COLORS[family],clip_on=False)
                ax.text(a.x_min+.015*(a.x_max-a.x_min),y-.25,f'Mean {mean:.3g}; 95% CI [{lo:.3g}, {hi:.3g}]',
                    ha='left',va='bottom',fontsize=8,color=COLORS[family],bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=1))
            else:
                for boundary,marker in [(a.x_min,'<'),(a.x_max,'>')]:
                    if lo<boundary<hi:ax.plot(boundary,y,marker=marker,ms=6,color=COLORS[family],clip_on=False)
    axes[0].set(yticks=np.arange(len(c['families'])),yticklabels=[short_label(f) for f in c['families']]);axes[0].invert_yaxis()
    fig.suptitle(f'{a.metric}: AUC of mean-performance curves\nCommon linear x-axis {a.x_min:g}–{a.x_max:g}; arrows mark off-scale values; 95% paired-realization intervals',fontsize=13)
    temporary=base/f'{prefix}_xlim.tmp.png';save(fig,base,temporary.name)
    shutil.copyfile(target,backup);shutil.copyfile(manifest,base/f'performance_auc_manifest.before_{a.metric}_xlim.json')
    shutil.copyfile(main_path,base/f'manifest.before_{a.metric}_xlim.json');temporary.replace(target)
    record=dict(target=target.name,x_limits=[a.x_min,a.x_max],x_scale='linear',observations_removed=0,off_scale_values=off_scale,
        unrestricted_figure=backup.name,unrestricted_sha256=file_digest(backup),updated_sha256=file_digest(target),
        source=source.name,source_sha256=file_digest(source),script_sha256=file_digest(Path(__file__)))
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    old['outputs'][target.name]=file_digest(target);old['outputs'][backup.name]=file_digest(backup)
    old.setdefault('figure_adjustments',[]).append(dict(target=target.name,record=record_path.name,record_sha256=file_digest(record_path)))
    manifest.write_text(json.dumps(old,indent=2)+'\n')
    for extension in parent['extensions']:
        if extension['manifest']==manifest.name:extension['sha256']=file_digest(manifest)
    main_path.write_text(json.dumps(parent,indent=2)+'\n')
    assert file_digest(source)==record['source_sha256']
    print(f'{target}: linear x-axis {a.x_min:g}–{a.x_max:g}; {len(off_scale)} off-scale rows labelled; original: {backup.name}')


if __name__=='__main__':main()
