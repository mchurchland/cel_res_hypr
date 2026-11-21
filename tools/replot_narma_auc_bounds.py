#!/usr/bin/env python3
"""Tighten NARMA performance-AUC display bounds, preserving unrestricted data."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from tools.plot_performance_auc import plt,performance_panel,short_label,save
from tools.plot_architecture_duels_by_model import MODES
from architecture_mc_narma import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--auc-dir',type=Path,required=True)
    p.add_argument('--x-min',type=float,default=.135);p.add_argument('--x-max',type=float,default=.28);a=p.parse_args()
    if not np.isfinite([a.x_min,a.x_max]).all() or a.x_min>=a.x_max:raise ValueError('Invalid display bounds')
    base=a.auc_dir.resolve();target=base/'performance_auc_NARMA10.png';backup=base/'performance_auc_NARMA10_full_range.png'
    metadata=base/'performance_auc_NARMA10_xlim.json';source=base/'performance_curve_auc_summary.csv'
    manifest=base/'performance_auc_manifest.json';old=json.loads(manifest.read_text())
    main_path=base/'manifest.json';parent=json.loads(main_path.read_text())
    if backup.exists() or metadata.exists():raise FileExistsError('Bounds already adjusted; preserve the recorded originals')
    for path in [source,target]:
        if file_digest(path)!=old['outputs'][path.name]:raise ValueError(f'Source changed: {path}')
    c=json.loads((Path(old['source_analysis'])/'manifest.json').read_text())['config']
    d=pd.read_csv(source,dtype={'input_scale':str});d=d[(d.metric=='NARMA10_test_nmse')&(d.input_scale=='ALL')]
    fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True,sharex=True,layout='constrained');clipped=[]
    for ax,(norm,(label,_,_)) in zip(axes,MODES.items()):
        group=d[d.normalization==norm];performance_panel(ax,group,c,'NARMA10_test_nmse');ax.set_title(label)
        ax.set_xlim(a.x_min,a.x_max);ax.set_xticks(np.arange(.14,.281,.02))
        for y,family in enumerate(c['families']):
            row=group[group.family==family].iloc[0]
            if row.ci_low<a.x_min or row.ci_high>a.x_max or not a.x_min<=row.performance_auc<=a.x_max:
                clipped.append(dict(normalization=norm,family=family,mean=float(row.performance_auc),ci_low=float(row.ci_low),ci_high=float(row.ci_high)))
                boundary=a.x_max if row.performance_auc>a.x_max else a.x_min
                ax.plot(boundary,y,marker='>' if boundary==a.x_max else '<',ms=8,color='#0072b2',clip_on=False)
                ax.text(a.x_max-.003,y-.25,f'Mean {row.performance_auc:.2f}; 95% CI [{row.ci_low:.2f}, {row.ci_high:.2f}]',
                        ha='right',va='bottom',fontsize=8,color='#0072b2',bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
    axes[0].set(yticks=np.arange(len(c['families'])),yticklabels=[short_label(f) for f in c['families']]);axes[0].invert_yaxis()
    fig.suptitle(f'NARMA10: AUC of mean-performance curves\nCommon x-axis zoom {a.x_min:g}–{a.x_max:g}; arrow marks off-scale estimate; 95% paired-realization intervals',fontsize=13)
    temporary=base/'performance_auc_NARMA10_xlim.tmp.png';save(fig,base,temporary.name)
    shutil.copyfile(target,backup);shutil.copyfile(manifest,base/'performance_auc_manifest.before_narma_xlim.json')
    shutil.copyfile(main_path,base/'manifest.before_narma_xlim.json');temporary.replace(target)
    record=dict(target=target.name,x_limits=[a.x_min,a.x_max],observations_removed=0,off_scale_values=clipped,
                unrestricted_figure=backup.name,unrestricted_sha256=file_digest(backup),updated_sha256=file_digest(target),
                source=source.name,source_sha256=file_digest(source),script_sha256=file_digest(Path(__file__)))
    metadata.write_text(json.dumps(record,indent=2)+'\n')
    old['outputs'][target.name]=file_digest(target);old['outputs'][backup.name]=file_digest(backup)
    old.setdefault('figure_adjustments',[]).append(dict(target=target.name,record=metadata.name,record_sha256=file_digest(metadata)))
    manifest.write_text(json.dumps(old,indent=2)+'\n')
    for extension in parent['extensions']:
        if extension['manifest']==manifest.name:extension['sha256']=file_digest(manifest)
    main_path.write_text(json.dumps(parent,indent=2)+'\n')
    assert file_digest(source)==record['source_sha256']
    print(f'{target}: x limits {a.x_min:g}–{a.x_max:g}; unrestricted figure saved as {backup.name}')


if __name__=='__main__':main()
