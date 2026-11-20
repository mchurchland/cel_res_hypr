#!/usr/bin/env python3
"""Plot CE matched-setting win fractions separately for every control model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-architecture-sweep')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/architecture-sweep-xdg-cache')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from tools.analyze_architecture_mc_narma import LABELS

MODES={'own_radius':('Own radius','#0072b2','o'),
       'fixed_reference':('Fixed reference','#d55e00','s'),
       'shared_ce_scale':('Shared CE multiplier','#009e73','^')}
METRICS={'MC':'MC','NARMA10_test_nmse':'NARMA10','IPC':'IPC'}
INPUT_COLORS=['#332288','#0072b2','#e69f00','#cc3311']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style(ax,fractions):
    ticks=100*np.asarray(sorted(fractions))
    ax.set(xlim=(ticks.min()-1,ticks.max()+1),ylim=(-3,103),xticks=ticks,yticks=[0,25,50,75,100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
    ax.axhline(50,color='.4',ls=':',lw=1,zorder=0)
    ax.grid(axis='y',alpha=.16)
    ax.spines[['top','right']].set_visible(False)


def draw(ax,w,control,metric):
    for mode,(label,color,marker) in MODES.items():
        g=w[(w.control==control)&(w.metric==metric)&(w.normalization==mode)].sort_values('negative_fraction')
        if g.empty:continue
        assert len(g)==w.negative_fraction.nunique()
        ax.plot(g.negative_fraction*100,g.ce_winning_fraction*100,label=label,color=color,
                marker=marker,ms=4,lw=1.7)
        ax.fill_between(g.negative_fraction*100,g.ci_low*100,g.ci_high*100,color=color,alpha=.12,lw=0)
    style(ax,w.negative_fraction.unique())


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,default=ROOT/'analysis/architecture_mc_narma/architecture_mc_narma_271846')
    p.add_argument('--out-dir',type=Path)
    args=p.parse_args(argv)
    base=args.analysis_dir.resolve();out=args.out_dir or base/'control_interpretation/duel_plots_by_model'
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; no overwrite')
    # Validate all source tables against the existing recorded checksums.
    original=json.loads((base/'manifest.json').read_text())
    for name in ['ce_winning_fraction.csv','configuration_advantages.csv']:
        if sha(base/name)!=original['outputs'][name]:raise ValueError(f'Source changed: {name}')
    w=pd.read_csv(base/'ce_winning_fraction.csv');a=pd.read_csv(base/'configuration_advantages.csv')
    assert w.n_configurations.nunique()==1 and w.n_realizations.nunique()==1
    nsettings=int(w.n_configurations.iloc[0]);nseeds=int(w.n_realizations.iloc[0])
    reconstructed=a.assign(won=a.ce_advantage>0).groupby(['normalization','metric','control','negative_fraction']).won.agg(['sum','count']).reset_index()
    check=w.merge(reconstructed,on=['normalization','metric','control','negative_fraction'],validate='one_to_one')
    np.testing.assert_allclose(check.ce_winning_fraction,check['sum']/check['count'])
    controls=[f for f in LABELS if f in set(w.control)]
    assert set(controls)==set(w.control)
    by_input=a.assign(ce_won=a.ce_advantage>0).groupby(['normalization','metric','control','negative_fraction','input_scale']).ce_won.agg(['sum','count']).reset_index()
    assert by_input['count'].nunique()==1
    ninput=int(by_input['count'].iloc[0])
    metrics={name:label for name,label in METRICS.items() if name in set(w.metric)}
    by_input['ce_win_percent']=100*by_input['sum']/by_input['count']
    out.mkdir(parents=True,exist_ok=True)
    by_input.to_csv(out/'input_win_curves.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'legend.fontsize':9})
    for metric,label in metrics.items():
        rows=max(1,(len(controls)+2)//3)
        fig,axes=plt.subplots(rows,3,figsize=(13,3.7*rows),sharex=True,sharey=True,squeeze=False)
        for ax,control in zip(axes.flat,controls):
            draw(ax,w,control,metric);ax.set_title(LABELS[control])
        fig.suptitle(f'{label}: fraction of matched hyperparameter settings CE wins',fontsize=16,y=.985)
        fig.supxlabel('Inhibitory edges (%)',y=.105)
        for ax in list(axes.flat)[len(controls):]:ax.set_visible(False)
        fig.supylabel(f'CE wins / {nsettings} matched settings (%)',x=.015)
        handles,names=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
        fig.text(.5,.012,f'{nseeds} paired realizations per setting; shading: pointwise 95% realization-bootstrap intervals. Dotted line: half the settings.',ha='center',fontsize=8)
        fig.tight_layout(rect=(.035,.13,1,.96))
        save(fig,out,f'overview_{label}_all_controls.png')
    for control in controls:
        fig,axes=plt.subplots(1,len(metrics),figsize=(6*len(metrics),5.1),sharex=True,sharey=True,squeeze=False)
        axes=axes[0]
        for ax,(metric,label) in zip(axes,metrics.items()):
            draw(ax,w,control,metric);ax.set_title(label);ax.set_xlabel('Inhibitory edges (%)')
        axes[0].set_ylabel('Matched hyperparameter settings CE wins (%)')
        secondary=axes[-1].secondary_yaxis('right',functions=(lambda percent:percent*nsettings/100,lambda count:count*100/nsettings))
        secondary.set_yticks(np.linspace(0,nsettings,5));secondary.set_ylabel(f'CE wins out of {nsettings} settings')
        fig.suptitle(f'CE empirical versus {LABELS[control]}',fontsize=15)
        handles,names=axes[0].get_legend_handles_labels()
        fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
        fig.text(.5,.012,f'Compare seed-mean scores at identical settings, then count wins; {nseeds} paired realizations. Shading: pointwise 95% intervals.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.16,1,.93))
        save(fig,out,f'{control}_full_grid.png')
        fig,axes=plt.subplots(len(metrics),3,figsize=(13,4*len(metrics)),sharex=True,sharey=True,squeeze=False)
        for row,(metric,label) in enumerate(metrics.items()):
            for col,(mode,(mode_label,_,_)) in enumerate(MODES.items()):
                ax=axes[row,col]
                for inp,color in zip(sorted(by_input.input_scale.unique()),INPUT_COLORS):
                    g=by_input[(by_input.control==control)&(by_input.metric==metric)&(by_input.normalization==mode)&(by_input.input_scale==inp)].sort_values('negative_fraction')
                    if g.empty:continue
                    assert len(g)==w.negative_fraction.nunique()
                    ax.plot(100*g.negative_fraction,g.ce_win_percent,marker='o',ms=3,lw=1.6,color=color,label=f'Input {inp:g}')
                style(ax,w.negative_fraction.unique())
                if row==0:ax.set_title(mode_label)
                if col==0:ax.set_ylabel(f'{label}\nCE wins (%)')
                if row==len(metrics)-1:ax.set_xlabel('Inhibitory edges (%)')
        fig.suptitle(f'CE versus {LABELS[control]}: matched wins by input scale',fontsize=15)
        handles,names=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.047),ncol=4,frameon=False)
        fig.text(.5,.012,f'Each input curve counts wins over {ninput} matched radius/leak settings; {nseeds} paired realizations. Descriptive point estimates; no intervals.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.12,1,.94))
        save(fig,out,f'{control}_by_input.png')
    lines=['# Matched hyperparameter win curves by control','','Every figure compares empirical CE with the named control. Higher means CE wins more of the tested settings; NARMA uses lower error to decide a win. These are win-count curves, not grid-mean performance curves.','',
        f'The full-grid plots use {nsettings} configurations and the recorded pointwise paired-realization bootstrap intervals. Input-specific panels count {ninput} radius/leak settings per input; they show descriptive point estimates without intervals. The dotted line is half of the configurations.','',
        ' · '.join(f'[{label} overview](overview_{label}_all_controls.png)' for label in metrics.values()),'','| Control | Full grid, all metrics | By input, all metrics |','|---|---|---|']
    for control in controls:lines.append(f'| {LABELS[control]} | [PNG]({control}_full_grid.png) | [PNG]({control}_by_input.png) |')
    lines+=['','The six inhibitory fractions are observed points; line segments connect them and do not establish an exact crossover. The grid is fixed, and results do not extend automatically to untested settings. Every strict positive seed-mean advantage counts, including a small advantage at low absolute performance.','',
        'All existing source tables and figures were preserved. No simulations were run.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(source_analysis=str(base),sources={name:sha(base/name) for name in ['ce_winning_fraction.csv','configuration_advantages.csv']},
        plot_script_sha256=sha(Path(__file__)),models=controls,reference_model='ce_empirical',
        primary_estimand=f'fraction of matched settings whose {nseeds}-realization mean score favors CE at each inhibitory-edge fraction',
        n_curve_points_reconstructed=len(w),all_curve_points_independently_reconstructed=True,outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(out);print(f'{len(metrics)+2*len(controls)} PNGs: metric overviews, model comparisons and input-specific comparisons.')

if __name__=='__main__':main()
