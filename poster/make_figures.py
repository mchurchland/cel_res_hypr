"""Render a poster-sized subset of the supplied raw-radius summary (no new fits)."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurreps-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / 'assets/raw_rho_summary.csv')
# Same below-50% sweep subset as the abstract's figure. Retain every shuffle.
df = df[(df.series_kind == 'shuffle') | (df.sign_frac < .5)]
styles = [
    ('Binary controls', 'Binary', '#765496', 's'),
    ('PM1 controls', 'Signed-unit controls', '#b77800', 'D'),
    ('C. elegans shuffle controls', 'C. elegans shuffles', '#00876b', '^'),
    ('Matched C. elegans sweep', 'C. elegans E/I sweep', '#df775f', 'o'),
    ('Removed C. elegans sweep', 'Known-edge E/I sweep', '#507eae', 'o'),
    ('Matched ER sweep', 'Random E/I sweep', '#9a7bbc', 'o'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42})
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
for row, metric in enumerate(['MC', 'GR']):
    for col, value in enumerate(['performance', 'cv']):
        ax = axes[row, col]
        for key, label, color, marker in styles:
            d = df[(df.metric == metric) & (df.series == key)].sort_values('raw_rho')
            assert len(d), key
            # Connect sweep points only; shuffles are discrete controls.
            ax.plot(d.raw_rho, d[value], color=color, marker=marker,
                    ls='-' if 'sweep' in key.lower() else 'None',
                    lw=2, ms=7, mfc='white', mew=1.8)
        ax.set_xscale('log')
        ax.set_xticks([3,10,30,100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=.2)
        if row == 0: ax.set_title(['Mean metric', 'Mean CV'][col], weight='bold', pad=12)
        if col == 0: ax.set_ylabel({'MC':'Memory capacity (MC)', 'GR':'Generalization rank (GR)'}[metric], fontsize=15)
        if row == 1: ax.set_xlabel(r'Raw spectral radius $\rho_{\rm raw}$', fontsize=15)
handles = [Line2D([],[], color=c, marker=m, mfc='white', lw=2, label=l) for _,l,c,m in styles]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=13.5,
           handlelength=1.5, columnspacing=1.2)
fig.subplots_adjust(left=.13,right=.98,top=.92,bottom=.25,wspace=.29,hspace=.36)
fig.savefig(ROOT/'assets/radius_poster.pdf')
fig.savefig(ROOT/'assets/radius_poster.png', dpi=180)
