"""
plot_fig_per_layer.py
---------------------
Decomposition of fig_layer_mean_vs_best_head.png: instead of one bar for the mean over
all 4 layers x 8 heads, one bar per decoder layer (mean over that layer's 8 heads),
with the cross-validated best head and the all-layer mean kept as reference.

The question it answers: is the all-layer mean's behaviour a property of the decoder as
a whole, or is it carried by one layer? Per-layer means are still a predefined rule --
a layer is named in advance rather than picked by score -- so they remain a legitimate
answer to the AC's head-selection objection.

Both metrics are shown, because the all-layer mean's raw and center-controlled scores
move in opposite directions and the per-layer view is where that becomes legible.

Source: task3_baselines/run_headrule_center.py (rules layer1..layer4).
Output: plots/head_selection/fig_per_layer.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import figstyle as FST

from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REBUTTAL = os.path.dirname(HERE)
RES = os.path.join(REBUTTAL, 'task3_baselines', 'results')
OUT = os.path.join(HERE, 'head_selection')
os.makedirs(OUT, exist_ok=True)

summ = pd.read_csv(os.path.join(RES, 'headrule_center_summary.csv'))
ceil = json.load(open(os.path.join(RES, 'headrule_center_ceilings.json')))
con = pd.read_csv(os.path.join(RES, 'headrule_center_contrasts.csv'))

COLORS = FST.ALL_COLORS
TASK_NAME = {'describe': 'Describe Task', 'social': 'Social Task'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 28, 19, 22, 34, 18

PAIRS = [('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer')]
# (which model, rule, x-label)
BARS = [('lstm', 'na', 'LSTM'),
        ('trans', 'cv_best', 'CV best\nhead'),
        ('trans', 'layer1', 'L1'), ('trans', 'layer2', 'L2'),
        ('trans', 'layer3', 'L3'), ('trans', 'layer4', 'L4'),
        ('trans', 'layer_mean', 'all\nlayers')]


def row(model, rule, task):
    m = summ[(summ.model == model) & (summ.rule == rule) & (summ.task == task)]
    if m.empty:
        raise KeyError(f'{model}/{rule}/{task} missing — re-run run_headrule_center.py')
    return m.iloc[0]


def pct_and_err(r, metric, ceil_z):
    pct = r[f'pct_{metric}']
    rr, sem_r = r[f'r_{metric}'], r[f'sem_{metric}']
    z = np.arctanh(np.clip(rr, -0.999999, 0.999999))
    z_hi = np.arctanh(np.clip(rr + sem_r, -0.999999, 0.999999))
    return pct, 100.0 * (z_hi - z) / ceil_z


fig, axes = plt.subplots(2, 2, figsize=(20, 14))
W = 0.38

for ri, (lstm, trans) in enumerate(PAIRS):
    for ci, task in enumerate(['describe', 'social']):
        ax = axes[ri, ci]
        cz = {'raw': ceil[task]['raw_z'], 'partial': ceil[task]['partial_z']}
        x = np.arange(len(BARS))
        for off, metric, alpha, hatch in [(-W / 2, 'raw', 0.38, '//'),
                                          (+W / 2, 'partial', 1.0, None)]:
            vals, errs, cols = [], [], []
            for kind, rule, _ in BARS:
                m = lstm if kind == 'lstm' else trans
                p, e = pct_and_err(row(m, rule, task), metric, cz[metric])
                vals.append(p); errs.append(e); cols.append(COLORS[m])
            ax.bar(x + off, vals, W, color=cols, alpha=alpha, hatch=hatch,
                   edgecolor='black', linewidth=0.6, zorder=3)
            ax.errorbar(x + off, vals, yerr=errs, fmt='none', color='black',
                        capsize=3.5, linewidth=1.5, zorder=4)

        # the LSTM's own center-controlled level, as the line every layer must clear
        lstm_par = row(lstm, 'na', task)['pct_partial']
        ax.axhline(lstm_par, color=COLORS[lstm], ls=':', lw=2.6, zorder=2)
        ax.axhline(row('CENTER_PRIOR', 'na', task)['pct_partial'],
                   color='#9e9e9e', ls='-.', lw=1.8, zorder=2)
        ax.axhline(100.0, color='black', ls='--', lw=2.2, zorder=2)
        ax.axhline(0, color='black', lw=0.8, zorder=2)

        # significance of each per-layer mean against the matched LSTM (partial metric)
        for k, (kind, rule, _) in enumerate(BARS):
            if kind != 'trans' or not rule.startswith('layer') or rule == 'layer_mean':
                continue
            c = con[(con.metric == 'r_partial') & (con.task == task) &
                    (con.a == f'{lstm}|na') & (con.b == f'{trans}|{rule}')]
            if c.empty:
                continue
            p, dz = c.iloc[0]['p_perm'], c.iloc[0]['dz']
            txt = 'n.s.' if p >= .05 else '*' if p >= .01 else '**' if p >= .001 else '***'
            # dz is LSTM minus Transformer, so dz < 0 means the Transformer wins;
            # colour the star by who it favours rather than leaving it directionless
            if p < .05 and dz < 0:
                txt += '▲'
                col = COLORS[trans]
            else:
                col = 'black'
            v, _ = pct_and_err(row(trans, rule, task), 'partial', cz['partial'])
            ax.text(k + W / 2, max(v, 0) + 4, txt, ha='center', va='bottom',
                    fontsize=FS_LEG * 0.85, fontweight='bold', color=col)

        ax.set_xticks(x)
        ax.set_xticklabels([b[2] for b in BARS], fontsize=FS_TICK)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.set_yticks([0, 50, 100])
        ax.set_ylim(-10, 118)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f'{lstm.split("-")[0]} encoder — {TASK_NAME[task]}',
                     fontsize=FS_TITLE, fontweight='bold', pad=12)
        ax.text(-0.09, 1.12, 'ABCD'[ri * 2 + ci], transform=ax.transAxes,
                fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
        if ci == 0:
            ax.set_ylabel('% of human upperbound', fontsize=FS_LABEL, labelpad=10)
        ax.margins(x=0.04)

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.legend(handles=[
    Patch(facecolor='#555555', alpha=0.38, hatch='//', label='raw Spearman'),
    Patch(facecolor='#555555', label='center bias removed'),
    plt.Line2D([0], [0], color='#888888', ls=':', lw=2.6, label='matched LSTM (partial)'),
    plt.Line2D([0], [0], color='#9e9e9e', ls='-.', lw=1.8, label='center prior (partial)'),
    plt.Line2D([0], [0], color='black', ls='--', lw=2.2, label='human upperbound'),
], fontsize=FS_LEG, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.005), frameon=False)
fig.suptitle('Transformer cross-attention averaged within each decoder layer\n'
'stars: per-layer mean vs the matched LSTM, center-controlled metric  '
'(coloured ▲ = the Transformer wins)',
fontsize=FS_TITLE * 0.72, y=0.99)

f = os.path.join(OUT, 'fig_per_layer.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
print('Saved:', f)

sel = summ[summ.rule.str.startswith('layer') | summ.rule.isin(['cv_best', 'na'])]
print('\n=== per-layer means ===')
print(sel[['model', 'rule', 'task', 'r_raw', 'pct_raw', 'r_partial', 'pct_partial']]
      .round(3).to_string(index=False))
