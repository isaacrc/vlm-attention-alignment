"""
plot_fig_center_corrected.py
----------------------------
Two figures for the head-selection / center-bias thread of the rebuttal, both
written to rebuttal/plots/head_selection/.

  fig_center_corrected.png
      Raw vs center-bias-controlled alignment for all six agents plus the center
      prior, each normalized by the ceiling computed under its own metric. This
      is the figure behind the rebuttal sentence "under center-bias control the
      picture reverses and Molmo becomes the best-aligned agent": the raw bars
      reproduce the paper's ordering, the partial bars reorder it.

  fig_layer_mean_vs_best_head_controlling_for_center_bias.png
      The AC's remedy for biased head selection ("averaging results over all
      layers/heads") applied to the two trained Transformer decoders, against
      their cross-validated best head and against the matched LSTM. Shows what
      the decoder effect looks like when the Transformer gets no head selection
      at all. Decomposed layer by layer in plot_fig_per_layer.py.

All numbers come from task3_baselines/run_headrule_center.py, which scores every
model under both metrics in one pass so raw and partial are strictly
like-for-like. Style follows the manuscript Fig 3 panels.
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
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5',
        'CENTER_PRIOR': 'Center prior'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 30, 20, 22, 34, 18
TASK_NAME = {'describe': 'Describe Task', 'social': 'Social Task'}

# One row per model for the main figure: the rule each model is *reported* under.
MAIN = [('CNN-LSTM', 'na'), ('CNN-Transformer', 'cv_best'),
        ('ViT-LSTM', 'na'), ('ViT-Transformer', 'cv_best'),
        ('Molmo', 'cv_best'), ('Qwen3.5', 'cv_best'), ('CENTER_PRIOR', 'na')]


def row(model, rule, task):
    m = summ[(summ.model == model) & (summ.rule == rule) & (summ.task == task)]
    if m.empty:
        raise KeyError(f'{model}/{rule}/{task} missing from headrule_center_summary.csv')
    return m.iloc[0]


def pct_and_err(r, metric, ceil_z):
    """% of ceiling and its SEM, both in Fisher-z space (the space the mean is
    taken in), so the error bar matches how the point estimate was computed."""
    pct = r[f'pct_{metric}']
    rr, sem_r = r[f'r_{metric}'], r[f'sem_{metric}']
    z_hi = np.arctanh(np.clip(rr + sem_r, -0.999999, 0.999999))
    z = np.arctanh(np.clip(rr, -0.999999, 0.999999))
    return pct, 100.0 * (z_hi - z) / ceil_z


def style(ax, xt, labels, ylab=None, letter=None, title=None, ymax=115):
    ax.set_xticks(xt)
    ax.set_xticklabels(labels, fontsize=FS_TICK, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8, zorder=2)
    ax.set_ylim(-12, ymax)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=12)
    if ylab:
        ax.set_ylabel(ylab, fontsize=FS_LABEL, labelpad=10)
    if letter:
        ax.text(-0.12, 1.10, letter, transform=ax.transAxes,
                fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — raw vs center-controlled
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True)
W = 0.38

for ax, task in zip(axes, ['describe', 'social']):
    cz_raw, cz_par = ceil[task]['raw_z'], ceil[task]['partial_z']
    x = np.arange(len(MAIN))
    for off, metric, cz, alpha, hatch in [(-W / 2, 'raw', cz_raw, 0.38, '//'),
                                          (+W / 2, 'partial', cz_par, 1.0, None)]:
        vals, errs, cols = [], [], []
        for m, rule in MAIN:
            r = row(m, rule, task)
            p, e = pct_and_err(r, metric, cz)
            vals.append(p); errs.append(e); cols.append(COLORS[m])
        ax.bar(x + off, vals, W, color=cols, alpha=alpha, hatch=hatch,
               edgecolor='black', linewidth=0.6, zorder=3)
        ax.errorbar(x + off, vals, yerr=errs, fmt='none', color='black',
                    capsize=4, linewidth=1.6, zorder=4)
    ax.axhline(100.0, color='black', linestyle='--', linewidth=2.2, zorder=2)
    style(ax, x, [TICK.get(m, m) for m, _ in MAIN],
          ylab='% of human upperbound' if task == 'describe' else None,
          letter='A' if task == 'describe' else 'B', title=TASK_NAME[task])
    ax.margins(x=0.04)

fig.tight_layout(rect=[0, 0.06, 1, 0.92])
fig.legend(handles=[
    Patch(facecolor='#555555', alpha=0.38, hatch='//', label='raw Spearman'),
    Patch(facecolor='#555555', label='center bias removed (partial)'),
    plt.Line2D([0], [0], color='black', ls='--', lw=2.2, label='human upperbound'),
], fontsize=FS_LEG, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01),
    frameon=False)

fig.suptitle('Alignment before and after removing the center-bias prior\n'
'each metric normalized by the human ceiling computed under that same metric; '
'Transformers and VLMs on cross-validated heads',
fontsize=FS_TITLE * 0.62, y=0.99)
f1 = os.path.join(OUT, 'fig_center_corrected.png')
fig.savefig(f1, dpi=300, bbox_inches='tight')
print('Saved:', f1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — layer-mean vs cross-validated best head, trained Transformers
# ══════════════════════════════════════════════════════════════════════════════
PAIRS = [('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer')]
BARS = [('LSTM', 'na', 'LSTM\n(single map)'),
        ('cv_best', 'cv_best', 'Transformer\nCV best head'),
        ('layer_mean', 'layer_mean', 'Transformer\nall layers x heads')]

fig, axes = plt.subplots(2, 2, figsize=(17, 15))
W2 = 0.38

for ri, (lstm, trans) in enumerate(PAIRS):
    for ci, task in enumerate(['describe', 'social']):
        ax = axes[ri, ci]
        cz_raw, cz_par = ceil[task]['raw_z'], ceil[task]['partial_z']
        x = np.arange(len(BARS))
        for off, metric, cz, alpha, hatch in [(-W2 / 2, 'raw', cz_raw, 0.38, '//'),
                                              (+W2 / 2, 'partial', cz_par, 1.0, None)]:
            vals, errs, cols = [], [], []
            for kind, rule, _ in BARS:
                m = lstm if kind == 'LSTM' else trans
                r = row(m, rule, task)
                p, e = pct_and_err(r, metric, cz)
                vals.append(p); errs.append(e); cols.append(COLORS[m])
            ax.bar(x + off, vals, W2, color=cols, alpha=alpha, hatch=hatch,
                   edgecolor='black', linewidth=0.6, zorder=3)
            ax.errorbar(x + off, vals, yerr=errs, fmt='none', color='black',
                        capsize=4, linewidth=1.6, zorder=4)
        # center prior as the floor this has to clear
        pf = row('CENTER_PRIOR', 'na', task)['pct_partial']
        ax.axhline(pf, color='#9e9e9e', ls='-.', lw=2.0, zorder=2)
        ax.axhline(100.0, color='black', ls='--', lw=2.2, zorder=2)
        style(ax, x, [b[2] for b in BARS],
              ylab='% of human upperbound' if ci == 0 else None,
              letter='ABCD'[ri * 2 + ci],
              title=f'{lstm.split("-")[0]} encoder — {TASK_NAME[task]}')
        ax.tick_params(axis='x', labelsize=FS_TICK * 0.8)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0); lbl.set_ha('center')
        ax.margins(x=0.12)

        # annotate the LSTM-vs-layer_mean partial contrast
        c = con[(con.metric == 'r_partial') & (con.task == task) &
                (con.a == f'{lstm}|na') & (con.b == f'{trans}|layer_mean')]
        if not c.empty:
            cc = c.iloc[0]
            p = cc['p_perm']
            ptxt = 'p < .001' if p < 0.001 else f'p = {p:.3f}'
            ax.text(0.98, 0.97, f"LSTM vs all-layer mean (partial)\n$d_z$ = {cc['dz']:.2f}, {ptxt}",
                    transform=ax.transAxes, fontsize=FS_LEG * 0.85, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#cccccc'))

fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.legend(handles=[
    Patch(facecolor='#555555', alpha=0.38, hatch='//', label='raw Spearman'),
    Patch(facecolor='#555555', label='center bias removed'),
    plt.Line2D([0], [0], color='#9e9e9e', ls='-.', lw=2.0, label='center prior (partial)'),
    plt.Line2D([0], [0], color='black', ls='--', lw=2.2, label='human upperbound'),
], fontsize=FS_LEG, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.005), frameon=False)

fig.suptitle('Removing head selection entirely: Transformer decoders averaged over all layers and heads\n'
'the AC\'s remedy for biased head selection, against the cross-validated head and the matched LSTM',
fontsize=FS_TITLE * 0.62, y=0.995)
f2 = os.path.join(OUT, 'fig_layer_mean_vs_best_head_controlling_for_center_bias.png')
fig.savefig(f2, dpi=300, bbox_inches='tight')
print('Saved:', f2)


# ── console summary ───────────────────────────────────────────────────────────
print('\n=== % of ceiling, raw vs partial ===')
show = summ.copy()
show['pct_raw'] = show['pct_raw'].round(1)
show['pct_partial'] = show['pct_partial'].round(1)
print(show[['model', 'rule', 'task', 'head', 'r_raw', 'pct_raw',
            'r_partial', 'pct_partial']].to_string(index=False))
print('\n=== partial-metric contrasts ===')
print(con[con.metric == 'r_partial'][['task', 'a', 'b', 'mean_diff', 'dz', 'p_perm']]
      .to_string(index=False))
