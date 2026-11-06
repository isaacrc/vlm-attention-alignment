"""
plot_fig_smoothness.py
----------------------
Smoothness appendix figure, merging plot_blur_sweep_r.py (A, B) and
plot_fig1_best_blur.py --r (C) on one canvas in the metric-suite style:
  A, B  alignment (Spearman r) against Gaussian blur sigma, Describe and Social, on the
        eleven-sigma grid of the sweep (sigma 40 and 56, added later for the per-model bar
        figures, are left out so A-B match the cross-validation in C)
  C     alignment at each model's cross-validated sigma, Describe | Social, sigma printed
        at the base of each bar
Model legend inside panel B, between the two ceilings.
Both human ceilings in every panel, grey and dashed alike, labelled in place: mean
pairwise (the manuscript's ceiling) and leave-one-out.
Writes final_figures/molmo_bilinear_pairwise/blur_sweep/fig_smoothness.png
"""
import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures')
BS = os.path.join(FF, 'molmo_bilinear_pairwise', 'blur_sweep')
SIGMAS = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
long = pd.read_parquet(os.path.join(BS, 'blur_sweep_perimage.parquet'))
long = long[long.sigma.isin(SIGMAS)]
subj_cv = pd.read_csv(os.path.join(BS, 'blur_sweep_cv_subject_z.csv'))
summ = pd.read_csv(os.path.join(BS, 'blur_sweep_summary.csv')).set_index(['task', 'model'])
PAIR = {t: np.tanh(z) for t, z in json.load(open(os.path.join(FF, 'molmo_bilinear_pairwise', 'results', 'ceilings.json'))).items()}
LOO = {t: v['loo_r'] for t, v in json.load(open(os.path.join(FF, 'qwen35_nothink', 'blur_sweep', 'ceilings_pairwise_vs_loo.json'))).items()}

COLORS = FST.ALL_COLORS
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TASKS = ['describe', 'social']
TASK_NAME = {'describe': 'Describe Task', 'social': 'Social Task'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 34, 24, 26, 40, 20
GREY = 'dimgray'


def ceilings(ax, task, ymax, ymin=0.0):
    for y, lab in ((PAIR[task], 'Human upperbound (pairwise)'), (LOO[task], 'Human upperbound (leave-one-out)')):
        ax.axhline(y, color=GREY, linestyle='--', linewidth=2.5, zorder=1)
        ax.text(0.01, (y - ymin) / (ymax - ymin) + 0.012, lab, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=FS_LEG, color=GREY)


def style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=FS_TICK)


# --compact: no titles on C and less space between the rows; writes fig_smoothness_compact.png
COMPACT = '--compact' in sys.argv
fig = plt.figure(figsize=(24, 20 if COMPACT else 21))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05], hspace=0.22 if COMPACT else 0.38, wspace=0.08)

# A, B: the sweep
YMIN, YMAX = 0.08, 0.40
top = []
for k, task in enumerate(TASKS):
    ax = fig.add_subplot(gs[0, k], sharey=top[0] if top else None)
    for m in ORDER:
        d = long[(long.task == task) & (long.model == m)]
        sz = d.groupby(['sigma', 'subject_id'])['z'].mean().unstack()
        mu, se = sz.mean(axis=1), sz.std(axis=1, ddof=1) / np.sqrt(sz.shape[1])
        ax.fill_between(mu.index, np.tanh(mu - se), np.tanh(mu + se), color=COLORS[m], alpha=0.2, lw=0, zorder=2)
        ax.plot(mu.index, np.tanh(mu), color=COLORS[m], lw=4, marker='o', ms=10,
                markeredgecolor='black', markeredgewidth=0.6, label=TICK.get(m, m), zorder=3)
    ax.set_ylim(YMIN, YMAX)
    ceilings(ax, task, YMAX, YMIN)
    ax.set_xscale('symlog', linthresh=2); ax.set_xlim(-0.3, 70)
    ax.set_xticks(SIGMAS); ax.set_xticklabels([str(s) for s in SIGMAS], fontsize=FS_TICK * 0.85)
    ax.minorticks_off()
    ax.set_xlabel(r'Gaussian blur $\sigma$ (pixels)', fontsize=FS_LABEL, labelpad=10)
    style(ax)
    ax.set_title(TASK_NAME[task], fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.text(-0.13 if k == 0 else -0.04, 1.10, 'AB'[k], transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
    if k:
        ax.tick_params(axis='y', labelleft=False)
    top.append(ax)
top[0].set_ylabel(r'Spearman $r$', fontsize=FS_LABEL, labelpad=10)
# legend in panel B, right side, in the free band between the two ceilings
top[1].legend(*top[0].get_legend_handles_labels(), fontsize=FS_LEG, ncol=2, frameon=False,
              loc='upper right', bbox_to_anchor=(1.0, (LOO['social'] - 0.008 - YMIN) / (YMAX - YMIN)),
              borderaxespad=0.2, handlelength=1.6, columnspacing=1.0)

# C: best blur per model
YMAX_C = 0.42
bot = []
for k, task in enumerate(TASKS):
    ax = fig.add_subplot(gs[1, k], sharey=bot[0] if bot else None)
    z = [subj_cv[(subj_cv.task == task) & (subj_cv.model == m)].mean_z for m in ORDER]
    mz = np.array([v.mean() for v in z]); sz = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in z])
    means = np.tanh(mz); sems = sz * (1 - means ** 2)       # SEM carried through tanh (delta rule)
    x = np.arange(len(ORDER))
    ax.bar(x, means, 0.55, color=[COLORS[m] for m in ORDER], edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=6, linewidth=2.0, zorder=4)
    for xi, m in zip(x, ORDER):
        s = Counter(summ.loc[(task, m), 'cv_sigmas'].split('/')).most_common(1)[0][0]
        ax.text(xi, 0.012, rf'$\sigma$' + f'\n{s}', ha='center', va='bottom', fontsize=FS_LEG,
                linespacing=1.1, color='white', fontweight='bold', zorder=5)
    ax.set_ylim(0, YMAX_C)
    ceilings(ax, task, YMAX_C)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticks(x); ax.set_xticklabels([TICK.get(m, m) for m in ORDER], fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    style(ax)
    if not COMPACT:
        ax.set_title(TASK_NAME[task], fontsize=FS_TITLE, fontweight='bold', pad=12)
    if k == 0:
        ax.text(-0.13, 1.04 if COMPACT else 1.10, 'C', transform=ax.transAxes, fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
    else:
        ax.tick_params(axis='y', labelleft=False)
    bot.append(ax)
bot[0].set_ylabel(r'Spearman $r$', fontsize=FS_LABEL, labelpad=10)

f = os.path.join(BS, 'fig_smoothness_compact.png' if COMPACT else 'fig_smoothness.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
print('wrote', f)
