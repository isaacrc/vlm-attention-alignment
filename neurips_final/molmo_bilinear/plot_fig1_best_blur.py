"""
plot_fig1_best_blur.py
----------------------
Figure 1A/B redrawn at each model's best Gaussian blur: alignment as % of the mean
pairwise human noise ceiling (Fisher-z ratio), with sigma chosen per model and task by
five-fold cross-validation over images (qwen35_nothink/blur_sweep.py, the selection the
Smoothness appendix reports; in-sample maxima are the same sigma in 11 of 12 cells).
Style as rebuttal/plots/plot_fig_head_selection_corrected.py. The chosen sigma is printed
inside the base of each bar.
Data: final_figures/molmo_bilinear_pairwise/blur_sweep/blur_sweep_cv_subject_z.csv,
blur_sweep_summary.csv. Writes fig1_alignment_best_blur.png next to them.
"""
import os, sys, json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
BS = os.path.join(FF, 'blur_sweep')
CEIL = json.load(open(os.path.join(FF, 'results', 'ceilings.json')))
subj = pd.read_csv(os.path.join(BS, 'blur_sweep_cv_subject_z.csv'))
summ = pd.read_csv(os.path.join(BS, 'blur_sweep_summary.csv')).set_index(['task', 'model'])

COLORS = FST.ALL_COLORS
MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LEGEND = 56, 40, 42, 36
FS_LETTER = FS_TITLE
TITLE_PAD = 14
BAR_W = 0.55
YMAX = 125
# --r: Spearman r on the y-axis with both human ceilings (mean pairwise, leave-one-out),
# as in plot_blur_sweep_r.py; writes fig1_alignment_best_blur_r.png
RAW = '--r' in sys.argv
YMAX_R = 0.42
LOO_R = {t: v['loo_r'] for t, v in json.load(open(os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink',
                                                          'blur_sweep', 'ceilings_pairwise_vs_loo.json'))).items()}

fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
    z = [subj[(subj.task == task) & (subj.model == m)].mean_z for m in MODELS_6]
    mz = np.array([v.mean() for v in z]); sz = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in z])
    if RAW:   # back-transformed mean r; SEM carried through tanh by the delta rule
        means = np.tanh(mz); sems = sz * (1 - means ** 2)
    else:
        means, sems = mz / CEIL[task] * 100, sz / CEIL[task] * 100
    x = np.arange(len(MODELS_6))
    ax.bar(x, means, BAR_W, color=[COLORS[m] for m in MODELS_6], edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=6, linewidth=2.0, zorder=4)
    for xi, m, y, e in zip(x, MODELS_6, means, sems):
        # the sigma picked on most folds (all 5 in 11 of 12 cells)
        s = Counter(summ.loc[(task, m), 'cv_sigmas'].split('/')).most_common(1)[0][0]
        ax.text(xi, 0.012 if RAW else 4, rf'$\sigma$'+f'\n{s}', ha='center', va='bottom', fontsize=FS_LEGEND * 0.75, linespacing=1.1,
                color='white', fontweight='bold', zorder=5)
    if RAW:
        for y, lab in ((np.tanh(CEIL[task]), 'Human ceiling (pairwise)'),
                       (LOO_R[task], 'Human ceiling (leave-one-out)')):
            ax.axhline(y, color='dimgray', linestyle='--', linewidth=2.5, zorder=2)
            ax.text(0.01, y / YMAX_R + 0.012, lab, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=FS_LEGEND * 0.75, color='dimgray')
    else:
        ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK.get(m, m) for m in MODELS_6], fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4] if RAW else [0, 50, 100])
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, YMAX_R if RAW else YMAX)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f'{task.capitalize()} Task', fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.13 if letter == 'A' else -0.018, 1.0, letter, transform=ax.transAxes + off,
            fontsize=FS_LETTER, fontweight='bold', va='baseline', ha='left')
axes[0].set_ylabel(r'Spearman $r$' if RAW else '% of Human\nUpperbound', fontsize=FS_LABEL, labelpad=12)
axes[1].tick_params(axis='y', labelleft=False)
if not RAW:
    axes[1].text(0.99, 100.0 / YMAX + 0.015, 'Human Upperbound', transform=axes[1].transAxes,
                 ha='right', va='bottom', fontsize=FS_LEGEND)
fig.tight_layout(w_pad=2)
f = os.path.join(BS, 'fig1_alignment_best_blur' + ('_r' if RAW else '') + '.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
print('wrote', f)
