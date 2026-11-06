"""
plot_blur_sweep_r.py
--------------------
Appendix blur-sweep figure in the style of the metric-suite overview
(rebuttal/plots/plot_fig_metric_suite.py): alignment as Spearman r against Gaussian blur
sigma, with both human noise ceilings drawn -- mean pairwise (the manuscript's ceiling)
and leave-one-out (each participant against the rank-mean of the others).
Data: final_figures/molmo_bilinear_pairwise/blur_sweep/blur_sweep_perimage.parquet
Writes fig_blur_sweep_r.png next to it.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures')
BS = os.path.join(FF, 'molmo_bilinear_pairwise', 'blur_sweep')
long = pd.read_parquet(os.path.join(BS, 'blur_sweep_perimage.parquet'))
pair = {t: np.tanh(z) for t, z in json.load(open(os.path.join(FF, 'molmo_bilinear_pairwise', 'results', 'ceilings.json'))).items()}
loo = {t: v['loo_r'] for t, v in json.load(open(os.path.join(FF, 'qwen35_nothink', 'blur_sweep', 'ceilings_pairwise_vs_loo.json'))).items()}

COLORS = FST.ALL_COLORS
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.',
        'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TASK_NAME = {'describe': 'Describe Task', 'social': 'Social Task'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 34, 24, 26, 40, 20
SIGMAS = sorted(long.sigma.unique())

fig, axes = plt.subplots(1, 2, figsize=(24, 9), sharey=True)
for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
    for m in ORDER:
        d = long[(long.task == task) & (long.model == m)]
        sz = d.groupby(['sigma', 'subject_id'])['z'].mean().unstack()      # sigma x subject
        mu, se = sz.mean(axis=1), sz.std(axis=1, ddof=1) / np.sqrt(sz.shape[1])
        ax.fill_between(mu.index, np.tanh(mu - se), np.tanh(mu + se), color=COLORS[m], alpha=0.2, lw=0, zorder=2)
        ax.plot(mu.index, np.tanh(mu), color=COLORS[m], lw=4, marker='o', ms=10,
                markeredgecolor='black', markeredgewidth=0.6, label=TICK.get(m, m), zorder=3)
    ax.axhline(pair[task], color='black', ls='--', lw=2.4, zorder=1, label='Human ceiling (pairwise)')
    ax.axhline(loo[task], color='dimgray', ls=':', lw=3.0, zorder=1, label='Human ceiling (leave-one-out)')
    ax.set_xscale('symlog', linthresh=2)
    ax.set_xlim(-0.3, 70)
    ax.set_xticks(SIGMAS)
    ax.set_xticklabels([str(int(s)) for s in SIGMAS], fontsize=FS_TICK * 0.85)
    ax.minorticks_off()
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_xlabel(r'Gaussian blur $\sigma$ (pixels)', fontsize=FS_LABEL, labelpad=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.set_title(TASK_NAME[task], fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.text(-0.06, 1.10, letter, transform=ax.transAxes,
            fontsize=FS_LETTER, fontweight='bold', va='top', ha='left')
axes[0].set_ylim(0.08, 0.38)
axes[0].set_ylabel(r'Spearman $r$', fontsize=FS_LABEL, labelpad=10)
fig.tight_layout()
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=FS_LEG, ncol=4,
           loc='upper center', bbox_to_anchor=(0.5, 0.0), frameon=False)
f = os.path.join(BS, 'fig_blur_sweep_r.png')
fig.savefig(f, dpi=300, bbox_inches='tight')
print('wrote', f, {t: (round(pair[t], 3), round(loo[t], 3)) for t in pair})
