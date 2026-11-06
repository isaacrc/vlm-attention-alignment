"""
plot_fig1_baselines.py -- exploratory, not in the manuscript.
Figure 1A/B (Describe, Social; % of the pairwise human upperbound, Fisher-z ratio) with the
eight baselines of score_baselines.py added to the right of the six models. Model bars are
the Figure 1 values (alignment_long_cv_honest.parquet, cross-validated heads, no blur).
Style as plot_fig1_best_blur.py.
Writes final_figures/molmo_bilinear/fig1_baselines/fig1_alignment_baselines.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear', 'fig1_baselines')
CEIL = json.load(open(os.path.join(FF, 'results', 'ceilings.json')))
mod = (pd.read_parquet(os.path.join(FF, 'results', 'alignment_long_cv_honest.parquet'))
         .groupby(['model', 'task', 'subject_id']).z.mean().rename('mean_z').reset_index())
base = pd.read_csv(os.path.join(OUT, 'baseline_subject_z.csv'))
d = pd.concat([mod, base[['model', 'task', 'subject_id', 'mean_z']]])

MODELS = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
BASE = [  # name, colour, hatch
    ('Itti-Koch', '#7b5ea7', 'xx'), ('GBVS', '#a98bd0', 'xx'), ('Spectral residual', '#d6c6ec', 'xx'),
    ('Edge energy', '#9a9a9a', '///'), ('Luminance', '#cfcfcf', '///'),
    ('Center prior', '#8c6d31', '..'), ('Person prior', '#bd9e39', '..'), ('Face prior', '#e7cb94', '..')]
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.', 'Molmo': 'Molmo 7B-D',
        'Qwen3.5': 'Qwen 9B-3.5', 'Spectral residual': 'Spec. residual'}
FS_TITLE, FS_TICK, FS_LABEL, FS_LEGEND = 56, 34, 42, 36
TITLE_PAD = 14
BAR_W = 0.62
GAP = 0.8
YMAX = 125

names = MODELS + [b[0] for b in BASE]
x = np.r_[np.arange(len(MODELS)), np.arange(len(BASE)) + len(MODELS) + GAP]
colors = [FST.ALL_COLORS[m] for m in MODELS] + [b[1] for b in BASE]
hatches = [None] * len(MODELS) + [b[2] for b in BASE]

fig, axes = plt.subplots(1, 2, figsize=(40, 11.5), sharey=True)
for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
    z = [d[(d.task == task) & (d.model == m)].mean_z for m in names]
    means = np.array([v.mean() for v in z]) / CEIL[task] * 100
    sems = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in z]) / CEIL[task] * 100
    for xi, mval, c, h in zip(x, means, colors, hatches):
        ax.bar(xi, mval, BAR_W, color=c, hatch=h, edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=5, linewidth=2.0, zorder=4)
    ax.axhline(100.0, color='black', linestyle='--', linewidth=2.5, zorder=2)
    ax.axvline(len(MODELS) - 1 + (1 + GAP) / 2, color='gray', linewidth=1.2, linestyle=':', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK.get(n, n) for n in names], fontsize=FS_TICK, rotation=50, ha='right')
    ax.margins(x=0.02)
    ax.set_yticks([0, 50, 100]); ax.tick_params(axis='y', labelsize=FS_TICK + 6)
    ax.set_ylim(-20, YMAX); ax.axhline(0, color='black', linewidth=1.0, zorder=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f'{task.capitalize()} Task', fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.08 if letter == 'A' else -0.01, 1.0, letter, transform=ax.transAxes + off,
            fontsize=FS_TITLE, fontweight='bold', va='baseline', ha='left')
    print(task, ' '.join(f'{n}={v:.1f}' for n, v in zip(names, means)))
axes[0].set_ylabel('% of Human\nUpperbound', fontsize=FS_LABEL, labelpad=12)
axes[1].tick_params(axis='y', labelleft=False)
axes[1].text(0.99, 103, 'Human Upperbound', transform=axes[1].get_yaxis_transform(),
             ha='right', va='bottom', fontsize=FS_LEGEND)
fig.tight_layout(w_pad=2)
f = os.path.join(OUT, 'fig1_alignment_baselines.png')
fig.savefig(f, dpi=200, bbox_inches='tight')
print('wrote', f)
