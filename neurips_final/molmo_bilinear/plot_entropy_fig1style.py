"""
plot_entropy_fig1style.py
-------------------------
Normalized attention entropy (Results, Spread of Spatial Attention) drawn like Figure 1A/B
(rebuttal/plots/plot_fig_head_selection_corrected.py): Describe (A) and Social (B), the six
models on their cross-validated heads plus the human fixation maps (dark bar, as in Fig 1C).
1 = attention spread evenly over all 196 locations. Error bars: SEM across images for the
models, across participants for the humans (as entropy_cvheads.csv reports them).
Data: final_figures/molmo_bilinear_pairwise/results/entropy_cvheads.csv
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST
FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
e = pd.read_csv(os.path.join(FF, 'results', 'entropy_cvheads.csv')).set_index(['model', 'task'])

ORDER = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5', 'Human']
COLORS = dict(FST.ALL_COLORS, Human='#4a4a4a')
TICK = {'CNN-Transformer': 'CNN-Trans.', 'ViT-Transformer': 'ViT-Trans.', 'Molmo': 'Molmo 7B-D', 'Qwen3.5': 'Qwen 9B-3.5'}
FS_TITLE, FS_TICK, FS_LABEL = 56, 40, 42
TITLE_PAD, BAR_W = 14, 0.55

fig, axes = plt.subplots(1, 2, figsize=(26, 10), sharey=True)
for ax, task, letter in zip(axes, ['describe', 'social'], 'AB'):
    v = np.array([e.loc[(m, task), 'mean_entropy_norm'] for m in ORDER])
    s = np.array([e.loc[(m, task), 'sem'] for m in ORDER])
    x = np.arange(len(ORDER))
    ax.bar(x, v, BAR_W, color=[COLORS[m] for m in ORDER], edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, v, yerr=s, fmt='none', color='black', capsize=6, linewidth=2.0, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([TICK.get(m, m) for m in ORDER], fontsize=FS_TICK, rotation=45, ha='right')
    ax.margins(x=0.06)
    # axis starts at .5 so the differences between models show
    ax.set_ylim(0.5, 1.05); ax.set_yticks([0.5, 0.75, 1.0]); ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f'{task.capitalize()} Task', fontsize=FS_TITLE, fontweight='bold', pad=TITLE_PAD)
    off = ScaledTranslation(0, TITLE_PAD / 72, fig.dpi_scale_trans)
    ax.text(-0.13 if letter == 'A' else -0.018, 1.0, letter, transform=ax.transAxes + off,
            fontsize=FS_TITLE, fontweight='bold', va='baseline', ha='left')
axes[0].set_ylabel('Normalized\nentropy', fontsize=FS_LABEL, labelpad=12)
axes[1].tick_params(axis='y', labelleft=False)
fig.tight_layout(w_pad=2)
out = os.path.join(FF, 'fig_entropy_fig1style.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('wrote', out)
