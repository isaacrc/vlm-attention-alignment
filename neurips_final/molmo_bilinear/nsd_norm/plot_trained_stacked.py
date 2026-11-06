"""
plot_trained_stacked.py
-----------------------
Social (top) and Describe (bottom) NSD results from the normalized fits, trained models and
low-level controls; with --with-vlms, Molmo and Qwen are added on the far right. Raw test R^2 (attention-only), mean over the 8 subjects,
SEM error bars, y-axis to .07 -- the styling of the manuscript's regional figures.
Data: task8_nsd/results_norm/attnonly_per_subject_cvbehav.csv (column attn_unique = raw R^2).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import figstyle as FST
d = pd.read_csv(ROOT / 'rebuttal' / 'task8_nsd' / 'results_norm' / 'attnonly_per_subject_cvbehav.csv')
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_norm' / (
    'fig_nsd_trained_vlms_social_describe.png' if '--with-vlms' in sys.argv else 'fig_nsd_trained_social_describe.png')

PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),
        ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None),
        ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///'),
        ('Luminance', 'Luminance', '#d0d0d0', '///')]
VLMS = '--with-vlms' in sys.argv
if VLMS:
    PRED += [('Molmo', 'Molmo 7B-D', FST.ALL_COLORS['Molmo'], None),
             ('Qwen35', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'], None)]
ROIS = [('early', 'Early visual'), ('ventral', 'Ventral'), ('lateral', 'Lateral'), ('parietal', 'Parietal')]
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 30, 20, 26, 36, 20

fig, axes = plt.subplots(2, 4, figsize=(24 if VLMS else 20, 11), sharey=True)
for r, (task, letter) in enumerate((('social', 'A'), ('describe', 'B'))):
    for c, (roi, title) in enumerate(ROIS):
        ax = axes[r, c]
        for k, (m, lab, col, hat) in enumerate(PRED):
            v = d[(d.task == task) & (d.roi == roi) & (d.model == m)].attn_unique
            assert len(v) == 8, (task, roi, m, len(v))
            ax.bar(k, v.mean(), 0.75, color=col, hatch=hat, edgecolor='black', linewidth=0.8, zorder=3)
            ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(8), fmt='none', color='black',
                        capsize=5, linewidth=1.6, zorder=4)
        ax.set_ylim(0, 0.07); ax.set_yticks([0, 0.02, 0.04, 0.06]); ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=10)
    axes[r, 0].set_ylabel(f'{task.capitalize()} task\nTest $R^2$', fontsize=FS_LABEL, labelpad=10)
    axes[r, 0].text(-0.42, 1.12, letter, transform=axes[r, 0].transAxes, fontsize=FS_LETTER,
                    fontweight='bold', va='top')
handles = [Patch(facecolor=col, hatch=hat, edgecolor='black', label=lab) for _, lab, col, hat in PRED]
fig.legend(handles=handles, loc='lower center', ncol=len(PRED), fontsize=FS_LEG, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(h_pad=2.5)
fig.savefig(OUT, dpi=250, bbox_inches='tight')
print('wrote', OUT)
