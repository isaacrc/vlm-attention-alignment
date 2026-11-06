"""
plot_fig3_three_panel.py
------------------------
Figure 3 variant with both tasks: (A) the ROI atlas (rebuttal/plots/nsd_roi_atlas_grouped_A.png,
unchanged), (B) Social and (C) Describe test R^2 from the normalized fits, the four trained
models and the two low-level controls (no VLMs). Bars as plot_trained_stacked.py.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import figstyle as FST
d = pd.read_csv(ROOT / 'rebuttal' / 'task8_nsd' / 'results_norm' / 'attnonly_per_subject_cvbehav.csv')
OUTDIR = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_norm'

PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),
        ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None),
        ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///'),
        ('Luminance', 'Luminance', '#d0d0d0', '///')]
ROIS = [('early', 'Early visual'), ('ventral', 'Ventral'), ('lateral', 'Lateral'), ('parietal', 'Parietal')]
FS_TITLE, FS_TICK, FS_LABEL, FS_LETTER, FS_LEG = 30, 20, 26, 40, 20

fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
for r, (task, letter) in enumerate((('social', 'B'), ('describe', 'C'))):
    for c, (roi, title) in enumerate(ROIS):
        ax = axes[r, c]
        for k, (m, lab, col, hat) in enumerate(PRED):
            v = d[(d.task == task) & (d.roi == roi) & (d.model == m)].attn_unique
            assert len(v) == 8
            ax.bar(k, v.mean(), 0.75, color=col, hatch=hat, edgecolor='black', linewidth=0.8, zorder=3)
            ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(8), fmt='none', color='black',
                        capsize=5, linewidth=1.6, zorder=4)
        ax.set_ylim(0, 0.07); ax.set_yticks([0, 0.02, 0.04, 0.06]); ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=10)
    axes[r, 0].set_ylabel(f'{task.capitalize()} task\nTest $R^2$', fontsize=FS_LABEL, labelpad=10)
    axes[r, 0].text(-0.42, 1.14, letter, transform=axes[r, 0].transAxes, fontsize=FS_LETTER,
                    fontweight='bold', va='top')
handles = [Patch(facecolor=col, hatch=hat, edgecolor='black', label=lab) for _, lab, col, hat in PRED]
fig.legend(handles=handles, loc='lower center', ncol=len(PRED), fontsize=FS_LEG, frameon=False,
           bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(h_pad=2.5)
bars_path = OUTDIR / '_fig3_bars_BC.png'
fig.savefig(bars_path, dpi=250, bbox_inches='tight')

A = Image.open(ROOT / 'rebuttal' / 'plots' / 'nsd_roi_atlas_grouped_A.png').convert('RGB')
B = Image.open(bars_path).convert('RGB')
B = B.resize((A.width, round(B.height * A.width / B.width)), Image.LANCZOS)
gap = 40
canvas = Image.new('RGB', (A.width, A.height + gap + B.height), 'white')
canvas.paste(A, (0, 0)); canvas.paste(B, (0, A.height + gap))
out = OUTDIR / 'fig3_neural_three_panel.png'
canvas.save(out)
print('wrote', out, canvas.size)
