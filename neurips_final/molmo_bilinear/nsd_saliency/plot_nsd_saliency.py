"""
plot_nsd_saliency.py -- exploratory, not in the manuscript.
Test R^2 (attention-only, normalized maps) by region for the eight manuscript predictors
plus three bottom-up saliency models, Social (top) and Describe (bottom). Region means are
taken exactly as summarize_by_roi.py takes them (mean over the region's vertices, per
subject); bars are means over the 8 subjects, error bars SEM.
Writes final_figures/molmo_bilinear/nsd_saliency/fig_nsd_with_saliency.png + .csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import figstyle as FST
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_saliency'; OUT.mkdir(parents=True, exist_ok=True)

masks = np.load(NSD / 'rois' / 'roi_masks_fs5.npz')
ROIS = [('early', 'grp_early', 'Early visual'), ('ventral', 'grp_ventral', 'Ventral'),
        ('lateral', 'grp_lateral', 'Lateral'), ('parietal', 'grp_parietal', 'Parietal')]
PRED = [  # (file stem, label, colour, hatch, folder)
    ('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None, 'encoding_norm'),
    ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None, 'encoding_norm'),
    ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None, 'encoding_norm'),
    ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None, 'encoding_norm'),
    ('Molmo', 'Molmo 7B-D', FST.ALL_COLORS['Molmo'], None, 'encoding_norm'),
    ('Qwen35', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'], None, 'encoding_norm'),
    ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///', 'encoding_norm'),
    ('Luminance', 'Luminance', '#cfcfcf', '///', 'encoding_norm'),
    ('SalItti', 'Itti-Koch', '#7b5ea7', 'xx', 'encoding_norm_saliency'),
    ('SalGBVS', 'GBVS', '#a98bd0', 'xx', 'encoding_norm_saliency'),
    ('SalSR', 'Spectral residual', '#d6c6ec', 'xx', 'encoding_norm_saliency'),
]
rows = []
for stem, lab, col, hat, folder in PRED:
    for task in ('social', 'describe'):
        for s in range(1, 9):
            d = np.load(NSD / folder / f'r2_{stem}_{task}_subj{s:02d}_cvbehav.npz')
            for roi, key, _ in ROIS:
                sel = masks[key][d['vertices']]
                rows.append(dict(model=stem, task=task, subject=s, roi=roi, r2_attn=float(d['r2_attn'][sel].mean())))
df = pd.DataFrame(rows); df.to_csv(OUT / 'nsd_with_saliency_per_subject.csv', index=False)

fig, axes = plt.subplots(2, 4, figsize=(22, 11), sharey=True)
w = 0.8 / len(PRED)
for r, task in enumerate(('social', 'describe')):
    for c, (roi, _, title) in enumerate(ROIS):
        ax = axes[r, c]
        for k, (stem, lab, col, hat, _) in enumerate(PRED):
            v = df[(df.task == task) & (df.roi == roi) & (df.model == stem)].r2_attn
            ax.bar(k, v.mean(), 0.8, color=col, hatch=hat, edgecolor='black', linewidth=0.6,
                   yerr=v.std(ddof=1) / np.sqrt(len(v)), capsize=3, label=lab)
        ax.set_xticks([]); ax.set_ylim(0, 0.07)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=16)
        if r == 0:
            ax.set_title(title, fontsize=24, fontweight='bold')
    axes[r, 0].set_ylabel(f'{task.capitalize()} task\nTest $R^2$', fontsize=20)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=15,
           frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
LETTERS = '--letters' in sys.argv            # appendix version: Social = A, Describe = B
if LETTERS:
    for r, lab in enumerate('AB'):
        bb = axes[r, 0].get_position()
        fig.text(0.005, bb.y1 + 0.01, lab, fontsize=30, fontweight='bold', va='bottom', ha='left')
fig.savefig(OUT / ('fig_nsd_with_saliency_AB.png' if LETTERS else 'fig_nsd_with_saliency.png'),
            dpi=200, bbox_inches='tight')
print(df.groupby(['task', 'model', 'roi']).r2_attn.mean().unstack()[[r[0] for r in ROIS]].round(4).to_string())
