"""
fig_baselines_rsa.py -- exploratory two-panel figure for the manuscript_2026-10-23 folder.
  B  map-to-map Spearman r between every predictor, average of Social and Describe
     (molmo_bilinear/nsd_saliency/map_rsa_coco_priors.py output, re-drawn in B's order)
  A  alignment with human gaze as % of the pairwise human upperbound, averaged over the two
     tasks (mean of the Social and Describe percentages; SEM = sqrt(SEM_s^2 + SEM_d^2) / 2,
     the participant groups being independent). Model bars from alignment_long_cv_honest.parquet
     (Figure 1 values), baselines from fig1_baselines/baseline_subject_z.csv (unblurred, as Figure 1).
Both panels list the predictors in one order: trained models | image statistics (saliency and low-level,
one purple group) | priors | VLMs.
Writes final_figures/manuscript_2026-10-23/fig_baselines_rsa.png
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST

FF = os.path.join(ROOT, 'neurips_final', 'final_figures')
CEIL = json.load(open(os.path.join(FF, 'molmo_bilinear_pairwise', 'results', 'ceilings.json')))
mod = (pd.read_parquet(os.path.join(FF, 'molmo_bilinear_pairwise', 'results', 'alignment_long_cv_honest.parquet'))
         .groupby(['model', 'task', 'subject_id']).z.mean().rename('mean_z').reset_index())
base = pd.read_csv(os.path.join(FF, 'molmo_bilinear', 'fig1_baselines', 'baseline_subject_z.csv'))
d = pd.concat([mod, base[['model', 'task', 'subject_id', 'mean_z']]])
rsa = pd.read_csv(os.path.join(FF, 'molmo_bilinear', 'nsd_saliency', 'map_rsa_coco_priors_spearman_average.csv'), index_col=0)

# name in d, name in the RSA csv, label, colour, hatch
ITEMS = [('CNN-LSTM', 'CNN-LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
         ('CNN-Transformer', 'CNN-T', 'CNN-Trans.', FST.ALL_COLORS['CNN-Transformer'], None),
         ('ViT-LSTM', 'ViT-LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
         ('ViT-Transformer', 'ViT-T', 'ViT-Trans.', FST.ALL_COLORS['ViT-Transformer'], None),
         ('Itti-Koch', 'Itti-Koch', 'Itti-Koch', '#5e3c99', 'xx'),
         ('GBVS', 'GBVS', 'GBVS', '#7b5ea7', 'xx'),
         ('Spectral residual', 'Spec. res.', 'Spec. residual', '#a98bd0', 'xx'),
         ('Edge energy', 'Edge', 'Edge energy', '#c9b6e4', 'xx'),
         ('Luminance', 'Luminance', 'Luminance', '#e8dff5', 'xx'),
         ('Center prior', 'Center', 'Center prior', '#8c6d31', '..'),
         ('Person prior', 'Person', 'Person prior', '#bd9e39', '..'),
         ('Face prior', 'Face', 'Face prior', '#e7cb94', '..'),
         ('Molmo', 'Molmo', 'Molmo 7B-D', FST.ALL_COLORS['Molmo'], None),
         ('Qwen3.5', 'Qwen3.5', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'], None)]
GROUP_ENDS = [4, 9, 12]             # trained | image statistics | priors | VLMs
FS_TITLE, FS_TICK, FS_LABEL, FS_CELL, FS_LETTER = 30, 20, 24, 13, 40

fig = plt.figure(figsize=(32, 13.5))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1], wspace=0.30)

# B: map-to-map similarity (right)
ax = axM = fig.add_subplot(gs[1])
keys = [it[1] for it in ITEMS]; labels = [it[2] for it in ITEMS]
M = rsa.loc[keys, keys].values.astype(float); np.fill_diagonal(M, np.nan)
n = len(keys)
im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
ax.set_facecolor('#e6e6e6')
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f'{M[i, j]:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=FS_CELL,
                    color='white' if abs(M[i, j]) > 0.5 else '#222222')
for b in GROUP_ENDS:
    ax.axhline(b - 0.5, color='black', lw=1.5); ax.axvline(b - 0.5, color='black', lw=1.5)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=FS_TICK)
ax.set_yticklabels(labels, fontsize=FS_TICK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title('Map-to-map similarity', fontsize=FS_TITLE, fontweight='bold', pad=14)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.ax.tick_params(labelsize=FS_TICK - 2)
cb.set_label(r'Spearman $r$ between maps', fontsize=FS_LABEL - 2)

# A: alignment, mean of the two tasks (left)
ax = axBar = fig.add_subplot(gs[0])
x = np.arange(n, dtype=float)
for b in GROUP_ENDS:
    x[b:] += 0.6
means, sems = [], []
for name, *_ in ITEMS:
    pct, se = [], []
    for t in ('social', 'describe'):
        v = d[(d.task == t) & (d.model == name)].mean_z
        pct.append(v.mean() / CEIL[t] * 100); se.append(v.std(ddof=1) / np.sqrt(len(v)) / CEIL[t] * 100)
    means.append(np.mean(pct)); sems.append(np.sqrt(se[0] ** 2 + se[1] ** 2) / 2)
for xi, m, (_, _, _, c, h) in zip(x, means, ITEMS):
    ax.bar(xi, m, 0.7, color=c, hatch=h, edgecolor='black', linewidth=0.6, zorder=3)
ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, linewidth=1.8, zorder=4)
ax.axhline(100, color='black', linestyle='--', linewidth=2.2, zorder=2)
ax.axhline(0, color='black', linewidth=1.0, zorder=2)
for b in GROUP_ENDS:
    ax.axvline((x[b - 1] + x[b]) / 2, color='gray', linewidth=1.0, linestyle=':', zorder=1)
ax.text(0.99, 102, 'Human Upperbound', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=FS_TICK)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=FS_TICK)
ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
ax.set_ylim(-20, 120); ax.set_yticks([0, 50, 100]); ax.tick_params(axis='y', labelsize=FS_TICK + 2)
ax.set_ylabel('% of Human Upperbound', fontsize=FS_LABEL, labelpad=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0); ax.set_axisbelow(True)
ax.set_title('Alignment with human gaze', fontsize=FS_TITLE, fontweight='bold', pad=14)
# A spans the matrix's height, so the two titles and letters share a baseline
fig.canvas.draw()
pa, pb = axM.get_position(), axBar.get_position()
axBar.set_position([pb.x0, pa.y0, pb.width, pa.height])
y = pa.y1 + 0.035
fig.text(pb.x0 - 0.045, y, 'A', fontsize=FS_LETTER, fontweight='bold', va='bottom')
fig.text(pa.x0 - 0.085, y, 'B', fontsize=FS_LETTER, fontweight='bold', va='bottom')
print(' '.join(f'{it[2]}={m:.1f}' for it, m in zip(ITEMS, means)))

out = os.path.join(FF, 'manuscript_2026-10-23', 'fig_baselines_rsa.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print('wrote', out)
