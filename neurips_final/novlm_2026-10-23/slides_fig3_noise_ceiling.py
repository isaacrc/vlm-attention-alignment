"""
slides_fig3_noise_ceiling.py -- fig3_neural_social_slide.png with test R^2 expressed as % of the NSD
noise ceiling instead of raw R^2.
Noise ceiling: the cross-participant ceiling of rebuttal/task8_nsd/prepare_responses.py (per vertex,
leave-one-participant-out Pearson r across the 907 shared images all 8 participants saw, Fisher-z
averaged), squared per vertex and averaged over each ROI's analysed vertices (ncsnr > 0.2, finite in all
participants), the same vertices and the same per-vertex averaging as the ROI R^2 (summarize_by_roi.py).
Percent = each participant's ROI R^2 / ROI ceiling r^2; bars are the mean over the 8 participants, error
bars SEM. The ceiling is a between-participant bound and the encoding model is fit within participant,
so this is an approximate scale, not a strict upper limit.
R^2: task8_nsd/results_norm/attnonly_per_subject_cvbehav.csv (attn_unique = raw test R^2), as Figure 3B.
Usage: python slides_fig3_noise_ceiling.py [social|describe]
Writes final_figures/manuscript_2026-10-23/slides/fig3_neural_{social,describe}_noiseceiling_slide.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']   # Nimbus Sans where Neue is not installed
ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NSD = os.path.join(ROOT, 'rebuttal', 'task8_nsd')
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST
TASK = sys.argv[1] if len(sys.argv) > 1 else 'social'   # social | describe
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides',
                   f'fig3_neural_{TASK}_noiseceiling_slide.png')
ROIS = [('early', 'grp_early', 'Early visual'), ('ventral', 'grp_ventral', 'Ventral'),
        ('lateral', 'grp_lateral', 'Lateral'), ('parietal', 'grp_parietal', 'Parietal')]
PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),
        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
        ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None),
        ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///'),
        ('Luminance', 'Luminance', '#d0d0d0', '///')]

nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']                      # the 5,032 analysed vertices
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
CEIL = {roi: float((nc['ceiling'][fit & masks[key].astype(bool)] ** 2).mean()) for roi, key, _ in ROIS}
print('ceiling r^2:', {k: round(v, 4) for k, v in CEIL.items()})
d = pd.read_csv(os.path.join(NSD, 'results_norm', 'attnonly_per_subject_cvbehav.csv'))

fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
L, R, GAP = 0.11, 0.985, 0.012
W = (R - L - 3 * GAP) / 4
ymax = 0
for c, (roi, _, title) in enumerate(ROIS):
    ax = fig.add_axes([L + c * (W + GAP), 0.2, W, 0.62])
    for k, (m, lab, col, hat) in enumerate(PRED):
        v = 100 * d[(d.task == TASK) & (d.roi == roi) & (d.model == m)].attn_unique / CEIL[roi]
        assert len(v) == 8
        se = v.std(ddof=1) / np.sqrt(8); ymax = max(ymax, v.mean() + se)
        ax.bar(k, v.mean(), 0.75, color=col, hatch=hat, edgecolor='black', linewidth=0.6, zorder=3)
        ax.errorbar(k, v.mean(), yerr=se, fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
        print(f'{title:12s} {lab:12s} {v.mean():5.1f}%')
    ax.set_xticks([]); ax.tick_params(axis='y', labelsize=14, labelleft=(c == 0))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=24, fontweight='bold', pad=10)
    ax.text(0.5, 0.97, f'ceiling $r^2$ = {CEIL[roi]:.3f}', transform=ax.transAxes, ha='center', va='top',
            fontsize=12, color='#555555')
    if c == 0:
        ax.set_ylabel(f'{TASK.capitalize()} task\n% of noise ceiling', fontsize=18, labelpad=8)
top = 100   # fixed, so the Social and Describe slides share a scale
assert ymax <= top, ymax
for ax in fig.axes:
    ax.set_ylim(0, top)
fig.legend(handles=[Patch(facecolor=col, hatch=hat, edgecolor='black', label=lab) for _, lab, col, hat in PRED],
           loc='lower center', ncol=len(PRED), fontsize=14, frameon=False, bbox_to_anchor=(0.55, 0.05))
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', OUT)
