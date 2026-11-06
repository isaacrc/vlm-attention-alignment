"""
slides_fig3_noise_ceiling_compare.py -- Figure 3 test R^2 as % of two noise ceilings, one row each:
  top     between-participant ceiling (as slides_fig3_noise_ceiling.py): per vertex, leave-one-participant-
          out Pearson r across the 907 shared images all 8 saw (prepare_responses.py), squared and averaged
          over the ROI. One ceiling for all participants; only stimulus-driven signal shared across people.
  bottom  within-participant ceiling (NSD's own): per vertex and participant, NC = ncsnr^2 / (ncsnr^2 + 1/3)
          for responses averaged over 3 repetitions (Allen et al., 2022), ncsnr from NSD's
          betas_fithrf_GLMdenoise_RR (task8_nsd/betas/ncsnr_subjXX.npy), averaged over the ROI. Matches the
          encoding model, which is fit within participant.
Both ceilings are averaged over the same analysed vertices as the ROI R^2 (ncsnr > 0.2, finite in all
participants; summarize_by_roi.py). Percent = each participant's ROI R^2 / ceiling (the within ceiling is the
participant's own); bars mean over the 8 participants, error bars SEM. Rows have their own y scales.
R^2: task8_nsd/results_norm/attnonly_per_subject_cvbehav.csv (attn_unique = raw test R^2), as Figure 3B/C.
Usage: python slides_fig3_noise_ceiling_compare.py [social|describe]
Writes final_figures/manuscript_2026-10-23/slides/fig3_neural_{task}_noiseceiling_compare_slide.png
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
TASK = sys.argv[1] if len(sys.argv) > 1 else 'social'
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides',
                   f'fig3_neural_{TASK}_noiseceiling_compare_slide.png')
ROIS = [('early', 'grp_early', 'Early visual'), ('ventral', 'grp_ventral', 'Ventral'),
        ('lateral', 'grp_lateral', 'Lateral'), ('parietal', 'grp_parietal', 'Parietal')]
PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),
        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
        ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None),
        ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///'),
        ('Luminance', 'Luminance', '#d0d0d0', '///')]
SUBJ = range(1, 9)

nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']                      # the 5,032 analysed vertices
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
sel = {roi: fit & masks[key].astype(bool) for roi, key, _ in ROIS}
BETWEEN = {roi: float((nc['ceiling'][sel[roi]] ** 2).mean()) for roi, _, _ in ROIS}
WITHIN = {}                                                  # (roi, subject) -> mean NC over the ROI
for s in SUBJ:
    sn = np.load(os.path.join(NSD, 'betas', f'ncsnr_subj{s:02d}.npy'))
    for roi, _, _ in ROIS:
        WITHIN[roi, s] = float((sn[sel[roi]] ** 2 / (sn[sel[roi]] ** 2 + 1 / 3)).mean())
d = pd.read_csv(os.path.join(NSD, 'results_norm', 'attnonly_per_subject_cvbehav.csv'))
d = d[d.task == TASK]

ROWS = [('between', 'Between-participant ceiling', 100), ('within', 'Within-participant ceiling (NSD)', 20)]
fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
L, R, GAP = 0.11, 0.985, 0.012
W = (R - L - 3 * GAP) / 4
for r, (kind, rlabel, top) in enumerate(ROWS):
    y0 = 0.54 if r == 0 else 0.15
    for c, (roi, _, title) in enumerate(ROIS):
        ax = fig.add_axes([L + c * (W + GAP), y0, W, 0.3])
        for k, (m, lab, col, hat) in enumerate(PRED):
            x = d[(d.roi == roi) & (d.model == m)].set_index('subject').attn_unique
            assert len(x) == 8
            ceil = pd.Series({s: BETWEEN[roi] if kind == 'between' else WITHIN[roi, s] for s in SUBJ})
            v = 100 * x / ceil[x.index]
            se = v.std(ddof=1) / np.sqrt(8)
            assert v.mean() + se <= top, (kind, roi, m, v.mean())
            ax.bar(k, v.mean(), 0.75, color=col, hatch=hat, edgecolor='black', linewidth=0.6, zorder=3)
            ax.errorbar(k, v.mean(), yerr=se, fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
            print(f'{kind:7s} {title:12s} {lab:12s} {v.mean():5.1f}%')
        ax.set_ylim(0, top); ax.set_yticks(np.linspace(0, top, 5)); ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=13, labelleft=(c == 0))
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        cv = BETWEEN[roi] if kind == 'between' else np.mean([WITHIN[roi, s] for s in SUBJ])
        ax.text(0.5, 0.98, f'ceiling = {cv:.3f}' + ('' if kind == 'between' else ' (mean)'), transform=ax.transAxes,
                ha='center', va='top', fontsize=11, color='#555555')
        if r == 0:
            ax.set_title(title, fontsize=22, fontweight='bold', pad=8)
        if c == 0:
            ax.set_ylabel(rlabel.replace(' ceiling', '\nceiling') + '\n(% of ceiling)', fontsize=13, labelpad=8)
fig.text(0.5, 0.965, f'{TASK.capitalize()} task: test $R^2$ as % of the noise ceiling', ha='center', va='center',
         fontsize=24, fontweight='bold')
fig.legend(handles=[Patch(facecolor=col, hatch=hat, edgecolor='black', label=lab) for _, lab, col, hat in PRED],
           loc='lower center', ncol=len(PRED), fontsize=14, frameon=False, bbox_to_anchor=(0.55, 0.01))
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', OUT)
