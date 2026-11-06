"""
slides_fig7_no_lowlevel.py -- 16:9 slide versions (13.333 x 7.5 in, 300 dpi) of Figure 7 (appendix NSD:
trained models, saliency models, VLMs) without the two low-level image predictors, edge energy and
luminance. Data, bar order, colours and the within-participant noise-ceiling normalisation are those
of fig7_nsd_baselines_within_ceiling.py (Figure 7 of the camera-ready); font DejaVu Sans, as Figure 1.
Writes final_figures/slides/fig7_nsd_no_lowlevel_{social,describe,both}_slide.png
With --no_itti_vlm: also drops Itti-Koch, Molmo and Qwen3.5, and writes only
final_figures/slides/fig7_nsd_no_lowlevel_no_itti_vlm_both_slide.png
With --trained_only: the four trained models only, writes only
final_figures/slides/fig7_nsd_trained_only_both_slide.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['DejaVu Sans']
ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
NSD = os.path.join(ROOT, 'rebuttal', 'task8_nsd')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides')
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST
W, H_IN, DPI = 13.333, 7.5, 300

ROIS = [('early', 'Early visual'), ('ventral', 'Ventral'), ('lateral', 'Lateral'), ('parietal', 'Parietal')]
PRED = [  # (file stem, label, colour, hatch, folder) -- fig7_nsd_baselines.py without edge energy, luminance
    ('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None, 'encoding_norm'),
    ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None, 'encoding_norm'),
    ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None, 'encoding_norm'),
    ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None, 'encoding_norm'),
    ('SalItti', 'Itti-Koch', '#5e3c99', 'xx', 'encoding_norm_saliency'),
    ('SalGBVS', 'GBVS', '#7b5ea7', 'xx', 'encoding_norm_saliency'),
    ('SalSR', 'Spectral residual', '#a98bd0', 'xx', 'encoding_norm_saliency'),
    ('Molmo', 'Molmo 7B-D', FST.ALL_COLORS['Molmo'], None, 'encoding_norm'),
    ('Qwen35', 'Qwen 9B-3.5', FST.ALL_COLORS['Qwen3.5'], None, 'encoding_norm'),
]
NO_ITTI_VLM = '--no_itti_vlm' in sys.argv
if NO_ITTI_VLM:
    PRED = [p for p in PRED if p[0] not in ('SalItti', 'Molmo', 'Qwen35')]
TRAINED_ONLY = '--trained_only' in sys.argv
if TRAINED_ONLY:
    PRED = PRED[:4]
SUFFIX = '_no_itti_vlm' if NO_ITTI_VLM else ''
NCOL = 6 if NO_ITTI_VLM else 4 if TRAINED_ONLY else 5
BOTH_NAME = 'fig7_nsd_trained_only_both_slide.png' if TRAINED_ONLY else f'fig7_nsd_no_lowlevel{SUFFIX}_both_slide.png'

# within-participant noise ceiling per participant and region (as fig7_nsd_baselines_within_ceiling.py)
nc = np.load(os.path.join(NSD, 'responses', 'noise_ceiling.npz'))
fit = (nc['ncsnr'] > 0.2) & nc['valid']
masks = np.load(os.path.join(NSD, 'rois', 'roi_masks_fs5.npz'))
NC = {}
for s in range(1, 9):
    sn = np.load(os.path.join(NSD, 'betas', f'ncsnr_subj{s:02d}.npy'))
    for roi, _ in ROIS:
        v = sn[fit & masks[f'grp_{roi}'].astype(bool)]
        NC[s, roi] = float((v ** 2 / (v ** 2 + 1 / 3)).mean())

rows = []
for stem, _, _, _, folder in PRED:
    for task in ('social', 'describe'):
        for s in range(1, 9):
            d = np.load(os.path.join(NSD, folder, f'r2_{stem}_{task}_subj{s:02d}_cvbehav.npz'))
            for roi, _ in ROIS:
                sel = masks[f'grp_{roi}'][d['vertices']]
                rows.append(dict(model=stem, task=task, subject=s, roi=roi,
                                 pct=100 * float(d['r2_attn'][sel].mean()) / NC[s, roi]))
df = pd.DataFrame(rows)
agg = df.groupby(['task', 'model', 'roi']).pct.agg(['mean', lambda v: v.std(ddof=1) / np.sqrt(len(v))])
agg.columns = ['mean', 'sem']
YMAX = 20
assert (agg['mean'] + agg['sem']).max() <= YMAX


def panel_row(axes, task, titles=True):
    for ax, (roi, title) in zip(axes, ROIS):
        for k, (stem, lab, col, hat, _) in enumerate(PRED):
            m, e = agg.loc[(task, stem, roi)]
            ax.bar(k, m, 0.8, color=col, hatch=hat, edgecolor='black', linewidth=0.6, label=lab, zorder=3)
            ax.errorbar(k, m, yerr=e, fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
        ax.set_xticks([]); ax.set_ylim(0, YMAX)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=16)
        if titles:
            ax.set_title(title, fontsize=22, fontweight='bold')


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, facecolor='white'); plt.close(fig)
    print('wrote', path)


for task in (() if NO_ITTI_VLM or TRAINED_ONLY else ('social', 'describe')):
    fig, axes = plt.subplots(1, 4, figsize=(W, H_IN), sharey=True)
    panel_row(axes, task)
    axes[0].set_ylabel(f'{task.capitalize()} task\n% of noise ceiling', fontsize=20)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    save(fig, f'fig7_nsd_no_lowlevel_{task}_slide.png')

fig, axes = plt.subplots(2, 4, figsize=(W, H_IN), sharey=True)
for r, task in enumerate(('social', 'describe')):
    panel_row(axes[r], task, titles=(r == 0))
    axes[r, 0].set_ylabel(f'{task.capitalize()} task\n% of noise ceiling', fontsize=16)
for ax in axes.flat:
    ax.tick_params(axis='y', labelsize=13)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=NCOL, fontsize=13,
           frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.1, 1, 1])
save(fig, BOTH_NAME)
print(agg['mean'].unstack()[[r[0] for r in ROIS]].round(2).to_string())
