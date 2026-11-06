"""
slides_fig3_roi_social.py -- fig3_neural_social_slide.png with each ROI drawn above its panel: one
surface view per ROI, only that ROI coloured (masks, colours and surface as slides_roi_single.py), the
Social-task bars of Figure 3B below (data and styling as molmo_bilinear/nsd_norm/plot_fig3_three_panel.py,
bars in the behavioral order of fig3_neural.py).
Font: Helvetica Neue, falling back to Nimbus Sans (a Helvetica clone) where Helvetica Neue is not
installed, as on Della.
Needs nilearn: run in the tribe_enc env.
Writes final_figures/manuscript_2026-10-23/slides/fig3_neural_social_rois_slide.png
       and the single-view brains to slides/rois/roi_{early,ventral,lateral,parietal}_view.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image, ImageChops

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Nimbus Sans:italic'
ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import figstyle as FST
SL = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides')
OUT = os.path.join(SL, 'fig3_neural_social_rois_slide.png')
VERTS = 10242
# (mask key, short name, title, colour, hemisphere, view): the view that shows most of each ROI
ROIS = [('grp_early', 'early', 'Early visual', '#DFC27D', 'left', 'medial'),
        ('grp_ventral', 'ventral', 'Ventral', '#01665E', 'left', 'medial'),
        ('grp_lateral', 'lateral', 'Lateral', '#762A83', 'right', 'lateral'),
        ('grp_parietal', 'parietal', 'Parietal', '#B2182B', 'right', 'lateral')]
PRED = [('CNN_LSTM', 'CNN-LSTM', FST.ALL_COLORS['CNN-LSTM'], None),
        ('CNN_Transformer', 'CNN-Transf.', FST.ALL_COLORS['CNN-Transformer'], None),
        ('ViT_LSTM', 'ViT-LSTM', FST.ALL_COLORS['ViT-LSTM'], None),
        ('ViT_Transformer', 'ViT-Transf.', FST.ALL_COLORS['ViT-Transformer'], None),
        ('EdgeEnergy', 'Edge energy', '#9a9a9a', '///'),
        ('Luminance', 'Luminance', '#d0d0d0', '///')]


def brain(key, short, col, hemi, view):
    """One surface view with one ROI coloured, cropped to the brain; cached in slides/rois/."""
    path = os.path.join(SL, 'rois', f'roi_{short}_view.png')
    if not os.path.exists(path):
        from nilearn import plotting, datasets as nidatasets
        fs5 = nidatasets.fetch_surf_fsaverage('fsaverage5')
        lab = np.load(os.path.join(ROOT, 'rebuttal', 'task8_nsd', 'rois', 'roi_masks_fs5.npz'))[key].astype(np.int32)
        lab = lab[:VERTS] if hemi == 'left' else lab[VERTS:]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
        plotting.plot_surf_roi(surf_mesh=fs5[f'pial_{hemi}'], roi_map=lab, hemi=hemi, view=view,
                               bg_map=fs5[f'sulc_{hemi}'], cmap=ListedColormap(['#d9d9d9', col]), vmin=0, vmax=1,
                               avg_method='median', axes=ax, colorbar=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=250, facecolor='white'); plt.close(fig)
        im = Image.open(path).convert('RGB')
        im.crop(ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()).save(path)
    return Image.open(path)


d = pd.read_csv(os.path.join(ROOT, 'rebuttal', 'task8_nsd', 'results_norm', 'attnonly_per_subject_cvbehav.csv'))
fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
L, R, GAP = 0.11, 0.985, 0.012
W = (R - L - 3 * GAP) / 4
for c, (key, short, title, col, hemi, view) in enumerate(ROIS):
    x0 = L + c * (W + GAP)
    im = brain(key, short, col, hemi, view)
    bh = W * 13.333 / 7.5 * im.height / im.width          # brain height in figure units at full panel width
    bax = fig.add_axes([x0, 0.585, W, min(bh, 0.24)]); bax.imshow(im); bax.axis('off')
    fig.text(x0 + W / 2, 0.86, title, ha='center', va='bottom', fontsize=24, fontweight='bold', color='black')
    ax = fig.add_axes([x0, 0.17, W, 0.38])
    for k, (m, lab, pc, hat) in enumerate(PRED):
        v = d[(d.task == 'social') & (d.roi == short) & (d.model == m)].attn_unique
        assert len(v) == 8
        ax.bar(k, v.mean(), 0.75, color=pc, hatch=hat, edgecolor='black', linewidth=0.6, zorder=3)
        ax.errorbar(k, v.mean(), yerr=v.std(ddof=1) / np.sqrt(8), fmt='none', color='black', capsize=3, linewidth=1.2, zorder=4)
    ax.set_ylim(0, 0.07); ax.set_yticks([0, 0.02, 0.04, 0.06]); ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=14, labelleft=(c == 0))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if c == 0:
        ax.set_ylabel('Social task\nTest $R^2$', fontsize=18, labelpad=8)
fig.legend(handles=[Patch(facecolor=pc, hatch=hat, edgecolor='black', label=lab) for _, lab, pc, hat in PRED],
           loc='lower center', ncol=len(PRED), fontsize=14, frameon=False, bbox_to_anchor=(0.55, 0.03))
fig.savefig(OUT, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', OUT)
