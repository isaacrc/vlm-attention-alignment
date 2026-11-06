"""
slides_roi_single.py -- each of the four encoding ROIs on its own: the four views of the Figure 3A atlas
(rebuttal/task8_nsd/plot_roi_atlas.py, grouped mode) with only one ROI coloured and the rest of the
cortex grey. Same masks (rois/roi_masks_fs5.npz; the four do not overlap), colours, surface and views.
Needs nilearn: run in the tribe_enc env.
Writes final_figures/manuscript_2026-10-23/slides/rois/roi_{early,ventral,lateral,parietal}.png
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import plotting, datasets as nidatasets

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']   # Nimbus Sans where Neue is not installed

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides', 'rois')
VERTS = 10242
GROUPED = [('grp_early', 'early', 'Early visual', '#DFC27D'),
           ('grp_ventral', 'ventral', 'Ventral', '#01665E'),
           ('grp_lateral', 'lateral', 'Lateral', '#762A83'),
           ('grp_parietal', 'parietal', 'Parietal', '#B2182B')]
VIEWS = [('left', 'lateral'), ('left', 'medial'), ('right', 'lateral'), ('right', 'medial')]

os.makedirs(OUT, exist_ok=True)
m = np.load(os.path.join(ROOT, 'rebuttal', 'task8_nsd', 'rois', 'roi_masks_fs5.npz'))
fs5 = nidatasets.fetch_surf_fsaverage('fsaverage5')
sulc = {h: fs5.get(f'sulc_{h}', fs5.get(f'curv_{h}')) for h in ('left', 'right')}
pial = {h: fs5[f'pial_{h}'] for h in ('left', 'right')}
for key, short, name, col in GROUPED:
    lab = m[key].astype(bool).astype(np.int32)
    half = {'left': lab[:VERTS], 'right': lab[VERTS:]}
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.08, wspace=0.01)
    for ax, (hemi, view) in zip(axes, VIEWS):
        plotting.plot_surf_roi(surf_mesh=pial[hemi], roi_map=half[hemi], hemi=hemi, view=view, bg_map=sulc[hemi],
                               cmap=ListedColormap(['#d9d9d9', col]), vmin=0, vmax=1, avg_method='median',
                               axes=ax, colorbar=False)
        ax.set_title(f'{hemi[0].upper()}H {view}', fontsize=22, fontweight='bold', y=0.02)
    fig.suptitle(name, fontsize=34, fontweight='bold', color=col, y=0.97)
    fig.savefig(os.path.join(OUT, f'roi_{short}.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'{name}: {lab.sum()} vertices')
print('wrote', OUT)
