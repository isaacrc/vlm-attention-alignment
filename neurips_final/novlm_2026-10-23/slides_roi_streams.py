"""
slides_roi_streams.py -- the four encoding ROIs on the right-hemisphere lateral surface, with arrows for
the two classical visual streams (Ungerleider & Mishkin, 1982) drawn just outside the brain so the ROIs
stay fully visible: dorsal "where" above the occipito-parietal edge, ventral "what" below the inferior
temporal edge. A plain PNG cropped to its contents (no slide canvas, no legend).
Brain: slides/rois/roi_all_rh_lateral.png (masks and colours of Figure 3A, plot_roi_atlas.py grouped
mode, RH lateral view, rendered with nilearn in tribe_enc and cropped to the brain).
Font: Helvetica Neue, falling back to Nimbus Sans where it is not installed (Della).
Writes final_figures/manuscript_2026-10-23/slides/roi_streams.png
"""
import os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

plt.rcParams['font.family'] = ['Helvetica Neue', 'Nimbus Sans', 'DejaVu Sans']
SL = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/manuscript_2026-10-23/slides'
OUT = os.path.join(SL, 'roi_streams.png')
INK = '#1f2a36'

im = Image.open(os.path.join(SL, 'rois', 'roi_all_rh_lateral.png'))
W, H = im.size
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(im); ax.set_xlim(-250, W + 80); ax.set_ylim(H + 330, -360); ax.axis('off')

# (start, end, curvature, label, label position, label alignment), in image pixels; both arcs run outside the brain
STREAMS = [((-90, 520), (720, -110), -0.3, 'Dorsal stream\n“where”', (520, -270), 'center'),
           ((-10, 900), (950, 1220), 0.18, 'Ventral stream\n“what”', (1010, 1240), 'left')]
for (x0, y0), (x1, y1), rad, label, (lx, ly), ha in STREAMS:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}',
                                 arrowstyle='simple,head_width=1.6,head_length=1.6,tail_width=0.5',
                                 mutation_scale=22, fc='white', ec=INK, lw=2.2, zorder=5))
    ax.text(lx, ly, label, ha=ha, va='center', fontsize=24, fontweight='bold', color=INK, zorder=6)
fig.savefig(OUT, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.1); plt.close(fig)
print('wrote', OUT)
