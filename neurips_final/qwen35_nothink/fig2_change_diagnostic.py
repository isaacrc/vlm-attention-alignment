"""
fig2_change_diagnostic.py
-------------------------
Why do the ViT-Transformer and Molmo columns of Figure 2 differ from the May 2026
figure (CANS/.../plot_fig5_qwen35.py -> fig5_qwen35_attn_by_task.png)?

The May figure used, for BOTH tasks: CNN-T L3H5, ViT-T L3H8, Molmo L10H24 (attn[9, 23]),
with Molmo's token grid Gaussian-blurred (sigma 1.5 tokens) BEFORE the 3x3-crop overlap
margins were removed. The current figure uses the cross-validated per-task heads
(Sept 2026: Social CNN-T L4H4, ViT-T L4H8; Describe Molmo L8H7) and, since 2026-10-20,
Molmo without the blur (the alignment analysis never blurred).

Each row isolates one change on the three Figure 2 images. Also reports, over all 200
images, the Spearman between each displayed Molmo map and the map the alignment
analysis actually scores (rebuttal/shared/vlm_maps.py), at 14x14.

Output: final_figures/qwen35_nothink/diagnostics/fig2_change_diagnostic.png
"""
import os, sys
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink', 'diagnostics')
os.makedirs(OUT, exist_ok=True)
src = open(SRC).read()
g = {'__file__': SRC, '__name__': 'diag'}
exec(compile(src[:src.index('# ── 1. entropy')], SRC, 'exec'), g)
np_, torch, cv2, plt = g['np'], g['torch'], g['cv2'], g['plt']
IMGS, SIZE = g['BLEND_IMGS'], g['SIZE']
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
import vlm_maps as V


def trans(model, task, L, H):
    g['HEADS'][model][task] = (L, H)
    return g['trans_maps'](model, task)


def molmo(task, L, H, mode, only=IMGS):
    """mode: 'blur_then_crop' (May/Sept figure), 'none' (current), 'crop_then_blur'."""
    act_dir = os.path.join(g['MOLMO_DIR'], 'activations', task)
    grid36 = g['_molmo_grid'](act_dir)[0]
    vidx = np.where(grid36 > 0); pos = grid36[vidx]
    rows, cols = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
    out = {}
    for im in only:
        a = torch.load(os.path.join(act_dir, f'activations_img_{im}.pt'), map_location='cpu')[L - 1, H - 1].float().numpy()
        vol = np.zeros((a.shape[0],) + grid36.shape, np.float32)
        vol[:, vidx[0], vidx[1]] = a[:, pos]
        s = vol.sum(0)
        if mode == 'blur_then_crop':
            s = cv2.GaussianBlur(s, (0, 0), 1.5)
        s = s[rows][:, cols]
        if mode == 'crop_then_blur':
            s = cv2.GaussianBlur(s, (0, 0), 1.5)
        out[im] = cv2.resize(s, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return out


ROWS = [
    ('ViT-T Social\nMay: L3H8', trans('ViT-Transformer', 'social', 3, 8)),
    ('ViT-T Social\nnow: L4H8', trans('ViT-Transformer', 'social', 4, 8)),
    ('ViT-T Describe\nMay = now: L3H8', trans('ViT-Transformer', 'describe', 3, 8)),
    ('Molmo Social L10H24\nMay: blur then crop', molmo('social', 10, 24, 'blur_then_crop')),
    ('Molmo Social L10H24\nnow: no blur', molmo('social', 10, 24, 'none')),
    ('Molmo Social L10H24\ncrop then blur', molmo('social', 10, 24, 'crop_then_blur')),
    ('Molmo Describe\nMay: L10H24, blur then crop', molmo('describe', 10, 24, 'blur_then_crop')),
    ('Molmo Describe\nSept: L8H7, blur then crop', molmo('describe', 8, 7, 'blur_then_crop')),
    ('Molmo Describe\nnow: L8H7, no blur', molmo('describe', 8, 7, 'none')),
    ('Molmo Describe\nL8H7, crop then blur', molmo('describe', 8, 7, 'crop_then_blur')),
]
fig, axes = plt.subplots(len(IMGS), len(ROWS), figsize=(3.2 * len(ROWS), 3.6 * len(IMGS)), dpi=110)
for c, (lab, mp) in enumerate(ROWS):
    for r, im in enumerate(IMGS):
        img = np.array(Image.open(os.path.join(g['IMG_DIR'], f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))) / 255.0
        heat = g['CMAP'](g['normalize'](mp[im]))[:, :, :3]
        axes[r, c].imshow(np.clip(img * 0.4 + heat * 0.6, 0, 1)); axes[r, c].axis('off')
    axes[0, c].set_title(lab, fontsize=10)
plt.tight_layout()
f = os.path.join(OUT, 'fig2_change_diagnostic.png'); fig.savefig(f); print('wrote', f)

# Which display matches what the alignment analysis scores?
to14 = g['to_14x14']
for task, (L, H) in (('social', (10, 24)), ('describe', (8, 7))):
    ana = V.molmo_maps(task, L, H)
    ids = [k[:-4] for k in ana][:200]
    for mode in ('blur_then_crop', 'none', 'crop_then_blur'):
        disp = molmo(task, L, H, mode, only=ids)
        rs = [spearmanr(to14(disp[i]), to14(ana[i + '.jpg']))[0] for i in ids]
        print(f'Molmo {task} L{L}H{H} {mode:15s} vs analysis map: median rho = {np.median(rs):.3f} '
              f'(5th pct {np.percentile(rs, 5):.3f}, n = {len(rs)})', flush=True)
