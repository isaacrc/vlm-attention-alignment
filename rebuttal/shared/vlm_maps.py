"""
vlm_maps.py — reconstruct Molmo/Qwen per-(layer,head) spatial maps at 256x256.
Adapts analysis/vlm_head_molmo.py and analysis/vlm_head_qwen35.py (single-head paths).
Returns {img_name_with_.jpg: (256,256) float32}.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

BASE        = '/scratch/gpfs/GRAZIANO/CANS'
MOLMO_DIR   = os.path.join(BASE, 'molmo', 'molmo')
PROBING_DIR = os.path.join(BASE, 'vlm-probing')
GROUP_DIR   = os.path.join(BASE, 'group_analysis')
OUT_SIZE = 256
NUM_CROPS_X = NUM_CROPS_Y = 3
QWEN_FULL_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]

# best (layer, head) 1-indexed per task.
# NOTE: indexing is 1-based here (molmo_maps reads attn[L-1, Hd-1]), while the
# paper quotes Molmo's best head 0-indexed as L9H24. The full 28x28 sweep
# (task1/results/molmo_allL_summary_*.csv) confirms the maximum sits at
# attn[9] = layer 10 in this convention: social L10H24 r=.2169 (paper .215),
# describe L10H27 r=.2157. The previous (9, ...) entry was off by one layer and
# under-reported Molmo by ~.04.
MOLMO_BEST = {'social': (10, 24), 'describe': (10, 27)}

def qwen_best(task):
    f = os.path.join(GROUP_DIR, f'qwen35_head_attention_allL_{task}.csv')
    df = pd.read_csv(f); r = df.loc[df['mean_r'].idxmax()]
    return int(r['layer']), int(r['head'])

# ── Molmo ──────────────────────────────────────────────────────────────────────
def _molmo_grid(act_dir):
    idx = torch.load(os.path.join(act_dir, 'image_input_idx.pt'), map_location='cpu')[0].numpy()
    g = (idx[1:].reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
         .transpose(0, 2, 1, 3).reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    vidx = np.where(g > 0)
    H, W = NUM_CROPS_X * 336, NUM_CROPS_Y * 336
    res = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
    return g, vidx, res, (H, W)

def molmo_maps(task, layer=None, head=None):
    L, Hd = MOLMO_BEST[task] if layer is None else (layer, head)
    act_dir = os.path.join(MOLMO_DIR, 'activations', task)
    g, vidx, res, (H, W) = _molmo_grid(act_dir)
    pos = g[vidx]; rowv = np.any(res > 0, axis=1); colv = np.any(res > 0, axis=0)
    out = {}
    for fn in os.listdir(act_dir):
        if not (fn.startswith('activations_img_') and fn.endswith('.pt')):
            continue
        im = fn.replace('activations_img_', '').replace('.pt', '')
        attn = torch.load(os.path.join(act_dir, fn), map_location='cpu').numpy()  # (28,28,ntok,1028)
        ntok = attn.shape[2]; Hg, Wg = g.shape
        vol = np.zeros((ntok, Hg, Wg), np.float32)
        vol[:, vidx[0], vidx[1]] = attn[L - 1, Hd - 1, :, pos].T
        grid = vol.sum(0)
        big = cv2.resize(grid, (W, H), interpolation=cv2.INTER_NEAREST)
        crop = big[rowv][:, colv]
        out[im + '.jpg'] = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return out

# ── Qwen3.5 ────────────────────────────────────────────────────────────────────
_QCFG = {
    'social':   dict(act=os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35', 'activations'),
                     mask=os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35', 'image_mask.pt')),
    'describe': dict(act=os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35', 'activations'),
                     mask=os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35', 'image_mask.pt')),
}

def qwen_maps(task, layer=None, head=None):
    if layer is None:
        layer, head = qwen_best(task)
    cfg = _QCFG[task]
    base_mask = torch.load(cfg['mask'], map_location='cpu')
    base_mask = base_mask.squeeze(0) if base_mask.dim() == 2 else base_mask
    ldir = os.path.join(cfg['act'], f'attn-layer-{layer}')
    out = {}
    for fn in os.listdir(ldir):
        if not fn.endswith('.pt'):
            continue
        im = fn.replace('.pt', '')
        data = torch.load(os.path.join(ldir, fn), map_location='cpu').squeeze(0)  # (16,nq,seq)
        avg = data.float().mean(dim=1)                                            # (16,seq)
        seq = avg.shape[-1]; pad = seq - base_mask.shape[0]
        mask = F.pad(base_mask, (0, pad), value=False) if pad > 0 else base_mask[:seq]
        masked = avg[:, mask][:, 1:-1]
        n = masked.shape[1]; side = int(np.sqrt(n))
        if side * side != n:
            masked = masked[:, :side * side]
        grid = masked[head - 1].reshape(side, side).numpy()
        out[im + '.jpg'] = cv2.resize(grid, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return out

def vlm_maps(model, task, layer=None, head=None):
    return molmo_maps(task, layer, head) if model == 'Molmo' else qwen_maps(task, layer, head)
