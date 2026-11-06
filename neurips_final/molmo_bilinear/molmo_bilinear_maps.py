"""
molmo_bilinear_maps.py
----------------------
Molmo spatial maps resampled the same way as every other model: the 28x28 token grid is
bilinearly upsampled to 256x256.

The rebuttal loader (rebuttal/shared/vlm_maps.molmo_maps, and the 784-head sweep in
task1_head_selection/extract_molmo_alllayers.py) instead expanded the 36x36 crop grid to
pixels by nearest-neighbour, cropped the empty crop-overlap margins there, and shrank the
result to 256x256 with INTER_AREA -- in effect nearest-neighbour upsampling, which leaves
each token a flat block with hard edges. The pixel-level crop and the token-level crop
here select exactly the same tokens (every token is an integer 28-px block), so the only
change is the interpolation.

patch(V, EM) swaps both into the imported rebuttal modules.
"""
import os
import numpy as np
import torch
import cv2

OUT_SIZE = 256


def token_crop(grid36):
    """Rows/cols of the 36x36 crop grid that hold at least one image token (-> 28x28)."""
    rows, cols = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
    assert (grid36[rows][:, cols] > 0).all(), 'empty tokens inside the cropped Molmo grid'
    return rows, cols


def make_molmo_maps(V):
    def molmo_maps(task, layer=None, head=None):
        L, Hd = V.MOLMO_BEST[task] if layer is None else (layer, head)
        act_dir = os.path.join(V.MOLMO_DIR, 'activations', task)
        g, vidx, _, _ = V._molmo_grid(act_dir)
        pos = g[vidx]
        rows, cols = token_crop(g)
        out = {}
        for fn in os.listdir(act_dir):
            if not (fn.startswith('activations_img_') and fn.endswith('.pt')):
                continue
            im = fn.replace('activations_img_', '').replace('.pt', '')
            a = torch.load(os.path.join(act_dir, fn), map_location='cpu')[L - 1, Hd - 1].float().numpy()
            vol = np.zeros((a.shape[0],) + g.shape, np.float32)
            vol[:, vidx[0], vidx[1]] = a[:, pos]
            grid = vol.sum(0)[rows][:, cols]
            out[im + '.jpg'] = cv2.resize(grid, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
        return out
    return molmo_maps


def make_all_layer_head_maps(EM):
    def all_layer_head_maps(act_path, img_idx_all, valid_idx, resized, shape):
        """(N_LAYERS, N_HEADS, OUT_SIZE*OUT_SIZE) float32 for one image, bilinear."""
        attn = torch.load(act_path, map_location='cpu').float().numpy()   # (28,28,n_tok,1028)
        pos = img_idx_all[valid_idx]
        rows, cols = token_crop(img_idx_all)
        out = np.zeros((EM.N_LAYERS, EM.N_HEADS, OUT_SIZE * OUT_SIZE), np.float32)
        for L in range(EM.N_LAYERS):
            for h in range(EM.N_HEADS):
                vol = np.zeros((attn.shape[2],) + img_idx_all.shape, np.float32)
                vol[:, valid_idx[0], valid_idx[1]] = attn[L, h][:, pos]
                grid = vol.sum(0)[rows][:, cols]
                out[L, h] = cv2.resize(grid, (OUT_SIZE, OUT_SIZE),
                                       interpolation=cv2.INTER_LINEAR).ravel()
        return out
    return all_layer_head_maps


def patch(V, EM=None):
    V.molmo_maps = make_molmo_maps(V)
    if EM is not None:
        EM.all_layer_head_maps = make_all_layer_head_maps(EM)
