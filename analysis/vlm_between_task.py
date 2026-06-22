"""
vlm_between_task_analysis.py
------------------------------
Compute between-task spatial correlation for Molmo and Qwen using
the social-task best head, applied consistently to both tasks.

  Molmo:  Layer 9, Head 24 (social head) — both social and describe activations
  Qwen:   Layer 6, Head 8  (social head) — both social and describe activations

Between-task = per-image Spearman r between social attention map and
               describe attention map for the same head/layer.
Returns mean r and SEM across images.

Outputs:
  group_analysis/vlm_between_task.csv
  Columns: model, layer, head, mean_r, sem_r
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from scipy.stats import spearmanr
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
MOLMO_DIR   = '/scratch/gpfs/GRAZIANO/CANS/molmo/molmo'
PROBING_DIR = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
TARGET_CSV  = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
OUT_CSV     = os.path.join(GROUP_DIR, 'vlm_between_task.csv')

OUT_SIZE = 256

# ── Load target images ────────────────────────────────────────────────────────
target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_imgs = [x.replace('.jpg', '') for x in target_imgs]
print(f'Target images: {len(target_imgs)}')

records = []

# ══════════════════════════════════════════════════════════════════════════════
# MOLMO: L9H24 — both social and describe
# ══════════════════════════════════════════════════════════════════════════════
MOLMO_LAYER   = 9
MOLMO_HEAD    = 24   # 1-indexed
MOLMO_N_HEADS = 28
NUM_CROPS_X   = 3
NUM_CROPS_Y   = 3

print('\n=== Molmo between-task (L9H24) ===')

def setup_molmo_grid(act_dir):
    idx_path = os.path.join(act_dir, 'image_input_idx.pt')
    img_input_idx = torch.load(idx_path, map_location='cpu')[0].numpy()
    img_idx_all = (img_input_idx[1:]
                   .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                   .transpose(0, 2, 1, 3)
                   .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    valid_idx = np.where(img_idx_all > 0)
    images_all_h = NUM_CROPS_X * 336
    images_all_w = NUM_CROPS_Y * 336
    img_idx_all_resized = cv2.resize(
        img_idx_all, (images_all_w, images_all_h), interpolation=cv2.INTER_NEAREST)
    return img_idx_all, valid_idx, img_idx_all_resized, (images_all_h, images_all_w)

def extract_molmo_head_map(act_path, layer_idx, head_idx, img_idx_all, valid_idx,
                           img_idx_all_resized, images_all_shape):
    """Extract spatial map for a single head at a single layer. Returns (OUT_SIZE, OUT_SIZE)."""
    attn_maps = torch.load(act_path, map_location='cpu').numpy()
    n_tokens = attn_maps.shape[2]
    H_grid, W_grid = img_idx_all.shape
    images_all_h, images_all_w = images_all_shape

    img_positions = img_idx_all[valid_idx]
    vol = np.zeros((n_tokens, H_grid, W_grid), dtype=np.float32)
    for tok in range(n_tokens):
        vol[tok, valid_idx[0], valid_idx[1]] = attn_maps[layer_idx, head_idx, tok, img_positions]

    grid = vol.sum(axis=0)  # (H_grid, W_grid)

    row_valid = np.any(img_idx_all_resized > 0, axis=1)
    col_valid = np.any(img_idx_all_resized > 0, axis=0)

    resized = cv2.resize(grid, (images_all_w, images_all_h), interpolation=cv2.INTER_NEAREST)
    cropped = resized[row_valid][:, col_valid]
    out = cv2.resize(cropped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return out

# Setup grids (use social dir for idx — same tiling for all tasks)
soc_act_dir  = os.path.join(MOLMO_DIR, 'activations', 'social')
des_act_dir  = os.path.join(MOLMO_DIR, 'activations', 'describe')
soc_grid = setup_molmo_grid(soc_act_dir)
des_grid = setup_molmo_grid(des_act_dir)

layer_idx = MOLMO_LAYER
head_idx  = MOLMO_HEAD - 1  # 0-indexed

# Find images available in both tasks
soc_avail = {f.replace('activations_img_', '').replace('.pt', '')
             for f in os.listdir(soc_act_dir) if f.startswith('activations_img_') and f.endswith('.pt')}
des_avail = {f.replace('activations_img_', '').replace('.pt', '')
             for f in os.listdir(des_act_dir) if f.startswith('activations_img_') and f.endswith('.pt')}
molmo_common = [img for img in target_imgs if img in soc_avail and img in des_avail]
print(f'  Images in both tasks: {len(molmo_common)}')

molmo_rs = []
for img_name in tqdm(molmo_common):
    soc_path = os.path.join(soc_act_dir, f'activations_img_{img_name}.pt')
    des_path = os.path.join(des_act_dir, f'activations_img_{img_name}.pt')
    soc_map = extract_molmo_head_map(soc_path, layer_idx, head_idx, *soc_grid)
    des_map = extract_molmo_head_map(des_path, layer_idx, head_idx, *des_grid)
    sv, dv = soc_map.flatten().astype(np.float64), des_map.flatten().astype(np.float64)
    if np.std(sv) < 1e-12 or np.std(dv) < 1e-12:
        continue
    r, _ = spearmanr(sv, dv)
    if not np.isnan(r):
        molmo_rs.append(r)

molmo_rs = np.array(molmo_rs)
molmo_mean = float(molmo_rs.mean())
molmo_sem  = float(molmo_rs.std(ddof=1) / np.sqrt(len(molmo_rs)))
print(f'  Molmo between mean_r = {molmo_mean:.4f} ± {molmo_sem:.4f}  (n={len(molmo_rs)})')
records.append({'model': 'Molmo', 'layer': MOLMO_LAYER, 'head': MOLMO_HEAD,
                'mean_r': molmo_mean, 'sem_r': molmo_sem})

# ══════════════════════════════════════════════════════════════════════════════
# QWEN: L6H8 — both social and describe
# ══════════════════════════════════════════════════════════════════════════════
QWEN_LAYER  = 6
QWEN_HEAD   = 8   # 1-indexed
QWEN_N_HEADS = 28
GRID_SIDE   = 9

print('\n=== Qwen between-task (L6H8) ===')

qwen_soc_act_dir  = os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen', 'activations', f'attn-layer-{QWEN_LAYER}')
qwen_des_act_dir  = os.path.join(PROBING_DIR, 'output', 'coco',       'qwen', 'activations', f'attn-layer-{QWEN_LAYER}')
qwen_soc_mask_path = os.path.join(PROBING_DIR, 'output', 'social_cap', 'image_mask.pt')
qwen_des_mask_path = os.path.join(PROBING_DIR, 'output', 'coco',       'image_mask.pt')

soc_base_mask = torch.load(qwen_soc_mask_path, map_location='cpu')
soc_base_mask = soc_base_mask.squeeze(0) if soc_base_mask.dim() == 2 else soc_base_mask

des_base_mask = torch.load(qwen_des_mask_path, map_location='cpu')
des_base_mask = des_base_mask.squeeze(0) if des_base_mask.dim() == 2 else des_base_mask

def extract_qwen_head_map(act_path, base_mask, head_idx):
    """Extract spatial map for a single head. Returns (OUT_SIZE, OUT_SIZE)."""
    data = torch.load(act_path, map_location='cpu').squeeze(0)  # (N_HEADS, n_query, seq_len)
    avg  = data.float().mean(dim=1)                              # (N_HEADS, seq_len)
    seq_len = avg.shape[-1]
    pad_len = seq_len - base_mask.shape[0]
    if pad_len > 0:
        mask = F.pad(base_mask, (0, pad_len), value=False)
    else:
        mask = base_mask[:seq_len]
    masked = avg[:, mask]          # (N_HEADS, ~83)
    masked = masked[:, 1:-1]       # (N_HEADS, 81)
    n_tok = masked.shape[1]
    side  = int(np.sqrt(n_tok))
    if side * side != n_tok:
        masked = masked[:, :side*side]
    grid = masked[head_idx].reshape(side, side).numpy()   # (9, 9)
    out  = cv2.resize(grid, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    return out

head_idx = QWEN_HEAD - 1  # 0-indexed

soc_avail_q = {f.replace('.pt', '') for f in os.listdir(qwen_soc_act_dir) if f.endswith('.pt')}
des_avail_q = {f.replace('.pt', '') for f in os.listdir(qwen_des_act_dir) if f.endswith('.pt')}
qwen_common = [img for img in target_imgs if img in soc_avail_q and img in des_avail_q]
print(f'  Images in both tasks: {len(qwen_common)}')

qwen_rs = []
for img_name in tqdm(qwen_common):
    soc_path = os.path.join(qwen_soc_act_dir, f'{img_name}.pt')
    des_path = os.path.join(qwen_des_act_dir, f'{img_name}.pt')
    soc_map = extract_qwen_head_map(soc_path, soc_base_mask, head_idx)
    des_map = extract_qwen_head_map(des_path, des_base_mask, head_idx)
    sv, dv = soc_map.flatten().astype(np.float64), des_map.flatten().astype(np.float64)
    if np.std(sv) < 1e-12 or np.std(dv) < 1e-12:
        continue
    r, _ = spearmanr(sv, dv)
    if not np.isnan(r):
        qwen_rs.append(r)

qwen_rs   = np.array(qwen_rs)
qwen_mean = float(qwen_rs.mean())
qwen_sem  = float(qwen_rs.std(ddof=1) / np.sqrt(len(qwen_rs)))
print(f'  Qwen between mean_r = {qwen_mean:.4f} ± {qwen_sem:.4f}  (n={len(qwen_rs)})')
records.append({'model': 'Qwen', 'layer': QWEN_LAYER, 'head': QWEN_HEAD,
                'mean_r': qwen_mean, 'sem_r': qwen_sem})

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}')
print(df.to_string(index=False))
