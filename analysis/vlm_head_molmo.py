"""
molmo_head_attention_analysis.py
---------------------------------
For Molmo (7B-D), extract per-head spatial attention maps at Layer 9
(the layer with peak spatial correlation) and correlate each of the 28 heads
against human gaze heatmaps for the Social and Describe tasks.

Raw activations come from:
  molmo/molmo/activations/{task}/activations_img_{id}.pt
  Shape: (28 layers, 28 heads, n_tokens, 1028)

Spatial reconstruction follows plot_attn.py:
  - img_input_idx maps 2D grid positions → attention vector indices
  - For each head at layer 9: sum over caption tokens → 2D spatial map
  - Resize to 336×336 (human heatmap resolution)

Output:
  group_analysis/molmo_head_attention_L9.csv
  Columns: head, task, mean_r, sem_r
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from scipy.stats import rankdata as scipy_rankdata
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
MOLMO_DIR   = '/scratch/gpfs/GRAZIANO/CANS/molmo/molmo'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
TARGET_CSV  = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')
HM_DIRS = {
    'social':   os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social'),
    'describe': os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive'),
}
OUT_CSV = os.path.join(GROUP_DIR, 'molmo_head_attention_L9.csv')

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER     = 9
N_HEADS   = 28
N_BLOCKS  = 28
OUT_SIZE  = 256          # standardize all maps to 256×256 for correlation
# Molmo uses 3×3 crop tiling for COCO images (verified from image_input_idx shape)
NUM_CROPS_X = 3
NUM_CROPS_Y = 3

# ── Load target images ────────────────────────────────────────────────────────
target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
# Strip .jpg if present
target_imgs = [x.replace('.jpg', '') for x in target_imgs]
print(f'Target images: {len(target_imgs)}')


# ── Spatial reconstruction setup (same for all images in a task) ──────────────
def setup_spatial_grid(act_dir):
    """
    Load image_input_idx and compute the spatial index grid (img_idx_all)
    and valid positions (valid_idx) used to map attention weights onto the
    image plane. This is shared across all images for a given task.

    Returns: (img_idx_all, valid_idx, img_idx_all_resized, images_all_shape)
      - img_idx_all:          (H_grid, W_grid) int array; 0 = padding
      - valid_idx:            tuple of (row_indices, col_indices) for non-zero positions
      - images_all_shape:     (H_img, W_img) of the full reconstructed image (for resize ref)
    """
    idx_path = os.path.join(act_dir, 'image_input_idx.pt')
    img_input_idx = torch.load(idx_path, map_location='cpu')[0].numpy()
    # img_input_idx shape: (10, 144) → [0] = global crop, [1:] = 9 local crops
    # Reshape local crops to (3, 3, 12, 12) grid, transpose to (36, 36)
    img_idx_all = (img_input_idx[1:]
                   .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                   .transpose(0, 2, 1, 3)
                   .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))

    valid_idx = np.where(img_idx_all > 0)

    # Full image dimensions (NUM_CROPS × 336 per side — Molmo native resolution)
    images_all_h = NUM_CROPS_X * 336
    images_all_w = NUM_CROPS_Y * 336

    img_idx_all_resized = cv2.resize(
        img_idx_all, (NUM_CROPS_Y * 336, NUM_CROPS_X * 336), interpolation=cv2.INTER_NEAREST
    )

    return img_idx_all, valid_idx, img_idx_all_resized, (images_all_h, images_all_w)


# ── Extract per-head map for one image at one layer ───────────────────────────
def extract_per_head_maps(act_path, layer_idx, img_idx_all, valid_idx,
                          img_idx_all_resized, images_all_shape):
    """
    Load raw activations for one image, extract spatial maps for each head
    at layer_idx, and return array of shape (N_HEADS, OUT_SIZE, OUT_SIZE).
    """
    attn_maps = torch.load(act_path, map_location='cpu').numpy()
    # attn_maps shape: (28 layers, 28 heads, n_tokens, 1028)
    n_tokens = attn_maps.shape[2]
    images_all_h, images_all_w = images_all_shape  # (NUM_CROPS_X*336, NUM_CROPS_Y*336)

    # Build spatial attention volume for this layer: (N_HEADS, n_tokens, H_grid, W_grid)
    # initialise with -1 (padding indicator)
    H_grid, W_grid = img_idx_all.shape
    attn_vol = np.full((N_HEADS, n_tokens, H_grid, W_grid), -1.0, dtype=np.float32)

    img_positions = img_idx_all[valid_idx]          # 1D indices into the 1028-dim vector
    for head in range(N_HEADS):
        for tok in range(n_tokens):
            attn_vol[head, tok, valid_idx[0], valid_idx[1]] = \
                attn_maps[layer_idx, head, tok, img_positions]

    # Sum over tokens → (N_HEADS, H_grid, W_grid)
    head_maps_grid = attn_vol.sum(axis=1)

    row_valid = np.any(img_idx_all_resized > 0, axis=1)
    col_valid = np.any(img_idx_all_resized > 0, axis=0)

    out = np.zeros((N_HEADS, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    for head in range(N_HEADS):
        # Resize grid to full image plane (Molmo native 336-based resolution)
        resized = cv2.resize(head_maps_grid[head], (images_all_w, images_all_h),
                             interpolation=cv2.INTER_NEAREST)
        # Crop to valid image region, then resize to OUT_SIZE for correlation
        cropped = resized[row_valid][:, col_valid]
        out[head] = cv2.resize(cropped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    return out   # (N_HEADS, OUT_SIZE, OUT_SIZE)


# ── Human heatmap loading & pre-ranking ──────────────────────────────────────
def load_and_prerank_human_heatmaps(task, valid_imgs):
    """
    Load human heatmaps for valid_imgs, pre-rank each subject's heatmaps
    (Spearman average-tie ranking across pixels).
    Returns: (hum_ranked, hum_valid)
      hum_ranked: (n_subs, n_imgs, OUT_SIZE*OUT_SIZE) float32, pre-ranked
      hum_valid:  (n_subs, n_imgs) bool
    """
    hm_dir = HM_DIRS[task]
    sub_files = sorted([f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy')])
    n_subs = len(sub_files)
    n_imgs = len(valid_imgs)
    flat = OUT_SIZE * OUT_SIZE

    hum_ranked = np.zeros((n_subs, n_imgs, flat), dtype=np.float32)
    hum_valid  = np.zeros((n_subs, n_imgs), dtype=bool)

    for si, fname in enumerate(sub_files):
        sub_data = np.load(os.path.join(hm_dir, fname), allow_pickle=True).item()
        rows_raw   = []
        row_valid  = []
        for img_name in valid_imgs:
            key_jpg  = img_name + '.jpg'
            key_bare = img_name
            hum = sub_data.get(key_jpg, sub_data.get(key_bare, None))
            if hum is not None:
                hum_clean = np.nan_to_num(hum, nan=0.0).astype(np.float32)
                hum_resized = np.array(
                    Image.fromarray(hum_clean).resize((OUT_SIZE, OUT_SIZE), resample=Image.NEAREST),
                    dtype=np.float32).flatten()
                rows_raw.append(hum_resized)
                row_valid.append(True)
            else:
                rows_raw.append(np.zeros(flat, dtype=np.float32))
                row_valid.append(False)

        hum_valid[si] = row_valid
        valid_i = np.where(row_valid)[0]
        if len(valid_i):
            mat = np.stack([rows_raw[i] for i in valid_i])   # (n_valid, flat)
            hum_ranked[si, valid_i] = scipy_rankdata(
                mat, method='average', axis=1
            ).astype(np.float32)

    return hum_ranked, hum_valid


# ── Spearman r: model head vs. human ─────────────────────────────────────────
def spearman_r_flat(x_ranked, y_ranked):
    """Pearson r on pre-ranked vectors = Spearman r."""
    xm = x_ranked - x_ranked.mean()
    ym = y_ranked - y_ranked.mean()
    num = (xm * ym).sum()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-10
    return float(num / denom)


def compute_head_correlations(head_maps, hum_ranked, hum_valid, valid_imgs, model_img_ids):
    """
    head_maps: dict {img_id: (N_HEADS, OUT_SIZE, OUT_SIZE)}
    Returns: (mean_r_per_head, sem_r_per_head) arrays of shape (N_HEADS,)
    """
    n_subs, n_imgs, flat = hum_ranked.shape
    # For each head, collect per-subject mean Spearman r
    head_sub_r = np.full((N_HEADS, n_subs), np.nan)

    for si in range(n_subs):
        for head in range(N_HEADS):
            rs = []
            for ii, img_name in enumerate(valid_imgs):
                if not hum_valid[si, ii]:
                    continue
                if img_name not in model_img_ids:
                    continue
                model_map = head_maps[img_name][head].flatten()
                model_ranked = scipy_rankdata(model_map, method='average').astype(np.float32)
                r = spearman_r_flat(model_ranked, hum_ranked[si, ii])
                rs.append(r)
            if rs:
                head_sub_r[head, si] = np.mean(rs)

    mean_r = np.nanmean(head_sub_r, axis=1)
    # SEM across subjects
    n_valid = np.sum(~np.isnan(head_sub_r), axis=1)
    sem_r   = np.nanstd(head_sub_r, axis=1) / np.sqrt(np.maximum(n_valid, 1))

    return mean_r, sem_r


# ── Main ──────────────────────────────────────────────────────────────────────
records = []

for task in ['social', 'describe']:
    print(f'\n=== Task: {task} ===')
    act_dir = os.path.join(MOLMO_DIR, 'activations', task)

    # Spatial grid setup (shared across images)
    img_idx_all, valid_idx, img_idx_all_resized, images_all_shape = \
        setup_spatial_grid(act_dir)

    # Find which target images have activations
    available = set()
    for fname in os.listdir(act_dir):
        if fname.startswith('activations_img_') and fname.endswith('.pt'):
            img_id = fname.replace('activations_img_', '').replace('.pt', '')
            available.add(img_id)

    valid_imgs = [img for img in target_imgs if img in available]
    print(f'  Images with activations: {len(valid_imgs)} / {len(target_imgs)}')

    # Pre-rank human heatmaps
    print('  Pre-ranking human heatmaps...')
    hum_ranked, hum_valid = load_and_prerank_human_heatmaps(task, valid_imgs)
    print(f'  Subjects: {hum_ranked.shape[0]}')

    # Extract per-head maps for all images
    print(f'  Extracting per-head maps at layer {LAYER}...')
    head_maps = {}
    for img_name in tqdm(valid_imgs):
        act_path = os.path.join(act_dir, f'activations_img_{img_name}.pt')
        maps = extract_per_head_maps(
            act_path, LAYER, img_idx_all, valid_idx,
            img_idx_all_resized, images_all_shape
        )
        head_maps[img_name] = maps   # (28, 336, 336)

    # Correlate each head vs. human gaze
    print('  Computing per-head correlations...')
    model_img_ids = set(head_maps.keys())
    mean_r, sem_r = compute_head_correlations(
        head_maps, hum_ranked, hum_valid, valid_imgs, model_img_ids
    )

    for head in range(N_HEADS):
        records.append({
            'task':   task,
            'head':   head + 1,   # 1-indexed
            'mean_r': mean_r[head],
            'sem_r':  sem_r[head],
        })

    best_head = int(np.argmax(mean_r)) + 1
    print(f'  Best head: {best_head}  (mean_r = {mean_r[best_head-1]:.4f})')

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}')
for task, grp in df.groupby('task'):
    best = grp.loc[grp['mean_r'].idxmax()]
    print(f"  {task}: best head={int(best['head'])}, mean_r={best['mean_r']:.4f}")
