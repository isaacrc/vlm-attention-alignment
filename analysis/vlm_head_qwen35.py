"""
qwen35_head_attention_analysis.py
----------------------------------
Per-head spatial alignment analysis for Qwen3.5-9B.

Qwen3.5-9B has 32 layers but only 8 use full (standard) attention:
  indices [3, 7, 11, 15, 19, 23, 27, 31]
Runs 8 layers × 16 heads for both tasks; picks the best (layer, head)
combination per task and writes two task-specific CSVs.

Raw activations:
  vlm-probing/output/coco/qwen35/activations/attn-layer-{N}/{id}.pt      (describe)
  vlm-probing/output/social_cap/qwen35/activations/attn-layer-{N}/{id}.pt (social)
  Shape: (1, N_HEADS, n_gen_steps, seq_len)

Outputs:
  group_analysis/qwen35_head_attention_allL_social.csv
  group_analysis/qwen35_head_attention_allL_describe.csv
    Columns: task, layer, head, mean_r, sem_r

  group_analysis/qwen35_head_attention_bestL_social.csv
  group_analysis/qwen35_head_attention_bestL_describe.csv
    Columns: task, layer, head, mean_r, sem_r  (rows = all heads at the best layer)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from scipy.stats import rankdata as scipy_rankdata
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_images', type=int, default=None,
                    help='Limit to first N images per task (for quick testing)')
args = parser.parse_args()

GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
PROBING_DIR = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
TARGET_CSV  = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')

HM_DIRS = {
    'social':   os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social'),
    'describe': os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive'),
}

TASK_CFG = {
    'social': {
        'act_dir':   os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35', 'activations'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35', 'image_mask.pt'),
        'out_all':   os.path.join(GROUP_DIR, 'qwen35_head_attention_allL_social.csv'),
        'out_best':  os.path.join(GROUP_DIR, 'qwen35_head_attention_bestL_social.csv'),
    },
    'describe': {
        'act_dir':   os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35', 'activations'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35', 'image_mask.pt'),
        'out_all':   os.path.join(GROUP_DIR, 'qwen35_head_attention_allL_describe.csv'),
        'out_best':  os.path.join(GROUP_DIR, 'qwen35_head_attention_bestL_describe.csv'),
    },
}

FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
N_HEADS  = 16
OUT_SIZE = 256

target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_imgs = [x.replace('.jpg', '') for x in target_imgs]
print(f'Target images: {len(target_imgs)}')


def extract_per_head_maps(act_path, base_mask):
    """Load one .pt → per-head spatial map. Returns (N_HEADS, OUT_SIZE, OUT_SIZE)."""
    data = torch.load(act_path, map_location='cpu').squeeze(0)
    avg  = data.float().nanmean(dim=1)   # (N_HEADS, seq_len)
    seq_len = avg.shape[-1]

    pad_len = seq_len - base_mask.shape[0]
    if pad_len > 0:
        mask = F.pad(base_mask, (0, pad_len), value=False)
    else:
        mask = base_mask[:seq_len]

    masked = avg[:, mask]
    masked = masked[:, 1:-1]
    n_tok  = masked.shape[1]
    side   = int(np.sqrt(n_tok))
    if side * side != n_tok:
        masked = masked[:, :side * side]

    maps_grid = masked.reshape(N_HEADS, side, side).numpy()

    out = np.zeros((N_HEADS, OUT_SIZE, OUT_SIZE), dtype=np.float32)
    for h in range(N_HEADS):
        out[h] = cv2.resize(maps_grid[h], (OUT_SIZE, OUT_SIZE),
                            interpolation=cv2.INTER_LINEAR)
    return out


def load_and_prerank_human_heatmaps(task, valid_imgs):
    hm_dir   = HM_DIRS[task]
    sub_files = sorted(f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy'))
    n_subs   = len(sub_files)
    n_imgs   = len(valid_imgs)
    flat     = OUT_SIZE * OUT_SIZE

    hum_ranked = np.zeros((n_subs, n_imgs, flat), dtype=np.float32)
    hum_valid  = np.zeros((n_subs, n_imgs), dtype=bool)

    for si, fname in enumerate(sub_files):
        sub_data = np.load(os.path.join(hm_dir, fname), allow_pickle=True).item()
        rows_raw, row_valid = [], []
        for img_name in valid_imgs:
            hum = sub_data.get(img_name + '.jpg', sub_data.get(img_name, None))
            if hum is not None:
                hum_clean   = np.nan_to_num(hum, nan=0.0).astype(np.float32)
                hum_resized = np.array(
                    Image.fromarray(hum_clean).resize((OUT_SIZE, OUT_SIZE),
                                                      resample=Image.NEAREST),
                    dtype=np.float32).flatten()
                rows_raw.append(hum_resized)
                row_valid.append(True)
            else:
                rows_raw.append(np.zeros(flat, dtype=np.float32))
                row_valid.append(False)
        hum_valid[si] = row_valid
        valid_i = np.where(row_valid)[0]
        if len(valid_i):
            mat = np.stack([rows_raw[i] for i in valid_i])
            hum_ranked[si, valid_i] = scipy_rankdata(
                mat, method='average', axis=1).astype(np.float32)

    return hum_ranked, hum_valid


def spearman_r_flat(x_ranked, y_ranked):
    xm  = x_ranked - x_ranked.mean()
    ym  = y_ranked - y_ranked.mean()
    num = (xm * ym).sum()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum()) + 1e-10
    return float(num / denom)


def compute_head_correlations(head_maps, hum_ranked, hum_valid, valid_imgs):
    n_subs = hum_ranked.shape[0]
    model_ids = set(head_maps.keys())
    head_sub_r = np.full((N_HEADS, n_subs), np.nan)

    for si in range(n_subs):
        for head in range(N_HEADS):
            rs = []
            for ii, img_name in enumerate(valid_imgs):
                if not hum_valid[si, ii] or img_name not in model_ids:
                    continue
                model_map    = head_maps[img_name][head].flatten()
                model_ranked = scipy_rankdata(model_map, method='average').astype(np.float32)
                r = spearman_r_flat(model_ranked, hum_ranked[si, ii])
                rs.append(r)
            if rs:
                head_sub_r[head, si] = np.mean(rs)

    mean_r  = np.nanmean(head_sub_r, axis=1)
    n_valid = np.sum(~np.isnan(head_sub_r), axis=1)
    sem_r   = np.nanstd(head_sub_r, axis=1) / np.sqrt(np.maximum(n_valid, 1))
    return mean_r, sem_r


for task, cfg in TASK_CFG.items():
    print(f'\n{"=" * 60}')
    print(f'Task: {task}')
    print(f'{"=" * 60}')

    act_dir  = cfg['act_dir']
    out_all  = cfg['out_all']
    out_best = cfg['out_best']

    base_mask = torch.load(cfg['mask_path'], map_location='cpu')
    base_mask = base_mask.squeeze(0) if base_mask.dim() == 2 else base_mask
    print(f'Image mask: {base_mask.shape}, n_image_tokens={base_mask.sum().item()}')

    # Find valid images using first full-attn layer dir
    layer_0_dir = os.path.join(act_dir, f'attn-layer-{FULL_ATTN_LAYERS[0]}')
    available   = {f.replace('.pt', '') for f in os.listdir(layer_0_dir) if f.endswith('.pt')}
    valid_imgs  = [img for img in target_imgs if img in available]
    if args.n_images is not None:
        valid_imgs = valid_imgs[:args.n_images]
        print(f'Testing with first {args.n_images} images')
    print(f'Images with activations: {len(valid_imgs)} / {len(target_imgs)}')

    print('Pre-ranking human heatmaps ...')
    hum_ranked, hum_valid = load_and_prerank_human_heatmaps(task, valid_imgs)
    print(f'Subjects: {hum_ranked.shape[0]}')

    all_records = []
    best_mean_r_global = -np.inf
    best_layer_global  = None
    best_head_global   = None

    for layer in FULL_ATTN_LAYERS:
        layer_dir = os.path.join(act_dir, f'attn-layer-{layer}')
        print(f'  Layer {layer:2d} ...', end='', flush=True)

        head_maps = {}
        for img_name in valid_imgs:
            act_path = os.path.join(layer_dir, f'{img_name}.pt')
            if not os.path.exists(act_path):
                continue
            maps = extract_per_head_maps(act_path, base_mask)
            head_maps[img_name] = maps

        if not head_maps:
            print(' [no data]')
            continue

        mean_r, sem_r = compute_head_correlations(head_maps, hum_ranked, hum_valid, valid_imgs)

        best_head_this_layer = int(np.argmax(mean_r))
        print(f' best head={best_head_this_layer + 1} r={mean_r[best_head_this_layer]:.4f}')

        if mean_r[best_head_this_layer] > best_mean_r_global:
            best_mean_r_global = mean_r[best_head_this_layer]
            best_layer_global  = layer
            best_head_global   = best_head_this_layer

        for h in range(N_HEADS):
            all_records.append({
                'task': task, 'layer': layer, 'head': h + 1,
                'mean_r': mean_r[h], 'sem_r': sem_r[h],
            })

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(out_all, index=False)
    print(f'\nSaved all-layer CSV: {out_all}')

    df_best = df_all[df_all['layer'] == best_layer_global].copy()
    df_best.to_csv(out_best, index=False)
    print(f'Best layer: {best_layer_global}, best head: {best_head_global + 1} '
          f'(mean_r={best_mean_r_global:.4f})')
    print(f'Saved best-layer CSV: {out_best}')
    print(df_best.sort_values('mean_r', ascending=False).head(5).to_string(index=False))
