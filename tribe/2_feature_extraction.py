"""
2_feature_extraction.py
-----------------------
Extract per-model, per-task spatial attention maps for all 200 target images.
Output: one (200, 196) array per model × task  →  12 files total.

All maps are normalized to 14×14 = 196 for consistent feature dimensionality.

Best heads (from existing alignment analysis):
  CNN-LSTM:       soft attention (single), describe + social
  ViT-LSTM:       soft attention (single), describe + social
  CNN-Transformer: L3 H5 (0-based H4)
  ViT-Transformer: L3 H8 (0-based H7)
  Molmo:          L9 H24/social (0-based H23), L9 H27/describe (0-based H26)
  Qwen3.5:        L15 H12 (0-based H11), both tasks

Run on CPU.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

BASE       = Path('/scratch/gpfs/GRAZIANO/CANS')
ARCH_DIR   = BASE / 'architecture_manipulation'
MOLMO_DIR  = BASE / 'molmo/molmo'
PROBE_DIR  = BASE / 'vlm-probing'
RNN_DIR    = BASE / 'rnn_model'
OUT_DIR    = BASE / 'tribe/outputs'
TARGET_CSV = RNN_DIR / 'results/target_imgs.csv'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Target image list ──────────────────────────────────────────────────────────
target_imgs_raw = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_imgs     = [x.replace('.jpg', '') for x in target_imgs_raw]   # strip .jpg
N = len(target_imgs)
print(f'Target images: {N}')

OUT_SIZE   = 14   # all maps → 14×14 = 196
FLAT       = OUT_SIZE * OUT_SIZE


def resize_to_14(arr2d):
    return cv2.resize(arr2d.astype(np.float32), (OUT_SIZE, OUT_SIZE),
                      interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════════════════
# CNN-LSTM  (soft Bahdanau attention, already 14×14)
# Source: rnn_model/results/h_c_alpha_n200_{task}_final.npz
# ══════════════════════════════════════════════════════════════════════════════

def extract_cnn_lstm(task):
    fname = RNN_DIR / f'results/h_c_alpha_n200_{task}_final.npz'
    d     = np.load(fname, allow_pickle=True)
    alphas = d['alphas'].item()   # dict: "000000420840.jpg" → (14,14)

    X = np.zeros((N, FLAT), dtype=np.float32)
    for i, img_id in enumerate(target_imgs):
        key = img_id + '.jpg'
        if key in alphas:
            a = np.array(alphas[key], dtype=np.float32)
            if a.shape != (OUT_SIZE, OUT_SIZE):
                a = resize_to_14(a.reshape(int(np.sqrt(a.size)), -1))
            X[i] = a.flatten()
        else:
            print(f'  [CNN-LSTM] missing: {key}')
    return X


# ══════════════════════════════════════════════════════════════════════════════
# ViT-LSTM  (soft Bahdanau attention, stored as 196-dim flat vector)
# Source: architecture_manipulation/ViT_LSTM/ckpt_cache/{phase}_beam5_E{max}.npz
# ══════════════════════════════════════════════════════════════════════════════

def _vit_lstm_phase_file(phase):
    cache = ARCH_DIR / 'ViT_LSTM/ckpt_cache'
    files = sorted(f for f in os.listdir(cache)
                   if f.startswith(phase) and f.endswith('.npz') and 'head_vecs' not in f)
    if not files:
        raise FileNotFoundError(f'No ViT-LSTM files for phase {phase}')
    return cache / files[-1]   # highest epoch = last alphabetically (E000, E001, ...)


def extract_vit_lstm(task):
    phase = 'phase2' if task == 'describe' else 'social_unfrozen'
    fpath = _vit_lstm_phase_file(phase)
    print(f'  ViT-LSTM {task}: {fpath.name}')
    d      = np.load(fpath, allow_pickle=True)
    alphas = d['alphas'].item()   # dict: "000000420840.jpg" → (196,) flat

    X = np.zeros((N, FLAT), dtype=np.float32)
    for i, img_id in enumerate(target_imgs):
        key = img_id + '.jpg'
        if key in alphas:
            a = np.array(alphas[key], dtype=np.float32).flatten()
            if len(a) == FLAT:
                X[i] = a
            else:
                side = int(np.sqrt(len(a)))
                X[i] = resize_to_14(a.reshape(side, side)).flatten()
        else:
            print(f'  [ViT-LSTM] missing: {key}')
    return X


# ══════════════════════════════════════════════════════════════════════════════
# CNN-Transformer / ViT-Transformer
# Source: architecture_manipulation/{model}/ckpt_cache/{phase}_beam5_E{N}_head_vecs_L3.npz
# head_vecs shape: (8 heads, 200 images, 65536)  where 65536 = 256×256
# ══════════════════════════════════════════════════════════════════════════════

TRANS_CFG = {
    'CNN_Transformer': {
        'describe': ('phase2',          14, 4),   # (phase, epoch, head_0based)
        'social':   ('social_unfrozen',  3, 4),
    },
    'ViT_Transformer': {
        'describe': ('phase2',          11, 7),
        'social':   ('social_unfrozen',  1, 7),
    },
}


def extract_transformer(model_key, task):
    phase, epoch, head_idx = TRANS_CFG[model_key][task]
    fname = (ARCH_DIR / model_key / 'ckpt_cache' /
             f'{phase}_beam5_E{epoch:03d}_head_vecs_L3.npz')
    if not fname.exists():
        # try without zero-padding (e.g. E3 vs E003)
        fname = (ARCH_DIR / model_key / 'ckpt_cache' /
                 f'{phase}_beam5_E{epoch}_head_vecs_L3.npz')
    print(f'  {model_key} {task}: {fname.name}')
    d    = np.load(fname, allow_pickle=True)
    hvec = d['head_vecs']          # (8, 200, 65536)
    cidx = d['common_idx']         # (200,) indices into target_imgs

    # hvec[head_idx, i, :] → attention for target_imgs[cidx[i]]
    maps_256 = hvec[head_idx]      # (200, 65536)
    X        = np.zeros((N, FLAT), dtype=np.float32)
    for pos, img_idx in enumerate(cidx):
        m = maps_256[pos].reshape(256, 256).astype(np.float32)
        X[int(img_idx)] = resize_to_14(m).flatten()
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Molmo 7B-D
# Source: molmo/molmo/activations/{task}/activations_img_{id}.pt
# Shape: (28 layers, 28 heads, n_tokens, 1028)
# Uses image_input_idx.pt for spatial reconstruction (same for all images/task)
# ══════════════════════════════════════════════════════════════════════════════

MOLMO_HEAD = {'describe': 26, 'social': 23}   # 0-based (H27/H24 1-based)
MOLMO_LAYER = 9

NUM_CROPS_X = 3
NUM_CROPS_Y = 3


def _setup_molmo_grid(act_dir):
    idx_path      = act_dir / 'image_input_idx.pt'
    img_input_idx = torch.load(idx_path, map_location='cpu')[0].numpy()
    # img_input_idx[0] = global crop; [1:] = 3×3 local crops, each 12×12 tokens
    img_idx_all   = (img_input_idx[1:]
                     .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                     .transpose(0, 2, 1, 3)
                     .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    valid_idx     = np.where(img_idx_all > 0)
    return img_idx_all, valid_idx


def _extract_molmo_map(act_path, layer_idx, head_idx, img_idx_all, valid_idx):
    attn = torch.load(act_path, map_location='cpu').numpy()  # (28,28,n_tok,1028)
    n_tokens = attn.shape[2]
    H, W     = img_idx_all.shape
    vol      = np.zeros((n_tokens, H, W), dtype=np.float32)

    img_positions = img_idx_all[valid_idx]
    for tok in range(n_tokens):
        vol[tok, valid_idx[0], valid_idx[1]] = attn[layer_idx, head_idx, tok, img_positions]

    head_map = vol.sum(axis=0)   # (H, W) = (36, 36)

    row_valid = np.any(img_idx_all > 0, axis=1)
    col_valid = np.any(img_idx_all > 0, axis=0)
    cropped   = head_map[row_valid][:, col_valid]
    return resize_to_14(cropped)


def extract_molmo(task):
    act_dir   = MOLMO_DIR / 'activations' / task
    head_idx  = MOLMO_HEAD[task]
    img_idx_all, valid_idx = _setup_molmo_grid(act_dir)

    X = np.zeros((N, FLAT), dtype=np.float32)
    for i, img_id in enumerate(tqdm(target_imgs, desc=f'Molmo {task}')):
        act_path = act_dir / f'activations_img_{img_id}.pt'
        if not act_path.exists():
            print(f'  [Molmo] missing: {act_path.name}')
            continue
        m       = _extract_molmo_map(act_path, MOLMO_LAYER, head_idx, img_idx_all, valid_idx)
        X[i]    = m.flatten()
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Qwen3.5 9B
# Source: vlm-probing/output/{task_dir}/qwen35/activations/attn-layer-15/{id}.pt
# Shape: (1, 16 heads, n_gen_steps, seq_len)
# Best head: L15 H12 (0-based H11), same for both tasks
# ══════════════════════════════════════════════════════════════════════════════

QWEN_LAYER = 15
QWEN_HEAD  = 11   # 0-based (H12 in 1-based CSVs)
TASK_DIR   = {'describe': 'coco', 'social': 'social_cap'}


def extract_qwen35(task):
    task_key = TASK_DIR[task]
    act_base = PROBE_DIR / 'output' / task_key / 'qwen35' / 'activations' / f'attn-layer-{QWEN_LAYER}'
    mask_path = PROBE_DIR / 'output' / task_key / 'qwen35' / 'image_mask.pt'

    base_mask = torch.load(mask_path, map_location='cpu')
    if base_mask.dim() == 2:
        base_mask = base_mask.squeeze(0)

    X = np.zeros((N, FLAT), dtype=np.float32)
    for i, img_id in enumerate(tqdm(target_imgs, desc=f'Qwen3.5 {task}')):
        act_path = act_base / f'{img_id}.pt'
        if not act_path.exists():
            print(f'  [Qwen3.5] missing: {act_path.name}')
            continue

        data    = torch.load(act_path, map_location='cpu').squeeze(0)  # (16, n_gen, seq_len)
        avg     = data[QWEN_HEAD].float().nanmean(dim=0)                 # mean over gen steps → (seq_len,)
        seq_len = avg.shape[0]

        import torch.nn.functional as F
        pad_len = seq_len - base_mask.shape[0]
        mask    = F.pad(base_mask, (0, pad_len), value=False) if pad_len > 0 else base_mask[:seq_len]

        masked  = avg[mask][1:-1]          # strip [CLS]-like boundary tokens
        n_tok   = masked.shape[0]
        side    = int(np.sqrt(n_tok))
        if side * side != n_tok:
            masked = masked[:side * side]  # trim to square

        grid    = masked.numpy().reshape(side, side)
        X[i]    = resize_to_14(grid).flatten()
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Run all extractions
# ══════════════════════════════════════════════════════════════════════════════

configs = [
    ('CNN-LSTM',        'describe', extract_cnn_lstm,                ('describe',)),
    ('CNN-LSTM',        'social',   extract_cnn_lstm,                ('social',)),
    ('ViT-LSTM',        'describe', extract_vit_lstm,                ('describe',)),
    ('ViT-LSTM',        'social',   extract_vit_lstm,                ('social',)),
    ('CNN-Transformer', 'describe', extract_transformer,             ('CNN_Transformer', 'describe')),
    ('CNN-Transformer', 'social',   extract_transformer,             ('CNN_Transformer', 'social')),
    ('ViT-Transformer', 'describe', extract_transformer,             ('ViT_Transformer', 'describe')),
    ('ViT-Transformer', 'social',   extract_transformer,             ('ViT_Transformer', 'social')),
    ('Molmo',           'describe', extract_molmo,                   ('describe',)),
    ('Molmo',           'social',   extract_molmo,                   ('social',)),
    ('Qwen3.5',         'describe', extract_qwen35,                  ('describe',)),
    ('Qwen3.5',         'social',   extract_qwen35,                  ('social',)),
]

for model_label, task, fn, args in configs:
    out_path = OUT_DIR / f'X_attn_{model_label}_{task}.npy'
    if out_path.exists():
        print(f'Skipping (exists): {out_path.name}')
        continue
    print(f'\n{"="*60}')
    print(f'Extracting: {model_label} / {task}')
    X = fn(*args)
    np.save(str(out_path), X)
    print(f'Saved: {out_path.name}  shape={X.shape}  non-zero rows: {(X.sum(axis=1) != 0).sum()}')

print('\nFeature extraction complete.')
