"""
plot_fig4_taskdiff_arch.py
---------------------------
Social vs. Describe attention decomposition for all models in fig3_revised.
7 columns: Human(Mean), CNN-LSTM, CNN-Transformer(H5), ViT-LSTM,
           ViT-Transformer(H1), Molmo(L9H24), Qwen(L6H8)
3 rows:    same images as fig2 (000000157601, 000000232649, 000000535306)

RGB decomposition per cell:
  R = social-unique  (social > describe, re-normalized)
  G = shared         (min(social, describe), re-normalized)
  B = describe-unique (describe > social, re-normalized)

For Molmo and Qwen: same head used for both tasks (social head).
  Molmo: L9 H24 — loaded from social and describe activation directories
  Qwen:  L6 H8  — loaded from social_cap and coco activation directories (layer 6 for both)

Output: architecture_manipulation/manuscript_plots/fig4_taskdiff_arch.png
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ARCH_DIR    = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
MOLMO_DIR   = '/scratch/gpfs/GRAZIANO/CANS/molmo/molmo'
PROBING_DIR = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
IMG_DIR     = os.path.join(GROUP_DIR, 'imgs')
HUM_SOC_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_DES_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
OUT_DIR     = os.path.join(ARCH_DIR, 'manuscript_plots')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')

# ── Constants ─────────────────────────────────────────────────────────────────
BLEND_IMGS   = ['000000157601', '000000232649', '000000535306']
SIZE         = 256
ALPHA        = 0.65
NUM_CROPS_X  = NUM_CROPS_Y = 3

MOLMO_LAYER, MOLMO_HEAD = 9, 24   # 1-indexed; same head for both tasks
QWEN_LAYER,  QWEN_HEAD  = 6, 8   # 1-indexed; layer 6 for both social and describe

BEST_HEAD_TRANS = {'CNN_Transformer': 5, 'ViT_Transformer': 1}   # 1-indexed
LAST_EPOCH = {
    'CNN_Transformer': {'social': 3,  'describe': 14},
    'ViT_Transformer': {'social': 1,  'describe': 11},
    'CNN_LSTM':        {'social': 2,  'describe': None},  # cans_beam5.npz for describe
    'ViT_LSTM':        {'social': 1,  'describe': 3},
}

# ── Column definitions ────────────────────────────────────────────────────────
COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer',
        'Molmo', 'Qwen']
COL_COLORS = {
    'Human':           '#2C3E50',
    'CNN-LSTM':        '#5D6D7E',
    'CNN-Transformer': '#E74C3C',
    'ViT-LSTM':        '#27AE60',
    'ViT-Transformer': '#8E44AD',
    'Molmo':           '#F39C12',
    'Qwen':            '#1ABC9C',
}
COL_HEADERS = {
    'Human':           'Human (Mean)',
    'CNN-LSTM':        'CNN-LSTM',
    'CNN-Transformer': 'CNN-Transformer (L3H5)',
    'ViT-LSTM':        'ViT-LSTM',
    'ViT-Transformer': 'ViT-Transformer (L3H1)',
    'Molmo':           f'Molmo 7B-D\n(L{MOLMO_LAYER}H{MOLMO_HEAD})',
    'Qwen':            f'Qwen2-VL 7B\n(L{QWEN_LAYER}H{QWEN_HEAD})',
}


# ── Utility ───────────────────────────────────────────────────────────────────
def normalize(x):
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


ATTN_THRESHOLD = 0.1

def rgb_decompose(soc, des):
    soc = normalize(soc)
    des = normalize(des)
    soc_bin = soc > ATTN_THRESHOLD
    des_bin = des > ATTN_THRESHOLD
    r = (soc_bin & ~des_bin).astype(np.float32)
    g = (soc_bin &  des_bin).astype(np.float32)
    b = (~soc_bin &  des_bin).astype(np.float32)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ── Load target image list ────────────────────────────────────────────────────
target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_idx  = {im: i for i, im in enumerate(target_imgs)}


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN: mean across subjects
# ══════════════════════════════════════════════════════════════════════════════
def _load_mean_human(hm_dir, img_id):
    maps = []
    for fname in sorted(os.listdir(hm_dir)):
        if not fname.endswith('_heatmaps.npy'):
            continue
        sdat = np.load(os.path.join(hm_dir, fname), allow_pickle=True).item()
        key  = f'{img_id}.jpg'
        if key in sdat:
            maps.append(np.nan_to_num(sdat[key], nan=0.0).astype(np.float32))
    if not maps:
        return None
    mean = np.mean(maps, axis=0)
    resized = np.array(Image.fromarray(mean).resize((SIZE, SIZE), resample=Image.NEAREST))
    return normalize(resized)

print('Pre-loading human gaze maps...')
human_soc = {img: _load_mean_human(HUM_SOC_DIR, img) for img in BLEND_IMGS}
human_des = {img: _load_mean_human(HUM_DES_DIR, img) for img in BLEND_IMGS}
print('  done')


# ══════════════════════════════════════════════════════════════════════════════
# RNN LSTM MODELS (CNN-LSTM, ViT-LSTM): alphas dict, 14×14 grid
# ══════════════════════════════════════════════════════════════════════════════
def alpha_to_map(alpha_vec):
    arr = np.array(alpha_vec).flatten()
    side = int(np.sqrt(len(arr)))
    resized = np.array(
        Image.fromarray(arr.reshape(side, side).astype(np.float32))
        .resize((SIZE, SIZE), resample=Image.BILINEAR))
    return normalize(resized)


REPL_DIR = os.path.join(GROUP_DIR, 'learning_to_attend', 'social_replication')

LSTM_NPZ = {
    'CNN-LSTM': {
        'social':   os.path.join(REPL_DIR, 'final_replicated_model', 'ckpt_cache',
                                 'soc_bs32_beam5_E2.npz'),
        'describe': os.path.join(REPL_DIR, 'ckpt_cache', 'P2_E14_beam5.npz'),
    },
    'ViT-LSTM': {
        'social':   os.path.join(ARCH_DIR, 'ViT_LSTM', 'ckpt_cache',
                                 'social_unfrozen_beam5_E001.npz'),
        'describe': os.path.join(ARCH_DIR, 'ViT_LSTM', 'ckpt_cache',
                                 'phase2_beam5_E003.npz'),
    },
}

print('Loading LSTM attention maps...')
lstm_alphas = {}
for model, paths in LSTM_NPZ.items():
    lstm_alphas[model] = {}
    for task, path in paths.items():
        raw = np.load(path, allow_pickle=True)
        lstm_alphas[model][task] = raw['alphas'].item()
print('  done')


def get_lstm_map(model, task, img_id):
    alphas = lstm_alphas[model][task]
    key = f'{img_id}.jpg' if f'{img_id}.jpg' in alphas else img_id
    if key not in alphas:
        return None
    return alpha_to_map(alphas[key])


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER MODELS (CNN-Transformer H5, ViT-Transformer H1): head_vecs
# ══════════════════════════════════════════════════════════════════════════════
TRANS_HEAD_VEC_NPZ = {
    'CNN-Transformer': {
        'social':   os.path.join(ARCH_DIR, 'CNN_Transformer', 'ckpt_cache',
                                 'social_unfrozen_beam5_E003_head_vecs_L3.npz'),
        'describe': os.path.join(ARCH_DIR, 'CNN_Transformer', 'ckpt_cache',
                                 'phase2_beam5_E014_head_vecs_L3.npz'),
    },
    'ViT-Transformer': {
        'social':   os.path.join(ARCH_DIR, 'ViT_Transformer', 'ckpt_cache',
                                 'social_unfrozen_beam5_E001_head_vecs_L3.npz'),
        'describe': os.path.join(ARCH_DIR, 'ViT_Transformer', 'ckpt_cache',
                                 'phase2_beam5_E011_head_vecs_L3.npz'),
    },
}

print('Loading Transformer head_vecs...')
trans_data = {}
for model, paths in TRANS_HEAD_VEC_NPZ.items():
    trans_data[model] = {}
    cond_key = model.replace('-', '_')
    head_idx = BEST_HEAD_TRANS[cond_key] - 1   # 0-indexed
    for task, path in paths.items():
        d = np.load(path, allow_pickle=True)
        trans_data[model][task] = {
            'head_vecs':  d['head_vecs'][head_idx],   # (n_common, 65536)
            'common_idx': d['common_idx'].astype(int).tolist(),
        }
print('  done')


def get_trans_map(model, task, img_id):
    td      = trans_data[model][task]
    cidx    = td['common_idx']
    hvecs   = td['head_vecs']
    ti = target_idx.get(f'{img_id}.jpg', target_idx.get(img_id, -1))
    if ti < 0 or ti not in cidx:
        return None
    ii  = cidx.index(ti)
    arr = hvecs[ii].reshape(SIZE, SIZE).astype(np.float32)
    return normalize(arr)


# ══════════════════════════════════════════════════════════════════════════════
# MOLMO: L9H24 for both tasks
# ══════════════════════════════════════════════════════════════════════════════
def _setup_molmo_grid(act_dir):
    idx = torch.load(os.path.join(act_dir, 'image_input_idx.pt'), map_location='cpu')[0].numpy()
    img_idx_all = (idx[1:]
                   .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                   .transpose(0, 2, 1, 3)
                   .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    valid_idx   = np.where(img_idx_all > 0)
    img_pos     = img_idx_all[valid_idx]
    full_h      = NUM_CROPS_X * 336
    full_w      = NUM_CROPS_Y * 336
    idx_resized = cv2.resize(img_idx_all, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    row_valid   = np.any(idx_resized > 0, axis=1)
    col_valid   = np.any(idx_resized > 0, axis=0)
    return img_idx_all, valid_idx, img_pos, full_h, full_w, row_valid, col_valid


print('Setting up Molmo grids...')
_molmo_grids = {}
for task in ('social', 'describe'):
    act_dir = os.path.join(MOLMO_DIR, 'activations', task)
    _molmo_grids[task] = (act_dir,) + _setup_molmo_grid(act_dir)
print('  done')


def get_molmo_map(task, img_id):
    act_dir, img_idx_all, valid_idx, img_pos, full_h, full_w, row_valid, col_valid = \
        _molmo_grids[task]
    act_path = os.path.join(act_dir, f'activations_img_{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    attn      = torch.load(act_path, map_location='cpu').numpy()
    n_tokens  = attn.shape[2]
    head_idx  = MOLMO_HEAD - 1
    layer_idx = MOLMO_LAYER
    H, W = img_idx_all.shape
    vol  = np.zeros((n_tokens, H, W), dtype=np.float32)
    for tok in range(n_tokens):
        vol[tok, valid_idx[0], valid_idx[1]] = attn[layer_idx, head_idx, tok, img_pos]
    grid    = vol.sum(axis=0)
    grid    = cv2.GaussianBlur(grid, (0, 0), sigmaX=1.5, sigmaY=1.5)
    resized = cv2.resize(grid, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    cropped = resized[row_valid][:, col_valid]
    out     = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return normalize(out)


# ══════════════════════════════════════════════════════════════════════════════
# QWEN: L6H8 for both tasks (layer 6 exists for both social_cap and coco)
# ══════════════════════════════════════════════════════════════════════════════
_qwen_cfg = {
    'social': {
        'act_dir':  os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen',
                                 'activations', f'attn-layer-{QWEN_LAYER}'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'social_cap', 'image_mask.pt'),
    },
    'describe': {
        'act_dir':  os.path.join(PROBING_DIR, 'output', 'coco', 'qwen',
                                 'activations', f'attn-layer-{QWEN_LAYER}'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'coco', 'image_mask.pt'),
    },
}

print('Setting up Qwen masks...')
_qwen_masks = {}
for task, cfg in _qwen_cfg.items():
    raw = torch.load(cfg['mask_path'], map_location='cpu')
    _qwen_masks[task] = raw.squeeze(0) if raw.dim() == 2 else raw
print('  done')


def get_qwen_map(task, img_id):
    act_dir  = _qwen_cfg[task]['act_dir']
    act_path = os.path.join(act_dir, f'{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    data     = torch.load(act_path, map_location='cpu').squeeze(0).float()
    avg      = data.mean(dim=1)   # (N_HEADS, seq_len)
    base_mask = _qwen_masks[task]
    seq_len   = avg.shape[-1]
    pad_len   = seq_len - base_mask.shape[0]
    mask      = (F.pad(base_mask, (0, pad_len), value=False)
                 if pad_len > 0 else base_mask[:seq_len])
    head_idx  = QWEN_HEAD - 1
    masked    = avg[head_idx][mask]
    masked    = masked[1:-1]   # remove CLS/SEP
    side      = int(np.sqrt(masked.shape[0]))
    grid      = masked[:side*side].reshape(side, side).numpy()
    out       = cv2.resize(grid, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return normalize(out)


# ══════════════════════════════════════════════════════════════════════════════
# Master dispatch
# ══════════════════════════════════════════════════════════════════════════════
def get_map(col, task, img_id):
    if col == 'Human':
        return human_soc[img_id] if task == 'social' else human_des[img_id]
    if col in ('CNN-LSTM', 'ViT-LSTM'):
        return get_lstm_map(col, task, img_id)
    if col in ('CNN-Transformer', 'ViT-Transformer'):
        return get_trans_map(col, task, img_id)
    if col == 'Molmo':
        return get_molmo_map(task, img_id)
    if col == 'Qwen':
        return get_qwen_map(task, img_id)
    return None


# ── Plot ──────────────────────────────────────────────────────────────────────
print('\nPlotting...')
n_rows = len(BLEND_IMGS)
n_cols = len(COLS)

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(3.5 * n_cols, 4.0 * n_rows),
                          squeeze=False, dpi=150)

for j, col in enumerate(COLS):
    axes[0, j].set_title(COL_HEADERS[col], fontsize=12, fontweight='bold',
                         pad=10, color=COL_COLORS[col])

for row, img_id in enumerate(BLEND_IMGS):
    img   = np.array(Image.open(os.path.join(IMG_DIR, f'{img_id}.jpg'))
                     .convert('RGB').resize((SIZE, SIZE)))
    img_f = img.astype(np.float32) / 255.0
    print(f'  [{row+1}/{n_rows}] {img_id}')

    for col_idx, col in enumerate(COLS):
        ax      = axes[row, col_idx]
        soc_map = get_map(col, 'social',   img_id)
        des_map = get_map(col, 'describe', img_id)

        if soc_map is None or des_map is None:
            ax.imshow(img)
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='red')
            ax.axis('off')
            continue

        rgb    = rgb_decompose(soc_map, des_map)
        active = rgb.sum(axis=-1, keepdims=True) > 0
        a      = active * ALPHA
        blended = np.clip(img_f * (1 - a) + rgb * a, 0, 1)
        ax.imshow(blended)
        ax.axis('off')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=(0.9, 0.1, 0.1), label='Social-unique'),
    mpatches.Patch(color=(0.1, 0.8, 0.1), label='Shared'),
    mpatches.Patch(color=(0.1, 0.1, 0.9), label='Describe-unique'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=3,
           fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.02),
           edgecolor='#cccccc')

plt.suptitle('Social vs. Describe Attention Decomposition',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'fig4_taskdiff_arch_binary_t01.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.close(fig)
