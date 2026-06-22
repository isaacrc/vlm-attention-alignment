"""
plot_fig5_qwen35.py
--------------------
Raw attention maps for all 7 models × 3 images, split by task.
Replaces Qwen2-VL with Qwen3.5-9B.

Top half  (rows 1-3): Social task
Bottom half (rows 4-6): Describe task

Columns: Human(Mean), CNN-LSTM, CNN-Transformer(L3H5), ViT-LSTM,
         ViT-Transformer(L3H8), Molmo(L9H24), Qwen3.5(best social head)

Output: architecture_manipulation/manuscript_plots/fig5_qwen35_attn_by_task.png
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ARCH_DIR    = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
LTA_DIR     = os.path.join(GROUP_DIR, 'learning_to_attend')
REPL_DIR    = os.path.join(LTA_DIR, 'social_replication')
MOLMO_DIR   = '/scratch/gpfs/GRAZIANO/CANS/molmo/molmo'
PROBING_DIR = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
RNN_DIR     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model'
IMG_DIR     = os.path.join(GROUP_DIR, 'imgs')
HUM_SOC_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_DES_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
OUT_DIR     = os.path.join(ARCH_DIR, 'manuscript_plots')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_CSV = os.path.join(RNN_DIR, 'results', 'target_imgs.csv')

BLEND_IMGS  = ['000000157601', '000000232649', '000000535306']
SIZE        = 256
ALPHA       = 0.6
CMAP        = cm.hot
NUM_CROPS_X = NUM_CROPS_Y = 3

MOLMO_LAYER, MOLMO_HEAD = 9, 24
BEST_HEAD_TRANS = {'CNN_Transformer': 5, 'ViT_Transformer': 8}

# Qwen3.5: load best social head from analysis CSV
_qwen35_soc_csv = os.path.join(GROUP_DIR, 'qwen35_head_attention_allL_social.csv')
_q35_soc_df = pd.read_csv(_qwen35_soc_csv)
_q35_best   = _q35_soc_df.loc[_q35_soc_df['mean_r'].idxmax()]
QWEN35_LAYER = int(_q35_best['layer'])
QWEN35_HEAD  = int(_q35_best['head'])
print(f'Qwen3.5 best social head: L{QWEN35_LAYER}H{QWEN35_HEAD}')

COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer',
        'Molmo', 'Qwen3.5']
COL_COLORS = {
    'Human':           '#2C3E50',
    'CNN-LSTM':        '#5D6D7E',
    'CNN-Transformer': '#E74C3C',
    'ViT-LSTM':        '#27AE60',
    'ViT-Transformer': '#8E44AD',
    'Molmo':           '#F39C12',
    'Qwen3.5':         '#1ABC9C',
}
COL_HEADERS = {
    'Human':           'Human\n(Mean)',
    'CNN-LSTM':        'CNN\nLSTM',
    'CNN-Transformer': 'CNN\nTransformer\n(L3H5)',
    'ViT-LSTM':        'ViT\nLSTM',
    'ViT-Transformer': 'ViT\nTransformer\n(L3H8)',
    'Molmo':           f'Molmo\n7B-D\n(L{MOLMO_LAYER}H{MOLMO_HEAD})',
    'Qwen3.5':         f'Qwen\n9B-3.5\n(L{QWEN35_LAYER}H{QWEN35_HEAD})',
}


def normalize(x):
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def attn_overlay(img_f, attn_map):
    heat = CMAP(attn_map)[:, :, :3].astype(np.float32)
    return np.clip(img_f * (1 - ALPHA) + heat * ALPHA, 0, 1)


target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_idx  = {im: i for i, im in enumerate(target_imgs)}


def load_mean_human(hm_dir, img_id):
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
    return normalize(np.array(
        Image.fromarray(mean).resize((SIZE, SIZE), resample=Image.NEAREST)))


print('Pre-loading human gaze maps...')
human_maps = {
    'social':   {img: load_mean_human(HUM_SOC_DIR, img) for img in BLEND_IMGS},
    'describe': {img: load_mean_human(HUM_DES_DIR, img) for img in BLEND_IMGS},
}
print('  done')


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


def alpha_to_map(alpha_vec):
    arr  = np.array(alpha_vec).flatten()
    side = int(np.sqrt(len(arr)))
    resized = np.array(
        Image.fromarray(arr.reshape(side, side).astype(np.float32))
        .resize((SIZE, SIZE), resample=Image.BILINEAR))
    return normalize(resized)


print('Loading LSTM maps...')
lstm_alphas = {}
for model, paths in LSTM_NPZ.items():
    lstm_alphas[model] = {}
    for task, path in paths.items():
        lstm_alphas[model][task] = np.load(path, allow_pickle=True)['alphas'].item()
print('  done')


def get_lstm_map(model, task, img_id):
    alphas = lstm_alphas[model][task]
    key = f'{img_id}.jpg' if f'{img_id}.jpg' in alphas else img_id
    return alpha_to_map(alphas[key]) if key in alphas else None


TRANS_NPZ = {
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
for model, paths in TRANS_NPZ.items():
    trans_data[model] = {}
    cond_key = model.replace('-', '_')
    head_idx = BEST_HEAD_TRANS[cond_key] - 1
    for task, path in paths.items():
        d = np.load(path, allow_pickle=True)
        trans_data[model][task] = {
            'head_vecs':  d['head_vecs'][head_idx],
            'common_idx': d['common_idx'].astype(int).tolist(),
        }
print('  done')


def get_trans_map(model, task, img_id):
    td   = trans_data[model][task]
    cidx = td['common_idx']
    hvec = td['head_vecs']
    ti   = target_idx.get(f'{img_id}.jpg', target_idx.get(img_id, -1))
    if ti < 0 or ti not in cidx:
        return None
    return normalize(hvec[cidx.index(ti)].reshape(SIZE, SIZE).astype(np.float32))


def _setup_molmo_grid(act_dir):
    idx = torch.load(os.path.join(act_dir, 'image_input_idx.pt'),
                     map_location='cpu')[0].numpy()
    img_idx_all = (idx[1:]
                   .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                   .transpose(0, 2, 1, 3)
                   .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
    valid_idx   = np.where(img_idx_all > 0)
    img_pos     = img_idx_all[valid_idx]
    full_h      = NUM_CROPS_X * 336
    full_w      = NUM_CROPS_Y * 336
    idx_r       = cv2.resize(img_idx_all, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    row_valid   = np.any(idx_r > 0, axis=1)
    col_valid   = np.any(idx_r > 0, axis=0)
    return act_dir, img_idx_all, valid_idx, img_pos, full_h, full_w, row_valid, col_valid


print('Setting up Molmo grids...')
_molmo_grids = {task: _setup_molmo_grid(
    os.path.join(MOLMO_DIR, 'activations', task)) for task in ('social', 'describe')}
print('  done')


def get_molmo_map(task, img_id):
    act_dir, img_idx_all, valid_idx, img_pos, full_h, full_w, row_valid, col_valid = \
        _molmo_grids[task]
    act_path = os.path.join(act_dir, f'activations_img_{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    attn     = torch.load(act_path, map_location='cpu').numpy()
    n_tokens = attn.shape[2]
    H, W     = img_idx_all.shape
    vol      = np.zeros((n_tokens, H, W), dtype=np.float32)
    for tok in range(n_tokens):
        vol[tok, valid_idx[0], valid_idx[1]] = \
            attn[MOLMO_LAYER, MOLMO_HEAD - 1, tok, img_pos]
    grid    = cv2.GaussianBlur(vol.sum(axis=0), (0, 0), sigmaX=1.5, sigmaY=1.5)
    resized = cv2.resize(grid, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    cropped = resized[row_valid][:, col_valid]
    return normalize(cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))


# ── Qwen3.5 ───────────────────────────────────────────────────────────────────
_qwen35_cfg = {
    'social': {
        'act_dir':   os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35',
                                  'activations', f'attn-layer-{QWEN35_LAYER}'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35', 'image_mask.pt'),
    },
    'describe': {
        'act_dir':   os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35',
                                  'activations', f'attn-layer-{QWEN35_LAYER}'),
        'mask_path': os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35', 'image_mask.pt'),
    },
}

print('Setting up Qwen3.5 masks...')
_qwen35_masks = {}
for task, cfg in _qwen35_cfg.items():
    raw = torch.load(cfg['mask_path'], map_location='cpu')
    _qwen35_masks[task] = raw.squeeze(0) if raw.dim() == 2 else raw
print('  done')


def get_qwen35_map(task, img_id):
    cfg      = _qwen35_cfg[task]
    act_path = os.path.join(cfg['act_dir'], f'{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    data  = torch.load(act_path, map_location='cpu').squeeze(0).float()
    avg   = data.nanmean(dim=1)
    seq_len   = avg.shape[-1]
    base_mask = _qwen35_masks[task]
    pad_len   = seq_len - base_mask.shape[0]
    mask      = (F.pad(base_mask, (0, pad_len), value=False)
                 if pad_len > 0 else base_mask[:seq_len])
    masked = avg[QWEN35_HEAD - 1][mask][1:-1]
    side   = int(np.sqrt(masked.shape[0]))
    grid   = masked[:side * side].reshape(side, side).numpy()
    return normalize(cv2.resize(grid, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))


def get_map(col, task, img_id):
    if col == 'Human':             return human_maps[task][img_id]
    if col in ('CNN-LSTM', 'ViT-LSTM'): return get_lstm_map(col, task, img_id)
    if col in ('CNN-Transformer', 'ViT-Transformer'): return get_trans_map(col, task, img_id)
    if col == 'Molmo':             return get_molmo_map(task, img_id)
    if col == 'Qwen3.5':           return get_qwen35_map(task, img_id)


print('\nPlotting...')
N_IMGS = len(BLEND_IMGS)
N_COLS = len(COLS)
N_ROWS = N_IMGS * 2

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(4.5 * N_COLS, 4.5 * N_ROWS),
                          squeeze=False, dpi=150)

for j, col in enumerate(COLS):
    axes[0, j].set_title(COL_HEADERS[col], fontsize=55, fontweight='bold',
                         pad=4, color=COL_COLORS[col])

TASKS = ['social'] * N_IMGS + ['describe'] * N_IMGS
IMG_ORDER = BLEND_IMGS * 2

for row_idx, (task, img_id) in enumerate(zip(TASKS, IMG_ORDER)):
    img   = np.array(Image.open(os.path.join(IMG_DIR, f'{img_id}.jpg'))
                     .convert('RGB').resize((SIZE, SIZE)))
    img_f = img.astype(np.float32) / 255.0

    for col_idx, col in enumerate(COLS):
        ax   = axes[row_idx, col_idx]
        attn = get_map(col, task, img_id)

        if attn is None:
            ax.imshow(img)
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
        else:
            ax.imshow(attn_overlay(img_f, attn))
        ax.axis('off')

plt.tight_layout(rect=[0.08, 0, 1, 1])
fig.subplots_adjust(hspace=0.35)

fig.canvas.draw()
y_social_bottom = axes[N_IMGS - 1, 0].get_position().y0
y_describe_top  = axes[N_IMGS,     0].get_position().y1
line_y_center   = (y_social_bottom + y_describe_top) / 2

fig.add_artist(plt.Line2D([0.08, 0.99], [line_y_center, line_y_center],
                           transform=fig.transFigure,
                           color='#333333', linewidth=15, linestyle='--'))

y_social_top = axes[0, 0].get_position().y1
y_soc_center = (y_social_bottom + y_social_top) / 2
y_des_center = (0.0 + y_describe_top) / 2
fig.text(0.04, y_soc_center, 'Social Task',   fontsize=75, fontweight='bold',
         ha='center', va='center', rotation=90, color='#2C3E50')
fig.text(0.04, y_des_center, 'Describe Task', fontsize=75, fontweight='bold',
         ha='center', va='center', rotation=90, color='#2C3E50')

out_path = os.path.join(OUT_DIR, 'fig5_qwen35_attn_by_task.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.close(fig)
