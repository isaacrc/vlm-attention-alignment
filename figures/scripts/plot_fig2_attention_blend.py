"""
plot_fig2_attention_blend_vlm.py
---------------------------------
Extends fig2_attention_blend_best_heads with two additional columns:
  Molmo 7B-D (L9H24, social best head)
  Qwen2-VL 7B (L6H8, social best head)

7 columns total:
  Human (sub-003) | CNN-LSTM | CNN-Transformer (H5) | ViT-LSTM |
  ViT-Transformer (H8) | Molmo 7B-D (L9H24) | Qwen2-VL 7B (L6H8)

Output: architecture_manipulation/manuscript_plots/fig2_attention_blend_vlm.png
"""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import textwrap
import torch
import torch.nn.functional as F
import cv2

# ── Paths ─────────────────────────────────────────────────────────────────────
ARCH_DIR    = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR   = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
LTA_DIR     = os.path.join(GROUP_DIR, 'learning_to_attend')
REPL_DIR    = os.path.join(LTA_DIR, 'social_replication')
FINAL_DIR   = os.path.join(REPL_DIR, 'final_replicated_model')
MOLMO_DIR   = '/scratch/gpfs/GRAZIANO/CANS/molmo/molmo'
PROBING_DIR = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing'
OUT_DIR     = os.path.join(ARCH_DIR, 'manuscript_plots')

IMG_DIR     = os.path.join(GROUP_DIR, 'imgs')
HUM_SOC_DIR = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
CAP_SOC_DIR = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_social')

# ── Constants ─────────────────────────────────────────────────────────────────
BLEND_IMGS = ['000000157601', '000000232649', '000000535306']
SIZE       = 256
CMAP       = cm.hot
ALPHA      = 0.6

BEST_HEAD_TRANS = {'CNN_Transformer': 5, 'ViT_Transformer': 8}   # same as original
MOLMO_LAYER, MOLMO_HEAD = 9, 24   # 1-indexed; social task best
QWEN_LAYER,  QWEN_HEAD  = 6, 8   # 1-indexed; social task best

# Molmo tiling constants (3×3 crops, verified from image_input_idx.pt shape)
NUM_CROPS_X = NUM_CROPS_Y = 3

# ── Colors ────────────────────────────────────────────────────────────────────
COL_COLORS = {
    'Human':           '#7D3C98',
    'CNN-LSTM':        '#5D6D7E',
    'CNN-Transformer': '#E74C3C',
    'ViT-LSTM':        '#27AE60',
    'ViT-Transformer': '#8E44AD',
    'Molmo':           '#F39C12',
    'Qwen':            '#1ABC9C',
}
COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer',
        'Molmo', 'Qwen']
COL_HEADERS = {
    'Human':           'Human (sub-003)',
    'CNN-LSTM':        'CNN-LSTM',
    'CNN-Transformer': f'CNN-Transformer (H{BEST_HEAD_TRANS["CNN_Transformer"]})',
    'ViT-LSTM':        'ViT-LSTM',
    'ViT-Transformer': f'ViT-Transformer (H{BEST_HEAD_TRANS["ViT_Transformer"]})',
    'Molmo':           f'Molmo 7B-D (L{MOLMO_LAYER}H{MOLMO_HEAD})',
    'Qwen':            f'Qwen2-VL 7B (L{QWEN_LAYER}H{QWEN_HEAD})',
}

# ── Utility ───────────────────────────────────────────────────────────────────
def normalize(x):
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)

def alpha_to_heatmap(alpha_vec):
    arr = alpha_vec.reshape(14, 14).astype(np.float32)
    resized = np.array(Image.fromarray(arr).resize((SIZE, SIZE), resample=Image.BILINEAR))
    return normalize(resized)


# ── RNN model NPZs ────────────────────────────────────────────────────────────
MODEL_ALPHA_NPZS = {
    'CNN-LSTM':        os.path.join(FINAL_DIR, 'ckpt_cache', 'soc_bs32_beam5_E2.npz'),
    'CNN-Transformer': os.path.join(ARCH_DIR, 'CNN_Transformer', 'ckpt_cache',
                                    'social_unfrozen_beam5_E003.npz'),
    'ViT-LSTM':        os.path.join(ARCH_DIR, 'ViT_LSTM', 'ckpt_cache',
                                    'social_unfrozen_beam5_E001.npz'),
    'ViT-Transformer': os.path.join(ARCH_DIR, 'ViT_Transformer', 'ckpt_cache',
                                    'social_unfrozen_beam5_E001.npz'),
}
HEAD_VEC_NPZS = {
    'CNN-Transformer': os.path.join(ARCH_DIR, 'CNN_Transformer', 'ckpt_cache',
                                    'social_unfrozen_beam5_E003_head_vecs_L3.npz'),
    'ViT-Transformer': os.path.join(ARCH_DIR, 'ViT_Transformer', 'ckpt_cache',
                                    'social_unfrozen_beam5_E001_head_vecs_L3.npz'),
}

# ── Load RNN model data ───────────────────────────────────────────────────────
print('Loading RNN model attention maps and captions...')
model_alphas   = {}
model_captions = {}
for model_name, npz_path in MODEL_ALPHA_NPZS.items():
    raw = np.load(npz_path, allow_pickle=True)
    model_alphas[model_name]   = raw['alphas'].item()
    model_captions[model_name] = raw['caption'].item() if 'caption' in raw.files else {}

target_imgs = pd.read_csv(
    os.path.join('/scratch/gpfs/GRAZIANO/CANS/rnn_model', 'results', 'target_imgs.csv')
)['image_name'].tolist()
target_idx = {im: i for i, im in enumerate(target_imgs)}

head_vec_data = {}
for model_name, npz_path in HEAD_VEC_NPZS.items():
    data = np.load(npz_path, allow_pickle=True)
    head_vec_data[model_name] = {
        'head_vecs':  data['head_vecs'],
        'common_idx': data['common_idx'].astype(int).tolist(),
    }

def get_rnn_attn_map(model_name, img_id):
    if model_name in head_vec_data:
        hv   = head_vec_data[model_name]['head_vecs']
        cidx = head_vec_data[model_name]['common_idx']
        cond_key = model_name.replace('-', '_')
        head_idx = BEST_HEAD_TRANS[cond_key] - 1
        ti = target_idx.get(f'{img_id}.jpg', target_idx.get(img_id, -1))
        if ti < 0 or ti not in cidx:
            return None
        ii  = cidx.index(ti)
        arr = hv[head_idx, ii].reshape(256, 256).astype(np.float32)
        return normalize(arr)
    alphas = model_alphas[model_name]
    key = f'{img_id}.jpg' if f'{img_id}.jpg' in alphas else img_id
    if key not in alphas:
        return None
    return alpha_to_heatmap(alphas[key])

def get_rnn_caption(model_name, img_id):
    caps = model_captions[model_name]
    key  = f'{img_id}.jpg' if f'{img_id}.jpg' in caps else img_id
    cap  = str(caps.get(key, '')).strip()
    for tok in ['<start>', '<end>']:
        cap = cap.replace(tok, '').strip()
    if cap:
        cap = cap[0].upper() + cap[1:]
    if cap and not cap.endswith('.'):
        cap += '.'
    return cap


# ── Molmo: spatial reconstruction setup ──────────────────────────────────────
print('Setting up Molmo spatial grid...')
molmo_act_dir = os.path.join(MOLMO_DIR, 'activations', 'social')
_idx = torch.load(os.path.join(molmo_act_dir, 'image_input_idx.pt'),
                  map_location='cpu')[0].numpy()
_img_idx_all = (_idx[1:]
                .reshape(NUM_CROPS_X, NUM_CROPS_Y, 12, 12)
                .transpose(0, 2, 1, 3)
                .reshape(NUM_CROPS_X * 12, NUM_CROPS_Y * 12))
_valid_idx       = np.where(_img_idx_all > 0)
_img_pos         = _img_idx_all[_valid_idx]
_full_h = NUM_CROPS_X * 336
_full_w = NUM_CROPS_Y * 336
_idx_resized = cv2.resize(_img_idx_all, (_full_w, _full_h), interpolation=cv2.INTER_NEAREST)
_row_valid   = np.any(_idx_resized > 0, axis=1)
_col_valid   = np.any(_idx_resized > 0, axis=0)

def get_molmo_attn_map(img_id):
    """Extract L{MOLMO_LAYER} H{MOLMO_HEAD} spatial map for img_id."""
    act_path = os.path.join(molmo_act_dir, f'activations_img_{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    attn = torch.load(act_path, map_location='cpu').numpy()
    # attn: (28 layers, 28 heads, n_tokens, 1028)
    n_tokens  = attn.shape[2]
    head_idx  = MOLMO_HEAD - 1   # 0-indexed
    layer_idx = MOLMO_LAYER

    H, W = _img_idx_all.shape
    # Zero-initialize so non-image positions don't bleed into upsampled maps
    vol = np.zeros((n_tokens, H, W), dtype=np.float32)
    for tok in range(n_tokens):
        vol[tok, _valid_idx[0], _valid_idx[1]] = attn[layer_idx, head_idx, tok, _img_pos]

    grid = vol.sum(axis=0)  # (H, W) — summed over tokens
    # Molmo has 28×28 valid cells vs LSTM's 14×14, so cells are half the size
    # and the map is 2× sharper. Apply sigma=1.5 grid-cells of blur to match
    # the effective softness of a 14×14→256 bilinear upscale. (zero-init above
    # means non-image cells are 0 so blur doesn't bleed from outside the image.)
    grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=1.5, sigmaY=1.5)
    resized = cv2.resize(grid, (_full_w, _full_h), interpolation=cv2.INTER_LINEAR)
    cropped = resized[_row_valid][:, _col_valid]
    out = cv2.resize(cropped, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return normalize(out)


# ── Molmo captions ────────────────────────────────────────────────────────────
_molmo_caps_df = pd.read_csv(os.path.join(molmo_act_dir, 'generated_captions.csv'))
_molmo_caps_df['img_id'] = _molmo_caps_df['image_path'].apply(
    lambda p: os.path.basename(p).replace('.jpg', ''))
_molmo_caps = dict(zip(_molmo_caps_df['img_id'], _molmo_caps_df['generated_text']))

def get_molmo_caption(img_id):
    cap = str(_molmo_caps.get(img_id, '')).strip()
    if cap and not cap.endswith('.'):
        cap += '.'
    return cap


# ── Qwen: per-head map extraction ─────────────────────────────────────────────
print('Setting up Qwen image mask...')
_qwen_act_dir  = os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen',
                              'activations', f'attn-layer-{QWEN_LAYER}')
_qwen_mask_raw = torch.load(
    os.path.join(PROBING_DIR, 'output', 'social_cap', 'image_mask.pt'),
    map_location='cpu')
_qwen_base_mask = _qwen_mask_raw.squeeze(0) if _qwen_mask_raw.dim() == 2 else _qwen_mask_raw

def get_qwen_attn_map(img_id):
    """Extract L{QWEN_LAYER} H{QWEN_HEAD} spatial map for img_id."""
    act_path = os.path.join(_qwen_act_dir, f'{img_id}.pt')
    if not os.path.exists(act_path):
        return None
    data = torch.load(act_path, map_location='cpu').squeeze(0).float()
    # data: (N_HEADS, n_query_tokens, seq_len)
    avg = data.mean(dim=1)   # (28, seq_len)

    seq_len = avg.shape[-1]
    pad_len = seq_len - _qwen_base_mask.shape[0]
    mask = (F.pad(_qwen_base_mask, (0, pad_len), value=False)
            if pad_len > 0 else _qwen_base_mask[:seq_len])

    head_idx = QWEN_HEAD - 1   # 0-indexed
    masked = avg[head_idx][mask]   # (n_image_tokens,)
    masked = masked[1:-1]          # remove CLS/SEP → 81 tokens
    side   = int(np.sqrt(masked.shape[0]))
    grid   = masked[:side*side].reshape(side, side).numpy()
    out    = cv2.resize(grid, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return normalize(out)


# ── Qwen captions ─────────────────────────────────────────────────────────────
_qwen_resp_df = pd.read_csv(
    os.path.join(PROBING_DIR, 'output', 'social_cap', 'responses.csv'))
_qwen_resp_df['img_id'] = _qwen_resp_df['image_name'].astype(int).apply(
    lambda x: f'{x:012d}')

def parse_qwen_caption(raw):
    # Raw string stores newlines as literal \n — match accordingly
    m = re.search(r'assistant\\n(.+?)(?:\'|"]|\]|$)', str(raw))
    cap = m.group(1).strip().strip("'\"") if m else str(raw)
    if cap and not cap.endswith('.'):
        cap += '.'
    return cap

_qwen_caps = {row['img_id']: parse_qwen_caption(row['caption'])
              for _, row in _qwen_resp_df.iterrows()}

def get_qwen_caption(img_id):
    return _qwen_caps.get(img_id, '')


# ── Load Human data ───────────────────────────────────────────────────────────
print('Loading sub003 gaze and captions...')
human_maps    = np.load(os.path.join(HUM_SOC_DIR, 'sub003_heatmaps.npy'),
                        allow_pickle=True).item()
sub003_caps_df = pd.read_csv(os.path.join(CAP_SOC_DIR, 'sub-003_trans.csv'))
sub003_caps_df = sub003_caps_df.set_index('index')

def get_sub003_caption(img_id):
    key = f'{img_id}.jpg'
    return sub003_caps_df.loc[key, 'caption'] if key in sub003_caps_df.index else ''


# ── Master dispatch ───────────────────────────────────────────────────────────
def get_attn_map(col, img_id):
    if col == 'Molmo':
        return get_molmo_attn_map(img_id)
    if col == 'Qwen':
        return get_qwen_attn_map(img_id)
    return get_rnn_attn_map(col, img_id)

def get_caption(col, img_id):
    if col == 'Human':
        return get_sub003_caption(img_id)
    if col == 'Molmo':
        return get_molmo_caption(img_id)
    if col == 'Qwen':
        return get_qwen_caption(img_id)
    return get_rnn_caption(col, img_id)


# ── Plot ──────────────────────────────────────────────────────────────────────
print('\nPlotting...')
N_COLS = len(COLS)
N_ROWS = len(BLEND_IMGS)

fig, axes = plt.subplots(N_ROWS, N_COLS,
                          figsize=(3.5 * N_COLS, 4.5 * N_ROWS),
                          squeeze=False)

for j, col in enumerate(COLS):
    axes[0, j].set_title(COL_HEADERS[col], fontsize=11, fontweight='bold',
                         pad=10, color=COL_COLORS[col])

for row, img_id in enumerate(BLEND_IMGS):
    img = np.array(Image.open(os.path.join(IMG_DIR, f'{img_id}.jpg'))
                   .convert('RGB').resize((SIZE, SIZE)))
    print(f'  [{row+1}/{N_ROWS}] {img_id}')

    for col_idx, col in enumerate(COLS):
        ax = axes[row, col_idx]
        ax.imshow(img)

        if col == 'Human':
            fname = f'{img_id}.jpg'
            if fname in human_maps:
                hmap = np.array(Image.fromarray(
                    np.nan_to_num(human_maps[fname], nan=0.0)
                ).resize((SIZE, SIZE), resample=Image.NEAREST))
                ax.imshow(CMAP(normalize(hmap)), alpha=ALPHA)
        else:
            amap = get_attn_map(col, img_id)
            if amap is not None:
                ax.imshow(CMAP(amap), alpha=ALPHA)

        ax.axis('off')
        cap = get_caption(col, img_id)
        if cap:
            ax.text(0.5, -0.03, '\n'.join(textwrap.wrap(cap, width=34)),
                    fontsize=9, va='top', ha='center', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.75, pad=3, edgecolor='none'))

plt.tight_layout(h_pad=0.4)
out = os.path.join(OUT_DIR, 'fig2_attention_blend_vlm.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
plt.close(fig)
