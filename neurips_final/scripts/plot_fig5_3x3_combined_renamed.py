"""
plot_fig5_3x3_combined.py
--------------------------
Combines Social and Describe attention side-by-side for each model.

Column layout (3 groups of 2):
  [ Human: Social | Describe ] [ CNN-LSTM: Social | Describe ] [ CNN-Trans: Social | Describe ]

Two-level column headers:
  Top level  : model name (Human, CNN-LSTM, CNN-Transformer)  — spanning each pair
  Lower level : task (Social | Describe)                       — one per column

figsize = 2× width of individual fig5_3x3_*.png  (28.8 × 17.4 in)

Output: manuscript/figures/images/fig5_3x3_combined.png
"""

import os
import textwrap
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

ARCH_DIR       = '/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation'
GROUP_DIR      = '/scratch/gpfs/GRAZIANO/CANS/group_analysis'
LTA_DIR        = os.path.join(GROUP_DIR, 'learning_to_attend')
REPL_DIR       = os.path.join(LTA_DIR, 'social_replication')
IMG_DIR        = os.path.join(GROUP_DIR, 'imgs')
HUM_SOC_DIR    = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_social')
HUM_DES_DIR    = os.path.join(GROUP_DIR, 'processed_heatmaps_by_subject_descriptive')
HUM_CAP_SOC    = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_social', 'sub-003_trans.csv')
HUM_CAP_DES    = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_desc',   'sub-029_trans.csv')
OUT_DIR        = '/scratch/gpfs/GRAZIANO/CANS/manuscript/figures/images'
PRES_DIR       = os.path.join(OUT_DIR, 'presentation_plots')
TARGET_CSV     = '/scratch/gpfs/GRAZIANO/CANS/rnn_model/results/target_imgs.csv'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRES_DIR, exist_ok=True)

BLEND_IMGS      = ['000000157601', '000000232649', '000000535306']
SIZE            = 256
ALPHA           = 0.6
CMAP            = cm.hot
BEST_HEAD_TRANS = 5   # 1-indexed, L3H5

COLS       = ['Human', 'CNN-LSTM', 'CNN-Transformer']
COL_COLORS = {
    'Human':           '#2C3E50',
    'CNN-LSTM':        '#5D6D7E',
    'CNN-Transformer': '#E74C3C',
}
COL_HEADERS = {
    'Human':           'Human\n(Mean)',
    'CNN-LSTM':        'CNN-LSTM',
    'CNN-Transformer': 'CNN-\nTransformer\n(L3H5)',
}
TASK_COLORS = {'social': '#2980B9', 'describe': '#27AE60'}

# Ordered pairs of (model, task) for the 6 image columns
COLS_TASKS = [
    ('Human',           'social'),
    ('Human',           'describe'),
    ('CNN-LSTM',        'social'),
    ('CNN-LSTM',        'describe'),
    ('CNN-Transformer', 'social'),
    ('CNN-Transformer', 'describe'),
]

# Gridspec column indices for the 6 image axes (spacers at gs cols 2 and 5)
GS_COLS = [0, 1, 3, 4, 6, 7]

# Which COLS_TASKS indices belong to each model group (for spanning header)
MODEL_GROUPS = [
    ('Human',           0, 1),
    ('CNN-LSTM',        2, 3),
    ('CNN-Transformer', 4, 5),
]


def normalize(x):
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)


def attn_overlay(img_f, attn_map):
    heat = CMAP(attn_map)[:, :, :3].astype(np.float32)
    return np.clip(img_f * (1 - ALPHA) + heat * ALPHA, 0, 1)


# ── Human gaze maps ───────────────────────────────────────────────────────────
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


def _load_human_captions(csv_path):
    result = {}
    with open(csv_path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if ',' not in line:
                continue
            img, caption = line.split(',', 1)
            result[img.replace('.jpg', '')] = caption
    return result


human_captions = {
    'social':   _load_human_captions(HUM_CAP_SOC),
    'describe': _load_human_captions(HUM_CAP_DES),
}


def get_human_caption(task, img_id):
    cap = human_captions[task].get(img_id, '')
    return f'Subject 3: {cap}' if cap else ''


# ── CNN-LSTM ──────────────────────────────────────────────────────────────────
_lstm_npz = {
    'social':   REPL_DIR + '/final_replicated_model/ckpt_cache/soc_bs32_beam5_E2.npz',
    'describe': REPL_DIR + '/ckpt_cache/P2_E14_beam5.npz',
}
print('Loading CNN-LSTM data...')
_lstm_data = {}
for task, path in _lstm_npz.items():
    d = np.load(path, allow_pickle=True)
    _lstm_data[task] = {'alphas': d['alphas'].item(), 'caption': d['caption'].item()}
print('  done')


def get_lstm_attn(task, img_id):
    alphas = _lstm_data[task]['alphas']
    key    = f'{img_id}.jpg' if f'{img_id}.jpg' in alphas else img_id
    if key not in alphas:
        return None
    arr  = np.array(alphas[key]).flatten()
    side = int(np.sqrt(len(arr)))
    return normalize(np.array(
        Image.fromarray(arr.reshape(side, side).astype(np.float32))
        .resize((SIZE, SIZE), resample=Image.BILINEAR)))


def get_lstm_caption(task, img_id):
    caps = _lstm_data[task]['caption']
    key  = f'{img_id}.jpg' if f'{img_id}.jpg' in caps else img_id
    raw  = caps.get(key, '')
    return raw.replace('<start>', '').replace('<end>', '').strip()


# ── CNN-Transformer ───────────────────────────────────────────────────────────
target_imgs = pd.read_csv(TARGET_CSV)['image_name'].tolist()
target_idx  = {im: i for i, im in enumerate(target_imgs)}

_trans_attn_npz = {
    'social':   ARCH_DIR + '/CNN_Transformer/ckpt_cache/social_unfrozen_beam5_E003_head_vecs_L3.npz',
    'describe': ARCH_DIR + '/CNN_Transformer/ckpt_cache/phase2_beam5_E014_head_vecs_L3.npz',
}
_trans_cap_npz = {
    'social':   ARCH_DIR + '/CNN_Transformer/ckpt_cache/social_unfrozen_beam5_E003.npz',
    'describe': ARCH_DIR + '/CNN_Transformer/ckpt_cache/phase2_beam5_E014.npz',
}
print('Loading CNN-Transformer data...')
_trans_data = {}
_head_idx = BEST_HEAD_TRANS - 1
for task in ('social', 'describe'):
    d_attn = np.load(_trans_attn_npz[task], allow_pickle=True)
    d_cap  = np.load(_trans_cap_npz[task], allow_pickle=True)
    _trans_data[task] = {
        'head_vecs':  d_attn['head_vecs'][_head_idx],
        'common_idx': d_attn['common_idx'].astype(int).tolist(),
        'caption':    d_cap['caption'].item(),
    }
print('  done')


def get_trans_attn(task, img_id):
    td   = _trans_data[task]
    cidx = td['common_idx']
    hvec = td['head_vecs']
    ti   = target_idx.get(f'{img_id}.jpg', target_idx.get(img_id, -1))
    if ti < 0 or ti not in cidx:
        return None
    return normalize(hvec[cidx.index(ti)].reshape(SIZE, SIZE).astype(np.float32))


def get_trans_caption(task, img_id):
    caps = _trans_data[task]['caption']
    key  = f'{img_id}.jpg' if f'{img_id}.jpg' in caps else img_id
    return caps.get(key, '')


# ── Dispatch ──────────────────────────────────────────────────────────────────
def get_attn(col, task, img_id):
    if col == 'Human':           return human_maps[task][img_id]
    if col == 'CNN-LSTM':        return get_lstm_attn(task, img_id)
    if col == 'CNN-Transformer': return get_trans_attn(task, img_id)


def get_caption(col, task, img_id):
    if col == 'Human':           return get_human_caption(task, img_id)
    if col == 'CNN-LSTM':        return get_lstm_caption(task, img_id)
    if col == 'CNN-Transformer': return get_trans_caption(task, img_id)


def fmt_caption(text):
    if not text:
        return text
    if ': ' in text:
        prefix, body = text.split(': ', 1)
        return f'{prefix}: {body[:1].upper()}{body[1:]}'
    return text[:1].upper() + text[1:]


# ── Build figure ──────────────────────────────────────────────────────────────
print('\nPlotting...')

# figsize: 2× width of one fig5_3x3 (4.8*3=14.4 → 28.8), same height (5.8*3=17.4)
fig = plt.figure(figsize=(28.8, 17.4), dpi=150)

# 8 gridspec columns: [img img spacer img img spacer img img]
gs = gridspec.GridSpec(
    3, 8,
    width_ratios=[1, 1, 0.12, 1, 1, 0.12, 1, 1],
    hspace=0.28, wspace=0.05,
    left=0.005, right=0.995, top=0.86, bottom=0.14,
)

axes = np.array([
    [fig.add_subplot(gs[r, GS_COLS[c]]) for c in range(6)]
    for r in range(3)
])

# Task-level labels on first row (lower-level header)
for c, (col, task) in enumerate(COLS_TASKS):
    axes[0, c].set_title('Social' if task == 'social' else 'Describe',
                          fontsize=22, fontweight='bold',
                          pad=8, color=TASK_COLORS[task])

# Render images and captions
for row_idx, img_id in enumerate(BLEND_IMGS):
    img   = np.array(Image.open(os.path.join(IMG_DIR, f'{img_id}.jpg'))
                     .convert('RGB').resize((SIZE, SIZE)))
    img_f = img.astype(np.float32) / 255.0

    for col_idx, (col, task) in enumerate(COLS_TASKS):
        ax   = axes[row_idx, col_idx]
        attn = get_attn(col, task, img_id)

        if attn is None:
            ax.imshow(img)
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='red')
        else:
            ax.imshow(attn_overlay(img_f, attn))
        ax.axis('off')

        caption = fmt_caption(get_caption(col, task, img_id))
        if caption:
            wrapped = '\n'.join(textwrap.wrap(caption, width=28))
            ax.text(0.5, -0.04, wrapped,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=15, fontweight='bold',
                    color='#1a1a1a',
                    multialignment='center',
                    clip_on=False)

# Compute axis positions for top-level headers and separators
fig.canvas.draw()

# Model-level spanning headers (top-level)
for model_name, c0, c1 in MODEL_GROUPS:
    x0   = axes[0, c0].get_position().x0
    x1   = axes[0, c1].get_position().x1
    ytop = axes[0, c0].get_position().y1
    fig.text((x0 + x1) / 2, ytop + 0.013,
             COL_HEADERS[model_name],
             ha='center', va='bottom',
             fontsize=26, fontweight='bold',
             color=COL_COLORS[model_name])

# Vertical dashed separators between model groups
for c_left, c_right in [(1, 2), (3, 4)]:
    x_sep = (axes[0, c_left].get_position().x1 +
             axes[0, c_right].get_position().x0) / 2
    y_bot = axes[-1, 0].get_position().y0
    y_top = axes[0, 0].get_position().y1 + 0.06
    fig.add_artist(plt.Line2D([x_sep, x_sep], [y_bot, y_top],
                               transform=fig.transFigure,
                               color='#aaaaaa', linewidth=2, linestyle='--',
                               zorder=10))

out_path  = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/slides/fig5_3x3_combined.png'
pres_path = os.path.join(PRES_DIR, 'fig5_3x3_combined.png')
fig.savefig(out_path,  dpi=150, bbox_inches='tight')
# (presentation_plots copy disabled in this renamed version)
print(f'Saved: {out_path}')
print(f'Saved: {pres_path}')
plt.close(fig)
print('Done.')
