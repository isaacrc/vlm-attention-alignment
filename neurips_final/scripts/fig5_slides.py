"""
fig5_slides.py
--------------
Split Fig 5 (attention examples, CV heads) into one 16:9 PowerPoint slide per task.
Reuses the map loaders from archive/ICLR/scripts/regen_fig5_entropy_cvheads.py
(everything above its entropy section). Maps are cached so restyling is fast.

Outputs:
  neurips_final/final_figures/slides/fig5_{social,describe}_slide.png, fig5_describe_slide_captions.png

QWEN_NOTHINK=1 swaps the Qwen3.5 column for the thinking-off re-extraction
(neurips_final/qwen35_nothink/): its activations, its CV heads, and its captions.
Outputs then go to neurips_final/final_figures/qwen35_nothink/slides/.
"""
import os, sys, json, pickle
import numpy as np
from PIL import Image

ROOT    = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
SRC     = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
NOTHINK = os.environ.get('QWEN_NOTHINK') == '1'
OUT     = os.path.join(ROOT, 'neurips_final', 'final_figures', *(('qwen35_nothink', 'slides') if NOTHINK else ('slides',)))
CACHE   = os.path.join(OUT, 'fig5_maps_cache.pkl')
os.makedirs(OUT, exist_ok=True)

src = open(SRC).read()
exec(src[:src.index('# ── 1. entropy')])   # loaders, HEADS, BLEND_IMGS, constants
import matplotlib.pyplot as plt

# Molmo: the archived loader blurs the 36x36 crop grid BEFORE dropping the 8 overlap-margin
# rows/cols (all zeros), which smears zeros into the cells beside each crop border and draws
# seams/squares. Drop the margins first (leaving the true 28x28 token grid), then blur.
def molmo_maps(task, only=None):
    L, H = HEADS['Molmo'][task]
    act_dir = os.path.join(MOLMO_DIR, 'activations', task)
    g = _molmo_grid(act_dir)[0]
    vidx = np.where(g > 0); pos = g[vidx]; rows, cols = np.any(g > 0, axis=1), np.any(g > 0, axis=0)
    out = {}
    for im in only:
        attn = torch.load(os.path.join(act_dir, f'activations_img_{im}.pt'), map_location='cpu')[L - 1, H - 1].float().numpy()
        vol = np.zeros((attn.shape[0],) + g.shape, np.float32)
        vol[:, vidx[0], vidx[1]] = attn[:, pos]
        grid = cv2.GaussianBlur(vol.sum(0)[rows][:, cols], (0, 0), sigmaX=1.5, sigmaY=1.5)
        out[im] = cv2.resize(grid, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return out

# Qwen captions always come from the thinking-off run: the original run never finished a caption
QWEN_CAP_CSV = os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35_nothink', 'responses.csv')
if NOTHINK:
    QCFG = {'social':   (os.path.join(PROBING_DIR, 'output', 'social_cap', 'qwen35_nothink'),),
            'describe': (os.path.join(PROBING_DIR, 'output', 'coco', 'qwen35_nothink'),)}
    sel = json.load(open(os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink',
                                      'results', 'selected_heads.json')))
    HEADS['Qwen3.5'] = {t: (sel[f'Qwen3.5_{t}']['layer'], sel[f'Qwen3.5_{t}']['head']) for t in ('social', 'describe')}

COLS = ['Human', 'CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
COL_COLORS = {'Human': '#2C3E50', 'CNN-LSTM': '#5D6D7E', 'CNN-Transformer': '#E74C3C', 'ViT-LSTM': '#27AE60',
              'ViT-Transformer': '#8E44AD', 'Molmo': '#F39C12', 'Qwen3.5': '#1ABC9C'}
TITLES = {'social': 'Social Task', 'describe': 'Describe Task'}

def header(col, task):
    if col == 'Human': return 'Human\n(Mean)'
    if col in ('CNN-LSTM', 'ViT-LSTM'): return col.replace('-', '\n')
    L, H = HEADS[col][task]
    name = {'CNN-Transformer': 'CNN\nTransformer', 'ViT-Transformer': 'ViT\nTransformer',
            'Molmo': 'Molmo\n7B-D', 'Qwen3.5': 'Qwen\n9B-3.5'}[col]
    return f'{name}\n(L{L}H{H})'

if os.path.exists(CACHE) and '--rebuild' not in sys.argv:
    maps = pickle.load(open(CACHE, 'rb'))
else:
    maps = {}
    for task in ('social', 'describe'):
        subs = human_subject_maps(task)
        hum = {}
        for im in BLEND_IMGS:
            ms = [m[im] for m in subs.values() if im in m]
            hum[im] = normalize(np.array(Image.fromarray(np.mean(ms, 0)).resize((SIZE, SIZE), resample=Image.NEAREST)))
        maps[task] = {'Human': hum}
        for m in ('CNN-LSTM', 'ViT-LSTM'): maps[task][m] = lstm_maps(m, task)
        for m in ('CNN-Transformer', 'ViT-Transformer'): maps[task][m] = trans_maps(m, task)
        maps[task]['Molmo'] = molmo_maps(task, only=BLEND_IMGS)
        maps[task]['Qwen3.5'] = qwen_maps(task, only=BLEND_IMGS)
        maps[task] = {c: {im: d[im] for im in BLEND_IMGS if im in d} for c, d in maps[task].items()}
    pickle.dump(maps, open(CACHE, 'wb'))

# ── slide layout: 13.333 x 7.5 in (PowerPoint 16:9), positions in inches ──────
W, H_IN = 13.333, 7.5
CELL, GAP_X, GAP_Y = 1.72, 0.12, 0.08
TITLE_H, HEAD_H = 0.75, 1.05
N, NC = len(BLEND_IMGS), len(COLS)
x0 = (W - (NC * CELL + (NC - 1) * GAP_X)) / 2
y_top = H_IN - TITLE_H - HEAD_H                      # top edge of first image row
for task in ('social', 'describe'):
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    fig.text(0.5, 1 - 0.45 / H_IN, TITLES[task], fontsize=30, fontweight='bold', ha='center', va='center', color='#2C3E50')
    for r, im in enumerate(BLEND_IMGS):
        img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
        for c, col in enumerate(COLS):
            left = x0 + c * (CELL + GAP_X); bottom = y_top - (r + 1) * CELL - r * GAP_Y
            ax = fig.add_axes([left / W, bottom / H_IN, CELL / W, CELL / H_IN])
            a = maps[task][col].get(im)
            heat = CMAP(normalize(a))[:, :, :3].astype(np.float32)
            ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
            if r == 0:
                ax.set_title(header(col, task), fontsize=16, fontweight='bold', pad=6, color=COL_COLORS[col])
    out = os.path.join(OUT, f'fig5_{task}_slide.png')
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)

# ── Describe slide with each agent's caption under its map ────────────────────
# Human: one example participant (sub-029, not excluded); the map is the group mean.
# Qwen3.5: caption from the thinking-off re-run (the original run stopped mid-<think>).
# Without QWEN_NOTHINK the map is still from the original run, so the slide says so.
import textwrap
import pandas as pd
CAP_NPZ = {'CNN-LSTM':        LSTM_NPZ['CNN-LSTM']['describe'],
           'ViT-LSTM':        LSTM_NPZ['ViT-LSTM']['describe'],
           'CNN-Transformer': TRANS_NPZ['CNN-Transformer']['describe'] + '.npz',
           'ViT-Transformer': TRANS_NPZ['ViT-Transformer']['describe'] + '.npz'}
HUM_CAP = os.path.join(GROUP_DIR, 'captions_humans', 'CLEANED_trans_desc', 'sub-029_trans.csv')
MOLMO_CAP = os.path.join(MOLMO_DIR, 'activations', 'describe', 'generated_captions.csv')

def clean(t):
    t = t.replace('<start>', '').replace('<end>', '').strip()
    return t[:1].upper() + t[1:]

caps = {c: {} for c in COLS}
with open(HUM_CAP) as f:   # captions contain unquoted commas: split on the first one only
    next(f)
    caps['Human'] = {a.replace('.jpg', ''): b for a, b in (l.strip().split(',', 1) for l in f if ',' in l)}
for col, p in CAP_NPZ.items():
    caps[col] = {k.replace('.jpg', ''): v for k, v in np.load(p, allow_pickle=True)['caption'].item().items()}
mol = pd.read_csv(MOLMO_CAP, dtype=str)
caps['Molmo'] = {os.path.basename(p).replace('.jpg', ''): t for p, t in zip(mol.image_path, mol.generated_text)}
qw = pd.read_csv(QWEN_CAP_CSV, dtype=str)
caps['Qwen3.5'] = dict(zip(qw.image_name, qw.caption))

CELL_C, CAP_H = 1.28, 0.78
TITLE_C, HEAD_C = 0.45, 0.85
x0 = (W - (NC * CELL_C + (NC - 1) * 0.50)) / 2; PITCH = CELL_C + 0.50
y_top = H_IN - TITLE_C - HEAD_C
fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
fig.text(0.5, 1 - 0.28 / H_IN, TITLES['describe'], fontsize=24, fontweight='bold', ha='center', va='center', color='#2C3E50')
for r, im in enumerate(BLEND_IMGS):
    img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
    for c, col in enumerate(COLS):
        left = x0 + c * PITCH; bottom = y_top - (r + 1) * CELL_C - r * CAP_H
        ax = fig.add_axes([left / W, bottom / H_IN, CELL_C / W, CELL_C / H_IN])
        heat = CMAP(normalize(maps['describe'][col][im]))[:, :, :3].astype(np.float32)
        ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
        if r == 0:
            ax.set_title(header(col, 'describe'), fontsize=13, fontweight='bold', pad=5, color=COL_COLORS[col])
        fig.text((left + CELL_C / 2) / W, (bottom - 0.05) / H_IN, textwrap.fill(clean(caps[col][im]), 32),
                 fontsize=7, ha='center', va='top', color='#222', linespacing=1.15)
if not NOTHINK:
    fig.text(1 - 0.12 / W, 1 - 0.10 / H_IN, 'Qwen captions: thinking-off re-run; Qwen maps: original run',
             fontsize=7, ha='right', va='top', style='italic', color='#888')
out =os.path.join(OUT, 'fig5_describe_slide_captions.png')
fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
print('wrote', out, flush=True)
