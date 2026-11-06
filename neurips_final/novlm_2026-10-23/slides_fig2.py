"""
slides_fig2.py -- Figure 2 of main_2026-10-23.tex as 16:9 PowerPoint slides, one per task
(13.333 x 7.5 in, 300 dpi). The maps are computed exactly as in fig2_attention_maps.py (same
heads); the slide layout follows scripts/fig5_slides.py. Unlike the paper figure, the slides
show only the human mean and the four trained models (no Molmo or Qwen3.5 columns).
Writes final_figures/manuscript_2026-10-23/slides/fig2_{social,describe}_slide.png
"""
import os, json

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'archive', 'ICLR', 'scripts', 'regen_fig5_entropy_cvheads.py')
SEL = json.load(open(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results', 'selected_heads.json')))
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'slides')
os.makedirs(OUT, exist_ok=True)

# fig2_attention_maps.py's setup, stopped once the maps are computed (before the paper grid)
w = open(os.path.join(HERE, 'fig2_attention_maps.py')).read()
w = w.replace("g['OUT_FIG'] = OUT\n", '')
old = "exec(compile(src[cut:], SRC, 'exec'), g)"
assert w.count(old) == 1
w = w.replace(old, "exec(compile(src[cut:src.index('N = len(BLEND_IMGS); NR')], SRC, 'exec'), g)")
w = w.replace("OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'manuscript_2026-10-23', 'fig2_attention_maps.png')", '')
ns = {'__file__': os.path.join(HERE, 'fig2_attention_maps.py'), '__name__': 'fig2_slides'}
exec(compile(w, ns['__file__'], 'exec'), ns)
g = ns['g']
np, plt, Image = g['np'], g['plt'], g['Image']
maps, COL_COLORS, header = g['maps'], g['COL_COLORS'], g['header']
COLS = [c for c in g['COLS'] if c not in ('Molmo', 'Qwen3.5')]   # slides: human and the four trained models only
BLEND_IMGS, IMG_DIR, SIZE, ALPHA, CMAP, normalize = (g[k] for k in ('BLEND_IMGS', 'IMG_DIR', 'SIZE', 'ALPHA', 'CMAP', 'normalize'))

TITLES = {'social': 'Social Task', 'describe': 'Describe Task'}
W, H_IN = 13.333, 7.5
CELL, GAP_X, GAP_Y = 1.72, 0.12, 0.08
TITLE_H, HEAD_H = 0.75, 1.05
N, NC = len(BLEND_IMGS), len(COLS)
x0 = (W - (NC * CELL + (NC - 1) * GAP_X)) / 2
y_top = H_IN - TITLE_H - HEAD_H
for task in ('social', 'describe'):
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    fig.text(0.5, 1 - 0.45 / H_IN, TITLES[task], fontsize=30, fontweight='bold', ha='center', va='center', color='#2C3E50')
    for r, im in enumerate(BLEND_IMGS):
        img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
        for c, col in enumerate(COLS):
            left = x0 + c * (CELL + GAP_X); bottom = y_top - (r + 1) * CELL - r * GAP_Y
            ax = fig.add_axes([left / W, bottom / H_IN, CELL / W, CELL / H_IN])
            heat = CMAP(normalize(maps[task][col][im]))[:, :, :3].astype(np.float32)
            ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
            if r == 0:
                ax.set_title(header(col, task), fontsize=16, fontweight='bold', pad=6, color=COL_COLORS[col])
    out = os.path.join(OUT, f'fig2_{task}_slide.png')
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)

# ── same slides with each agent's caption under its map ──────────────────────
# Models: the caption generated with the map (beam 5, the checkpoints Figure 2 uses).
# Human: the map is the group mean, so the caption is one example participant's
# (Describe sub-029, as scripts/fig5_slides.py; Social sub-005, avoiding sub-003/004).
import textwrap
LSTM_NPZ, TRANS_NPZ = g['LSTM_NPZ'], g['TRANS_NPZ']
HUM_CAP = {t: os.path.join(g['GROUP_DIR'], 'captions_humans', d, f'{s}_trans.csv')
           for t, d, s in (('describe', 'CLEANED_trans_desc', 'sub-029'), ('social', 'CLEANED_trans_social', 'sub-005'))}


def clean(t):
    t = t.replace('<start>', '').replace('<end>', '').strip()
    return t[:1].upper() + t[1:]


def captions(task):
    caps = {}
    with open(HUM_CAP[task]) as f:   # captions contain unquoted commas: split on the first one only
        next(f)
        caps['Human'] = {a.replace('.jpg', ''): b for a, b in (l.strip().split(',', 1) for l in f if ',' in l)}
    for col, p in (('CNN-LSTM', LSTM_NPZ['CNN-LSTM'][task]), ('ViT-LSTM', LSTM_NPZ['ViT-LSTM'][task]),
                   ('CNN-Transformer', TRANS_NPZ['CNN-Transformer'][task] + '.npz'),
                   ('ViT-Transformer', TRANS_NPZ['ViT-Transformer'][task] + '.npz')):
        caps[col] = {k.replace('.jpg', ''): v for k, v in np.load(p, allow_pickle=True)['caption'].item().items()}
    return caps


CELL_C, CAP_H, GAP_C = 1.35, 0.70, 0.55
TITLE_C, HEAD_C = 0.45, 0.80
PITCH = CELL_C + GAP_C
x0 = (W - (NC * CELL_C + (NC - 1) * GAP_C)) / 2
y_top = H_IN - TITLE_C - HEAD_C
for task in ('social', 'describe'):
    caps = captions(task)
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    fig.text(0.5, 1 - 0.28 / H_IN, TITLES[task], fontsize=24, fontweight='bold', ha='center', va='center', color='#2C3E50')
    for r, im in enumerate(BLEND_IMGS):
        img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
        for c, col in enumerate(COLS):
            left = x0 + c * PITCH; bottom = y_top - (r + 1) * CELL_C - r * CAP_H
            ax = fig.add_axes([left / W, bottom / H_IN, CELL_C / W, CELL_C / H_IN])
            heat = CMAP(normalize(maps[task][col][im]))[:, :, :3].astype(np.float32)
            ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
            if r == 0:
                ax.set_title(header(col, task), fontsize=13, fontweight='bold', pad=5, color=COL_COLORS[col])
            fig.text((left + CELL_C / 2) / W, (bottom - 0.05) / H_IN, textwrap.fill(clean(caps[col][im]), 28),
                     fontsize=8, ha='center', va='top', color='#222', linespacing=1.15)
    out = os.path.join(OUT, f'fig2_{task}_captions_slide.png')
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)

# ── Human, CNN-LSTM, CNN-Transformer, each with a Social and a Describe column ─
# Same images as above. One column group per agent; within it, Social then Describe.
# CNN-Transformer uses one head for both tasks: the head of the Figure 1 between-task panel
# (molmo_bilinear_pairwise/results/model_between_task_fig1.csv, L4H4), so the two columns differ
# only by task. (The captions are the model's own and do not depend on the head.)
import pandas as pd
_bt = pd.read_csv(os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise', 'results',
                               'model_between_task_fig1.csv')).set_index('model').loc['CNN-Transformer', 'head']
BT_HEAD = tuple(int(v) for v in _bt[1:].split('H'))
bt_maps = {}
_saved = dict(g['HEADS']['CNN-Transformer'])
for _t in ('social', 'describe'):
    g['HEADS']['CNN-Transformer'][_t] = BT_HEAD
    bt_maps[_t] = g['trans_maps']('CNN-Transformer', _t)
g['HEADS']['CNN-Transformer'].update(_saved)
SUB = ['Human', 'CNN-LSTM', 'CNN-Transformer']
AGENT = {'Human': 'Human (Mean)', 'CNN-LSTM': 'CNN-LSTM', 'CNN-Transformer': f'CNN-Transformer (L{BT_HEAD[0]}H{BT_HEAD[1]})'}


def by_agent(name, cell, gap_in, group_gap, gap_y, top_h, cap_h=0.0, cap_fs=8, wrap=24, sub_fs=15):
    caps = {t: captions(t) for t in ('social', 'describe')} if cap_h else None
    group_w = 2 * cell + gap_in
    x0 = (W - (3 * group_w + 2 * group_gap)) / 2
    y_top = H_IN - top_h
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    for gi, col in enumerate(SUB):
        gx = x0 + gi * (group_w + group_gap)
        fig.text((gx + group_w / 2) / W, 1 - 0.38 / H_IN, AGENT[col], fontsize=21, fontweight='bold',
                 ha='center', va='center', color=COL_COLORS[col])
        fig.add_artist(plt.Line2D([gx / W, (gx + group_w) / W], [1 - 0.62 / H_IN] * 2, color=COL_COLORS[col], linewidth=2))
        for ti, task in enumerate(('social', 'describe')):
            left = gx + ti * (cell + gap_in)
            sub = TITLES[task].replace(' Task', '')
            fig.text((left + cell / 2) / W, (y_top + 0.12) / H_IN, sub, fontsize=sub_fs, fontweight='bold',
                     ha='center', va='bottom', color='#2C3E50')
            for r, im in enumerate(BLEND_IMGS):
                img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
                bottom = y_top - (r + 1) * cell - r * (gap_y + cap_h)
                ax = fig.add_axes([left / W, bottom / H_IN, cell / W, cell / H_IN])
                a = bt_maps[task][im] if col == 'CNN-Transformer' else maps[task][col][im]
                heat = CMAP(normalize(a))[:, :, :3].astype(np.float32)
                ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
                if cap_h:
                    fig.text((left + cell / 2) / W, (bottom - 0.04) / H_IN, textwrap.fill(clean(caps[task][col][im]), wrap),
                             fontsize=cap_fs, ha='center', va='top', color='#222', linespacing=1.15)
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)


by_agent('fig2_cnn_social_describe_slide.png', cell=1.88, gap_in=0.08, group_gap=0.40, gap_y=0.08, top_h=1.10)
by_agent('fig2_cnn_social_describe_captions_slide.png', cell=1.40, gap_in=0.62, group_gap=0.45, gap_y=0.05, top_h=1.05,
         cap_h=0.62, cap_fs=8, wrap=24)

# ── updated final_figures/slides/fig5_describe_slide.png: all seven columns, as paper Figure 2 ──
# Same layout as the old slide (scripts/fig5_slides.py) but with Figure 2's maps: Molmo on its
# pairwise-run head, bilinear and unblurred; Qwen3.5 from the thinking-off run.
COLS7 = g['COLS']
x7 = (W - (len(COLS7) * CELL + (len(COLS7) - 1) * GAP_X)) / 2
y7 = H_IN - TITLE_H - HEAD_H
for task in ('describe',):
    fig = plt.figure(figsize=(W, H_IN), dpi=300, facecolor='white')
    fig.text(0.5, 1 - 0.45 / H_IN, TITLES[task], fontsize=30, fontweight='bold', ha='center', va='center', color='#2C3E50')
    for r, im in enumerate(BLEND_IMGS):
        img = np.array(Image.open(os.path.join(IMG_DIR, f'{im}.jpg')).convert('RGB').resize((SIZE, SIZE))).astype(np.float32) / 255.0
        for c, col in enumerate(COLS7):
            left = x7 + c * (CELL + GAP_X); bottom = y7 - (r + 1) * CELL - r * GAP_Y
            ax = fig.add_axes([left / W, bottom / H_IN, CELL / W, CELL / H_IN])
            heat = CMAP(normalize(maps[task][col][im]))[:, :, :3].astype(np.float32)
            ax.imshow(np.clip(img * (1 - ALPHA) + heat * ALPHA, 0, 1)); ax.axis('off')
            if r == 0:
                ax.set_title(header(col, task), fontsize=16, fontweight='bold', pad=6, color=COL_COLORS[col])
    out = os.path.join(OUT, f'fig5_{task}_slide.png')
    fig.savefig(out, dpi=300, facecolor='white'); plt.close(fig)
    print('wrote', out, flush=True)
