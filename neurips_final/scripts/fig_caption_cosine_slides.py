"""
fig_caption_cosine_slides.py
----------------------------
Caption similarity to humans, in the style of fig3_describe_social_slide.png:
Describe and Social panels side by side, one bar per model.

Metric: all-mpnet-base-v2 embeddings, cosine(model caption, human caption) per
image, averaged over images within each participant; bars are the mean +- SEM
over participants (Describe N=22, Social N=27). Human reference (dashed line):
each participant's caption against every other participant's caption for the
same image, averaged the same way.

Captions come from the same checkpoints Fig 3 uses (rebuttal/shared/alignment_core.py
NPZ_FILE). Qwen3.5 uses the thinking-off re-run (CANS/vlm-probing/output/
{coco,social_cap}/qwen35_nothink), because the paper's thinking-on run stopped
inside <think> and never produced a caption.

Outputs (neurips_final/final_figures/slides/):
  caption_cosine_describe_social_slide.png
  caption_cosine_summary.csv
"""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
BASE = '/scratch/gpfs/GRAZIANO/CANS'
ARCH = os.path.join(BASE, 'architecture_manipulation')
GROUP = os.path.join(BASE, 'group_analysis')
PROBE = os.path.join(BASE, 'vlm-probing', 'output')
OUT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'slides')

HUM_DIR = {'describe': os.path.join(GROUP, 'captions_humans', 'CLEANED_trans_desc'),
           'social':   os.path.join(GROUP, 'captions_humans', 'CLEANED_trans_social')}
DESCRIBE_EXCLUDE = {'sub-032', 'sub-050', 'sub-057'}   # canonical Describe N=22

NPZ = {   # (cond, describe npz, social npz) — best epochs, as in alignment_core.NPZ_FILE
    'CNN-LSTM':        ('CNN_LSTM', os.path.join(GROUP, 'processed_heatmaps_rnn', 'h_c_alpha_n200_describe_final.npz'),
                        os.path.join(ARCH, 'CNN_LSTM', 'ckpt_cache', 'cans_beam5.npz')),
    'CNN-Transformer': ('CNN_Transformer', 'phase2_beam5_E014.npz', 'social_unfrozen_beam5_E003.npz'),
    'ViT-LSTM':        ('ViT_LSTM', 'phase2_beam5_E003.npz', 'social_unfrozen_beam5_E001.npz'),
    'ViT-Transformer': ('ViT_Transformer', 'phase2_beam5_E011.npz', 'social_unfrozen_beam5_E001.npz'),
}


def img_key(x):
    """'000000463618.jpg' / '463618' / '000000463618' -> 463618"""
    return int(os.path.splitext(os.path.basename(str(x).strip()))[0])


def clean(cap):
    toks = [t for t in str(cap).split() if t not in ('<start>', '<end>')]
    return ' '.join(toks).strip()


def model_captions(model, task):
    if model in NPZ:
        cond, des, soc = NPZ[model]
        f = des if task == 'describe' else soc
        if not os.path.isabs(f):
            f = os.path.join(ARCH, cond, 'ckpt_cache', f)
        caps = np.load(f, allow_pickle=True)['caption'].item()
        return {img_key(k): clean(v) for k, v in caps.items()}
    if model == 'Molmo':
        d = pd.read_csv(os.path.join(GROUP, 'captions_model', f'captions_molmo_{task}.csv'))
        return {img_key(p): str(t).strip() for p, t in zip(d.image_path, d.generated_text)}
    if model == 'Qwen3.5':
        sub = 'coco' if task == 'describe' else 'social_cap'
        d = pd.read_csv(os.path.join(PROBE, sub, 'qwen35_nothink', 'responses.csv'), dtype=str)
        return {img_key(k): str(c).strip() for k, c in zip(d.image_name, d.caption)}


def human_captions(task):
    """Same parsing as analysis/compute_correlations.py::compute_caption_cosine."""
    out = {}
    for fn in sorted(os.listdir(HUM_DIR[task])):
        if not fn.endswith('_trans.csv'):
            continue
        sub = fn.split('_')[0]
        if task == 'describe' and sub in DESCRIBE_EXCLUDE:
            continue
        fpath = os.path.join(HUM_DIR[task], fn)
        if task == 'describe':
            rows = []
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                next(f)
                for line in f:
                    line = line.strip()
                    if line:
                        idx, cap = line.split(',', 1)
                        rows.append((idx.strip(), cap.replace(',', '').strip().strip('"')))
            df = pd.DataFrame(rows, columns=['index', 'caption'])
        else:
            df = pd.read_csv(fpath, sep=None, engine='python', dtype=str, encoding_errors='replace')
            df.columns = [c.strip().lstrip('﻿') for c in df.columns]
            if 'index' not in df.columns:
                df.rename(columns={df.columns[0]: 'index'}, inplace=True)
        df = df[df['caption'].notna()]
        df = df[~df['caption'].astype(str).str.strip().isin(['NOT FOUND', ''])]
        out[sub] = {img_key(i): str(c) for i, c in zip(df['index'], df['caption'])}
    return out


MODELS_6 = ['CNN-LSTM', 'CNN-Transformer', 'ViT-LSTM', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
embedder = SentenceTransformer('all-mpnet-base-v2')
enc = lambda texts: embedder.encode(texts, normalize_embeddings=True, batch_size=128)

imgs = sorted(img_key(x) for x in pd.read_csv(os.path.join(BASE, 'rnn_model', 'results', 'target_imgs.csv')).iloc[:, 0])
rows = []
for task in ['describe', 'social']:
    hum = human_captions(task)
    subs = sorted(hum)
    H = {s: dict(zip(hum[s], enc(list(hum[s].values())))) for s in subs}
    # human-human: each participant vs every other participant, per image
    for s in subs:
        per_img = [np.mean([H[s][i] @ H[o][i] for o in subs if o != s and i in H[o]])
                   for i in imgs if i in H[s]]
        rows.append(dict(task=task, model='Human', subject_id=s, cos=np.mean(per_img)))
    for m in MODELS_6:
        caps = model_captions(m, task)
        M = dict(zip(caps, enc(list(caps.values()))))
        for s in subs:
            sims = [M[i] @ H[s][i] for i in imgs if i in M and i in H[s]]
            rows.append(dict(task=task, model=m, subject_id=s, cos=np.mean(sims)))
        print(task, m, f'{len(caps)} captions', flush=True)

per_sub = pd.DataFrame(rows)
summ = (per_sub.groupby(['task', 'model'])['cos']
        .agg(mean='mean', sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n='size').reset_index())
summ.to_csv(os.path.join(OUT, 'caption_cosine_summary.csv'), index=False)
print(summ.to_string())

# ── plot: reuse Fig 3 styling (colors, tick labels) ──────────────────────────────
SRC = os.path.join(ROOT, 'rebuttal', 'plots', 'plot_fig_head_selection_corrected.py')
src = open(SRC).read()
ns = {'__file__': SRC, '__name__': 'fig3_src'}
exec(src[:src.index('panel_configs = ')], ns)
plt, COLORS, make_tick = ns['plt'], ns['COLORS'], ns['make_tick']
FS_TITLE, FS_TICK, FS_LABEL = 34, 22, 24
Y_MAX = 0.8

fig = plt.figure(figsize=(2 * 13.333, 7.5), dpi=300, facecolor='white')
for k, task in enumerate(['describe', 'social']):
    ax = fig.add_axes([0.075 + 0.5 * k, 0.27, 0.40, 0.62])
    s = summ[summ.task == task].set_index('model')
    x = np.arange(len(MODELS_6))
    ax.bar(x, s.loc[MODELS_6, 'mean'], ns['BAR_W'], color=[COLORS[m] for m in MODELS_6],
           edgecolor='black', linewidth=0.6, zorder=3)
    ax.errorbar(x, s.loc[MODELS_6, 'mean'], yerr=s.loc[MODELS_6, 'sem'], fmt='none',
                color='black', capsize=6, linewidth=2.0, zorder=4)
    hh = s.loc['Human', 'mean']
    ax.axhline(hh, color='black', linestyle='--', linewidth=2.5, zorder=2)
    ax.text(0.01, hh / Y_MAX + 0.015, 'Human–Human', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=FS_TICK - 4)
    ax.set_xticks(x)
    ax.set_xticklabels([make_tick(m) for m in MODELS_6], fontsize=FS_TICK, rotation=45, ha='right')
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(np.arange(0, Y_MAX + 0.01, 0.2))
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'{task.capitalize()} Task', fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.set_ylabel('Caption similarity\nto humans (cosine)', fontsize=FS_LABEL, labelpad=12)

out = os.path.join(OUT, 'caption_cosine_describe_social_slide.png')
fig.savefig(out, dpi=300, facecolor='white')
print('wrote', out)
