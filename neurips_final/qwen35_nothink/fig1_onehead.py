"""
fig1_onehead.py
---------------
Figure 1 with ONE head per model used for both tasks, so panel C compares the same head
across the two prompts. Two variants:
  anchor = social   : each Transformer/VLM uses its cross-validated Social head in A, B, C
  anchor = describe : each uses its cross-validated Describe head in A, B, C
Heads from final_figures/qwen35_nothink/results/selected_heads.json. LSTMs have one map.
Scores in the anchor task equal the paper's CV-head table (the summary there is the
full-sample score at the modal CV head); scores in the other task are the transferred head.

Outputs (final_figures/qwen35_nothink/onehead/):
  alignment_summary_{anchor}head.csv, model_between_task_{anchor}head.csv,
  fig1_{anchor}head.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB = os.path.join(ROOT, 'rebuttal')
NT = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink')
RES = os.path.join(NT, 'results')
OUT = os.path.join(NT, 'onehead'); os.makedirs(OUT, exist_ok=True)
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
import alignment_core as A
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
ceil = json.load(open(os.path.join(RES, 'ceilings.json')))
imgs = A.target_images()
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
TASKS = ['social', 'describe']


def maps(m, task, head):
    if head is None:
        return A.trained_maps(m, task, imgs)
    if m.endswith('Transformer'):
        return A.trained_maps(m, task, imgs, layer=head[0], head=head[1])
    return V.vlm_maps(m, task, layer=head[0], head=head[1])


for anchor in TASKS:
    frames, bt = [], []
    for m in MODELS:
        head = None if m.endswith('LSTM') else (sel[f'{m}_{anchor}']['layer'], sel[f'{m}_{anchor}']['head'])
        mp = {t: maps(m, t, head) for t in TASKS}
        for t in TASKS:
            frames.append(A.per_image_z(mp[t], t, imgs, m))
        rs = np.array([spearmanr(mp['social'][im].flatten(), mp['describe'][im].flatten())[0] for im in imgs
                       if im in mp['social'] and im in mp['describe']
                       and mp['social'][im].std() > 1e-12 and mp['describe'][im].std() > 1e-12])
        hl = '' if head is None else f'L{head[0]}H{head[1]}'
        bt.append(dict(model=m, between_r=rs.mean(), sem=rs.std(ddof=1) / np.sqrt(len(rs)), n_img=len(rs),
                       task_diff_index=1 - rs.mean(), head=hl))
        print(anchor, m, hl, f'between={rs.mean():.3f}', flush=True)
    summ = A.agg(pd.concat(frames, ignore_index=True))
    summ['pct_ceiling'] = summ.apply(lambda r: 100 * np.arctanh(r['mean_r']) / ceil[r['task']], axis=1)
    summ.to_csv(os.path.join(OUT, f'alignment_summary_{anchor}head.csv'), index=False)
    pd.DataFrame(bt).to_csv(os.path.join(OUT, f'model_between_task_{anchor}head.csv'), index=False)

    # Figure: the paper's plotting code with the two data tables swapped in
    src_path = os.path.join(REB, 'plots', 'plot_fig_head_selection_corrected.py')
    src = open(src_path).read()
    for old, new in [("os.path.join(DATA, 'alignment_summary_cv_honest.csv')", repr(os.path.join(OUT, f'alignment_summary_{anchor}head.csv'))),
                     ("os.path.join(T1, 'selected_heads.json')", repr(os.path.join(RES, 'selected_heads.json'))),
                     ("os.path.join(T2, 'model_between_task.csv')", repr(os.path.join(OUT, f'model_between_task_{anchor}head.csv'))),
                     ("os.path.join(HERE, 'head_selection', 'fig_head_selection_corrected.png')",
                      repr(os.path.join(OUT, f'fig1_{anchor}head.png')))]:
        assert old in src, old
        src = src.replace(old, new)
    exec(src, {'__file__': src_path, '__name__': 'fig1_onehead'})

# side-by-side table
rows = []
for anchor in TASKS:
    s = pd.read_csv(os.path.join(OUT, f'alignment_summary_{anchor}head.csv')).pivot(index='model', columns='task', values='pct_ceiling')
    b = pd.read_csv(os.path.join(OUT, f'model_between_task_{anchor}head.csv')).set_index('model')
    rows.append(pd.DataFrame({'anchor': anchor, 'head': b['head'], 'soc_pct': s['social'], 'des_pct': s['describe'],
                              'between': b['between_r']}))
tab = pd.concat(rows).reset_index()
tab.to_csv(os.path.join(OUT, 'onehead_comparison.csv'), index=False)
print(tab.round(3).to_string(index=False))
