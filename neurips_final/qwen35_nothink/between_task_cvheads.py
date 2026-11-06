"""
between_task_cvheads.py
-----------------------
Figure 1C (between-task spatial alignment) on the cross-validated per-task heads
(selected_heads.json), so every model with heads uses the same head in panels A-C.
run_between_task.py used fixed heads instead: L3H5 / L3H8 for both tasks for the
Transformers (alignment_core best_head) and the in-sample argmax for Molmo
(L10H24 / L10H27). LSTMs have one map and are unchanged.

Output: final_figures/qwen35_nothink/results/model_between_task_cvheads.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB = os.path.join(ROOT, 'rebuttal')
RES = os.path.join(ROOT, 'neurips_final', 'final_figures', 'qwen35_nothink', 'results')
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
import alignment_core as A
import vlm_maps as V

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
sel = json.load(open(os.path.join(RES, 'selected_heads.json')))
imgs = A.target_images()


def maps(m, task):
    if m.endswith('LSTM'):
        return A.trained_maps(m, task, imgs)
    L, H = sel[f'{m}_{task}']['layer'], sel[f'{m}_{task}']['head']
    if m.endswith('Transformer'):
        return A.trained_maps(m, task, imgs, layer=L, head=H)
    return V.vlm_maps(m, task, layer=L, head=H)


rows = []
for m in ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']:
    soc, des = maps(m, 'social'), maps(m, 'describe')
    rs = np.array([spearmanr(soc[im].flatten(), des[im].flatten())[0] for im in imgs
                   if im in soc and im in des and soc[im].std() > 1e-12 and des[im].std() > 1e-12])
    heads = '' if m.endswith('LSTM') else '/'.join(
        f"L{sel[f'{m}_{t}']['layer']}H{sel[f'{m}_{t}']['head']}" for t in ('social', 'describe'))
    rows.append(dict(model=m, between_r=rs.mean(), sem=rs.std(ddof=1) / np.sqrt(len(rs)),
                     n_img=len(rs), task_diff_index=1 - rs.mean(), heads=heads))
    print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, 'model_between_task_cvheads.csv'), index=False)
old = pd.read_csv(os.path.join(RES, 'model_between_task.csv'))
print(old[['model', 'between_r', 'sem']].merge(out[['model', 'between_r', 'sem', 'heads']], on='model',
                                              suffixes=('_fixed', '_cv')).to_string(index=False))

