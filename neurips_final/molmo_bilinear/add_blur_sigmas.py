"""
add_blur_sigmas.py
------------------
Add sigma = 40 and 56 px to the bilinear blur sweep (qwen35_nothink/blur_sweep.py swept
0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64), so the sweep has every multiple of 8 from 0 to 64.
Same maps, heads and scoring as blur_sweep.py; rows are appended to
final_figures/molmo_bilinear_pairwise/blur_sweep/blur_sweep_perimage.parquet.
"""
import os, sys, json
import numpy as np
import pandas as pd
import cv2

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
REB = os.path.join(ROOT, 'rebuttal')
FF = os.path.join(ROOT, 'neurips_final', 'final_figures', 'molmo_bilinear_pairwise')
for sub in ('shared', 'task1_head_selection'):
    sys.path.insert(0, os.path.join(REB, sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alignment_core as A
import vlm_maps as V
import molmo_bilinear_maps as MB

PROBE = '/scratch/gpfs/GRAZIANO/CANS/vlm-probing/output'
V._QCFG = {t: dict(act=os.path.join(PROBE, d, 'qwen35_nothink', 'activations'),
                   mask=os.path.join(PROBE, d, 'qwen35_nothink', 'image_mask.pt'))
           for t, d in (('social', 'social_cap'), ('describe', 'coco'))}
MB.patch(V)
sel = json.load(open(os.path.join(FF, 'results', 'selected_heads.json')))
imgs = A.target_images()
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
NEW = [40, 56]


def maps(m, task):
    if m.endswith('LSTM'):
        return A.trained_maps(m, task, imgs)
    L, H = sel[f'{m}_{task}']['layer'], sel[f'{m}_{task}']['head']
    if m.endswith('Transformer'):
        return A.trained_maps(m, task, imgs, layer=L, head=H)
    return V.vlm_maps(m, task, layer=L, head=H)


pf = os.path.join(FF, 'blur_sweep', 'blur_sweep_perimage.parquet')
long = pd.read_parquet(pf)
long = long[~long.sigma.isin(NEW)]
frames = [long]
for task in ('social', 'describe'):
    for m in MODELS:
        base = maps(m, task)
        # sanity: sigma 0 recomputed here must equal the stored sweep
        chk = A.per_image_z(base, task, imgs, m)
        old = long[(long.task == task) & (long.model == m) & (long.sigma == 0)]
        assert np.isclose(chk.z.mean(), old.z.mean(), atol=1e-9), (task, m)
        for s in NEW:
            mp = {k: cv2.GaussianBlur(np.asarray(v, np.float32), (0, 0), s) for k, v in base.items()}
            df = A.per_image_z(mp, task, imgs, m); df['sigma'] = s
            frames.append(df[long.columns])
        print(f'{task} {m}: added', NEW, flush=True)
pd.concat(frames, ignore_index=True).to_parquet(pf)
print('DONE add_blur_sigmas', flush=True)
