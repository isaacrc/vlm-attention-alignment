"""
loo_ceiling.py
--------------
Compare the paper's pairwise noise ceiling (one participant's map vs another's) with a
leave-one-subject-out ceiling (each participant vs the rank-averaged map of all others).
The pairwise ceiling is not an upper bound for a smooth, group-like predictor; the LOO
ceiling is closer to one. Spearman on the canonical subject sets, Fisher-z averaged.
"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/shared')
import alignment_core as A

imgs = A.target_images(); out = {}
for task in ('social', 'describe'):
    R, valid, subs = A.load_human(task, imgs)
    pair, loo = [], []
    for j in range(len(imgs)):
        vi = [i for i in np.where(valid[:, j])[0] if R[i, j].std() > 0]
        if len(vi) < 3:
            continue
        V = R[vi, j].astype(np.float32)
        Vc = V - V.mean(1, keepdims=True); Vn = Vc / np.linalg.norm(Vc, axis=1, keepdims=True)
        C = Vn @ Vn.T; iu = np.triu_indices(len(vi), 1)
        pair.append(np.arctanh(np.clip(C[iu], -.999999, .999999)).mean())
        tot = V.sum(0); zs = []
        for k in range(len(vi)):
            m = rankdata((tot - V[k]) / (len(vi) - 1)).astype(np.float32)
            m -= m.mean(); m /= np.linalg.norm(m)
            zs.append(np.arctanh(np.clip(float(Vn[k] @ m), -.999999, .999999)))
        loo.append(np.mean(zs))
    out[task] = dict(pairwise_r=float(np.tanh(np.mean(pair))), loo_r=float(np.tanh(np.mean(loo))),
                     pairwise_z=float(np.mean(pair)), loo_z=float(np.mean(loo)))
    print(task, out[task], flush=True)
json.dump(out, open('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/qwen35_nothink/blur_sweep/ceilings_pairwise_vs_loo.json', 'w'), indent=2)
