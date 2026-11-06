"""Permutation floors for the three saliency maps, built exactly as
rebuttal/task8_nsd/make_shuffled_permodel.py builds them for the other predictors:
the predictor's own maps, image labels permuted with seed 42.
Output: rebuttal/task8_nsd/attn/ShufSal{Itti,GBVS,SR}_{task}.npz
"""
from pathlib import Path
import numpy as np

ATTN = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd/attn')
SEED = 42

for task in ('social', 'describe'):
    for model in ('SalItti', 'SalGBVS', 'SalSR'):
        d = np.load(ATTN / f'{model}_{task}.npz', allow_pickle=True)
        ids, feats = d['nsdId'], np.asarray(d['alphas'], dtype=np.float32)
        perm = np.random.default_rng(SEED).permutation(len(ids))
        out = ATTN / f'Shuf{model}_{task}.npz'
        np.savez_compressed(out, nsdId=ids, alphas=feats[perm],
                            caption=np.array([''] * len(ids), dtype=object))
        print(f'{out.name}: {feats.shape}, {(perm != np.arange(len(ids))).mean():.1%} displaced')
