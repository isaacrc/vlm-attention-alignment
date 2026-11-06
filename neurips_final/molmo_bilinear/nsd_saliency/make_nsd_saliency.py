"""
make_nsd_saliency.py -- exploratory, not in the manuscript.
Itti-Koch, GBVS and spectral residual maps for the NSD stimuli, with the implementations
used for the behavioural baselines (rebuttal/task3_baselines/make_maps_saliency.py),
reduced to the 14x14 grid of the encoding analysis the way the low-level controls are
(INTER_AREA). Writes task8_nsd/attn/Sal{Itti,GBVS,SR}_{social,describe}.npz (identical
across tasks, like the low-level controls).
"""
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
sys.path.insert(0, str(ROOT / 'rebuttal' / 'task3_baselines'))
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import make_maps_saliency as S

ids = pd.read_csv(NSD / 'stimuli_index.csv')['nsdId'].to_numpy()
F = S._spatial_falloff(32)
def to14(m):
    return cv2.resize(np.asarray(m, np.float32), (14, 14), interpolation=cv2.INTER_AREA).ravel()

def one(i):
    rgb = np.asarray(Image.open(NSD / 'stimuli' / f'{int(i):05d}.png').convert('RGB'))
    return to14(S.itti_koch(rgb)), to14(S.gbvs(rgb, F=F)), to14(S.spectral_residual(rgb))

if __name__ == '__main__':
    with Pool(8) as p:
        res = p.map(one, ids, chunksize=8)
    for k, name in enumerate(('SalItti', 'SalGBVS', 'SalSR')):
        X = np.stack([r[k] for r in res]).astype(np.float32)
        print(name, X.shape, 'min', X.min(), 'finite', np.isfinite(X).all(), flush=True)
        for task in ('social', 'describe'):
            np.savez_compressed(NSD / 'attn' / f'{name}_{task}.npz', nsdId=ids, alphas=X,
                                caption=np.array([''] * len(ids), dtype=object))
