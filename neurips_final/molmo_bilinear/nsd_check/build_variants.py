"""Molmo L10H27 under the Social prompt on the NSD images (the Describe head, so the two
tasks can be compared on one head). Same reduction as build_molmo_cvbehav.py."""
from pathlib import Path
import numpy as np, pandas as pd, torch
NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
ids = pd.read_csv(NSD / 'stimuli_index.csv')['nsdId'].to_numpy()
keep, al = [], []
for i in ids:
    p = NSD / 'vlm_out' / 'molmo_social' / f'attnmap_img_{int(i):05d}.pt'
    if p.exists():
        al.append(torch.load(p, map_location='cpu')[9, 26].float().numpy()); keep.append(int(i))
np.savez_compressed(Path(__file__).parent / 'Molmo_social_L10H27.npz', nsdId=np.array(keep), alphas=np.stack(al))
print('saved n =', len(keep))
