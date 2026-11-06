"""
nsd_molmo_prepare.py
--------------------
Point Molmo's NSD encoding band at the heads cross-validated on the bilinear behavioural
maps (final_figures/molmo_bilinear/results/selected_heads.json). For each task whose head
changed, the old cvbehav features, permutation floor and encoding fits are moved to
task8_nsd/_superseded_molmo_area/, the band is rebuilt exactly as build_molmo_cvbehav.py
does (the (28, 28, 196) per-image maps under vlm_out/molmo_{task}/, one head), and the
floor is rebuilt as make_shuffled_permodel.py does (same seed). Prints the tasks to refit.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
SEL = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/'
           'molmo_bilinear/results/selected_heads.json')
OLD = {'social': (10, 24), 'describe': (8, 7)}       # heads behind the current NSD fits
sys.path.insert(0, str(NSD))
import make_shuffled_permodel as MSP

sel = json.load(open(SEL))
new = {t: (sel[f'Molmo_{t}']['layer'], sel[f'Molmo_{t}']['head']) for t in OLD}
changed = [t for t in OLD if new[t] != OLD[t]]
print('old', OLD, 'new', new, 'refit:', changed, flush=True)

BAK = NSD / '_superseded_molmo_area'
ids_all = pd.read_csv(NSD / 'stimuli_index.csv')['nsdId'].to_numpy()
for task in changed:
    for sub in ('attn', 'encoding'):
        (BAK / sub).mkdir(parents=True, exist_ok=True)
    for f in [NSD / 'attn' / f'Molmo_{task}_cvbehav.npz', NSD / 'attn' / f'ShufMolmo_{task}.npz']:
        if f.exists():
            shutil.move(str(f), str(BAK / 'attn' / f.name))
    for f in list((NSD / 'encoding').glob(f'r2_Molmo_{task}_subj*_cvbehav.npz')) + \
             list((NSD / 'encoding').glob(f'r2_ShufMolmo_{task}_subj*_cvbehav.npz')):
        shutil.move(str(f), str(BAK / 'encoding' / f.name))

    L, H = new[task]
    keep, alphas = [], []
    d = NSD / 'vlm_out' / f'molmo_{task}'
    for nsd_id in ids_all:
        p = d / f'attnmap_img_{int(nsd_id):05d}.pt'
        if p.exists():
            alphas.append(torch.load(p, map_location='cpu').float().numpy()[L - 1, H - 1])
            keep.append(int(nsd_id))
    np.savez_compressed(NSD / 'attn' / f'Molmo_{task}_cvbehav.npz',
                        nsdId=np.array(keep), alphas=np.stack(alphas).astype(np.float32))
    ids, feats = MSP.cvbehav_features('Molmo', task)
    perm = np.random.default_rng(MSP.SEED).permutation(len(ids))
    np.savez_compressed(NSD / 'attn' / f'ShufMolmo_{task}.npz', nsdId=ids,
                        alphas=np.asarray(feats, np.float32)[perm],
                        caption=np.array([''] * len(ids), dtype=object))
    print(f'Molmo {task}: L{L}H{H} band n={len(keep)}, floor rebuilt', flush=True)

BAK.mkdir(exist_ok=True)
open(BAK / 'REFIT_TASKS.txt', 'w').write(' '.join(changed))
