"""
nsd_prepare.py
--------------
Swap the thinking-off Qwen3.5 attention into the NSD encoding pipeline.

  1. Move every thinking-on Qwen35 artefact (features, floors, encoding fits, and the
     per-subject tables they fed) to task8_nsd/_superseded_qwen35_thinking/, so
     fit_encoding.py refits instead of skipping and nothing stale is mixed in.
  2. Build attn/Qwen35_{task}.npz from vlm_out/qwen35_nothink_{task}/ with the reduction
     of build_vlm_features.build_qwen (mean over generated tokens, image tokens only,
     14x14 INTER_AREA), keeping all 16 heads of the layer that holds the behaviourally
     selected head (fit_encoding.CVBEHAV_HEADS).
  3. Rebuild the Qwen35 permutation floor exactly as make_shuffled_permodel.py does
     (same seed, same cvbehav feature selection).
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
SEL = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/'
           'qwen35_nothink/results/selected_heads.json')
sys.path.insert(0, str(NSD))
from fit_encoding import CVBEHAV_HEADS
import make_shuffled_permodel as MSP

BAK = NSD / '_superseded_qwen35_thinking'
for sub in ('attn', 'encoding', 'results'):
    (BAK / sub).mkdir(parents=True, exist_ok=True)

# sanity: the NSD head dict must match the behavioural selection
sel = json.load(open(SEL))
for t in ('social', 'describe'):
    assert CVBEHAV_HEADS['Qwen35'][t] == (sel[f'Qwen3.5_{t}']['layer'], sel[f'Qwen3.5_{t}']['head']), t

# ── 1. back up ────────────────────────────────────────────────────────────────
for f in list((NSD / 'attn').glob('*Qwen35_*.npz')) + list((NSD / 'encoding').glob('r2_*Qwen35_*.npz')) \
         + list((NSD / 'encoding').glob('weights_*Qwen35_*.npz')):
    dest = BAK / f.parent.name / f.name
    if not dest.exists():
        shutil.move(str(f), dest)
    else:
        f.unlink()
for name in ('attnonly_per_subject_cvbehav.csv', 'attnonly_contrasts_cvbehav.csv',
             'encoding_by_roi_per_subject_cvbehav.csv', 'encoding_by_roi_cvbehav.csv',
             'permodel_per_subject_cvbehav.csv', 'permodel_vs_floor_cvbehav.csv'):
    src = NSD / 'results' / name
    if src.exists() and not (BAK / 'results' / name).exists():
        shutil.copy2(src, BAK / 'results' / name)
print('backed up thinking-on Qwen35 files to', BAK)

# ── 2. features ───────────────────────────────────────────────────────────────
ids = pd.read_csv(NSD / 'stimuli_index.csv')['nsdId'].to_numpy()
for task in ('describe', 'social'):
    layer = CVBEHAV_HEADS['Qwen35'][task][0]
    base = NSD / 'vlm_out' / f'qwen35_nothink_{task}'
    act = base / 'activations' / f'attn-layer-{layer}'
    base_mask = torch.load(base / 'image_mask.pt', map_location='cpu')
    base_mask = base_mask.squeeze(0) if base_mask.dim() == 2 else base_mask
    heads, alphas, keep = [], [], []
    for nsd_id in ids:
        p = act / f'{int(nsd_id):05d}.pt'
        if not p.exists():
            continue
        data = torch.load(p, map_location='cpu').squeeze(0).float()     # (16, n_gen, seq)
        pad = data.shape[-1] - base_mask.shape[0]
        mask = F.pad(base_mask, (0, pad), value=False) if pad > 0 else base_mask[:data.shape[-1]]
        avg = data.nanmean(dim=1)                                        # (16, seq)
        masked = avg[:, mask][:, 1:-1]                                   # drop vision_start/end
        side = int(np.sqrt(masked.shape[1])); masked = masked[:, :side * side]
        ph = np.stack([cv2.resize(masked[h].numpy().reshape(side, side).astype(np.float32), (14, 14),
                                  interpolation=cv2.INTER_AREA).ravel() for h in range(masked.shape[0])])
        heads.append(ph); alphas.append(ph.mean(0)); keep.append(int(nsd_id))
    assert len(keep) == len(ids), f'{task}: only {len(keep)}/{len(ids)} images extracted'
    out = NSD / 'attn' / f'Qwen35_{task}.npz'
    np.savez_compressed(out, nsdId=np.array(keep), alphas=np.stack(alphas), heads=np.stack(heads),
                        layer=layer, caption=np.array([''] * len(keep), dtype=object))
    print(f'saved {out.name}: layer {layer}, heads {np.stack(heads).shape}, grid {side}x{side}')

# ── 3. permutation floor ──────────────────────────────────────────────────────
for task in ('social', 'describe'):
    nid, feats = MSP.cvbehav_features('Qwen35', task)
    feats = np.asarray(feats, dtype=np.float32)
    perm = np.random.default_rng(MSP.SEED).permutation(len(nid))
    out = NSD / 'attn' / f'ShufQwen35_{task}.npz'
    np.savez_compressed(out, nsdId=nid, alphas=feats[perm], caption=np.array([''] * len(nid), dtype=object))
    print(f'saved {out.name}: {feats.shape}')
