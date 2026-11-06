"""
fit_variants.py SUBJ -- attention-only encoding fits (fit_encoding.fit_single, corr2, same
vertex mask and image order) for controlled variants of the VLM bands, summarized by ROI
exactly as summarize_by_roi.py does (mean over the ROI's vertices).

  Molmo social L10H24 / describe L10H27   the bands in the paper (reproduction check)
  Molmo social L10H27                     Describe's head under the Social prompt
  Molmo describe L8H7                     the previous Describe head
  *_norm                                  every map scaled to sum to 1 (where, not how much)
  *_mass                                  the per-image total attention alone (1 feature)
  Qwen35 social / describe (+ _norm, _mass)
"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('fe', NSD / 'fit_encoding.py')
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)
subj = int(sys.argv[1])

Y_full = np.load(NSD / 'responses' / f'Y_shared_subj{subj:02d}.npy')
order = np.load(NSD / 'responses' / f'shared_nsdIds_subj{subj:02d}.npy')
nc = np.load(NSD / 'responses' / 'noise_ceiling.npz')
keep = np.where((nc['ncsnr'] > 0.2) & nc['valid'])[0]
Y = Y_full[:, keep]
masks = np.load(NSD / 'rois' / 'roi_masks_fs5.npz')
ROIS = {'early': 'grp_early', 'ventral': 'grp_ventral', 'lateral': 'grp_lateral', 'parietal': 'grp_parietal'}

def npz(path):
    d = np.load(path); return fe._align(d['nsdId'], d['alphas'].astype(np.float32), order)

bands = {
    'Molmo social L10H24': fe.load_attention('Molmo', 'social', order, 'cvbehav'),
    'Molmo describe L10H27': fe.load_attention('Molmo', 'describe', order, 'cvbehav'),
    'Molmo social L10H27': npz(HERE / 'Molmo_social_L10H27.npz'),
    'Molmo describe L8H7': npz(NSD / '_superseded_molmo_area' / 'attn' / 'Molmo_describe_cvbehav.npz'),
    'Qwen35 social L15H12': fe.load_attention('Qwen35', 'social', order, 'cvbehav'),
    'Qwen35 describe L7H9': fe.load_attention('Qwen35', 'describe', order, 'cvbehav'),
}
variants = {}
for k, (X, ok) in bands.items():
    assert ok.all(), (k, ok.sum())
    variants[k] = X
    if k.startswith(('Molmo social L10H24', 'Molmo describe L10H27', 'Qwen')):
        m = X.sum(1, keepdims=True)
        variants[k + ' norm'] = X / m
        variants[k + ' mass'] = m

rows = []
for k, X in variants.items():
    r2 = fe.fit_single(X, Y)
    for lab, roi in ROIS.items():
        sel = masks[roi][keep]
        rows.append(dict(subject=subj, variant=k, roi=lab, r2_attn=float(r2[sel].mean())))
    print(f'{k:32s} ' + '  '.join(f'{r["roi"]}={r["r2_attn"]:.4f}' for r in rows[-4:]), flush=True)
pd.DataFrame(rows).to_csv(HERE / f'variants_subj{subj:02d}.csv', index=False)
