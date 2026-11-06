"""
nsd_diag.py -- why do the VLM attention bands predict NSD responses so differently by task?

1. Rebuild the Molmo cvbehav bands from the raw per-image maps and check they equal the
   files the encoding used (Social L10H24 was never rebuilt; Describe L10H27 was).
2. For every model x task band the cvbehav encoding uses: per-image attention mass
   (sum over the 196 locations) and its spread across images, the within-image spatial
   similarity between the Social and Describe bands, and how similar the two bands are
   after the per-location z-scoring the encoding model applies.
3. Molmo and Qwen: the same with every map scaled to sum to 1, to see how much of each band
   is "how much attention went to the image" rather than "where".
"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

NSD = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/rebuttal/task8_nsd')
spec = importlib.util.spec_from_file_location('fe', NSD / 'fit_encoding.py')
fe = importlib.util.module_from_spec(spec); spec.loader.exec_module(fe)

ids_all = pd.read_csv(NSD / 'stimuli_index.csv')['nsdId'].to_numpy()
def rebuild(task, L, H):
    keep, al = [], []
    for i in ids_all:
        p = NSD / 'vlm_out' / f'molmo_{task}' / f'attnmap_img_{int(i):05d}.pt'
        if p.exists():
            al.append(torch.load(p, map_location='cpu')[L - 1, H - 1].float().numpy()); keep.append(int(i))
    return np.array(keep), np.stack(al)

print('== 1. Molmo bands rebuilt from raw maps ==')
for f, task, L, H in [('attn/Molmo_social_cvbehav.npz', 'social', 10, 24),
                      ('_superseded_molmo_area/attn/Molmo_describe_cvbehav.npz', 'describe', 8, 7),
                      ('attn/Molmo_describe_cvbehav.npz', 'describe', 10, 27)]:
    d = np.load(NSD / f); i1, a1 = rebuild(task, L, H)
    print(f'{f:58s} L{L}H{H}  n={len(d["nsdId"])}  ids equal={np.array_equal(d["nsdId"], i1)}  '
          f'max|diff|={float(np.abs(d["alphas"] - a1).max()):.2e}')
# Molmo L10H27 under the Social prompt (the one-head rule) for comparison
_, mol_s27 = rebuild('social', 10, 27)

order = np.load(NSD / 'responses' / 'shared_nsdIds_subj01.npy')
M = ['CNN_LSTM', 'ViT_LSTM', 'CNN_Transformer', 'ViT_Transformer', 'Molmo', 'Qwen35']
X = {(m, t): fe.load_attention(m, t, order, 'cvbehav') for m in M for t in ('social', 'describe')}

def z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-12)
def compare(a, b):
    sp = np.nanmean([spearmanr(a[i], b[i])[0] for i in range(len(a))])
    loc = float(np.mean((z(a) * z(b)).mean(0)))          # mean per-location correlation across images
    return sp, loc

print('\n== 2. bands as the encoding sees them (shared-1000, subj01 order) ==')
print(f'{"model":16s} {"task":8s} {"n_ok":>5s} {"mass mean":>10s} {"mass CV":>8s}')
for m in M:
    for t in ('social', 'describe'):
        x, ok = X[(m, t)]; s = x[ok].sum(1)
        print(f'{m:16s} {t:8s} {ok.sum():5d} {s.mean():10.4f} {s.std() / s.mean():8.3f}')
    (a, oka), (b, okb) = X[(m, 'social')], X[(m, 'describe')]
    k = oka & okb
    sp, loc = compare(a[k], b[k])
    print(f'   social vs describe: within-image spatial Spearman r = {sp:.3f}; '
          f'per-location across-image r (after z-scoring) = {loc:.3f}')

print('\n== 3. VLMs with each map scaled to sum to 1 (location only, mass removed) ==')
for m in ('Molmo', 'Qwen35'):
    (a, oka), (b, okb) = X[(m, 'social')], X[(m, 'describe')]
    k = oka & okb
    an, bn = a[k] / a[k].sum(1, keepdims=True), b[k] / b[k].sum(1, keepdims=True)
    sp, loc = compare(an, bn)
    ma, mb = a[k].sum(1), b[k].sum(1)
    print(f'{m}: normalized social vs describe per-location r = {loc:.3f}; '
          f'corr of per-image mass social vs describe = {np.corrcoef(ma, mb)[0, 1]:.3f}')
    for t, x in (('social', a[k]), ('describe', b[k])):
        mass = x.sum(1, keepdims=True)
        # share of across-image variance in the z-scored band explained by the mass alone
        zx = z(x); zm = (mass - mass.mean()) / mass.std()
        r = (zx * zm).mean(0)
        print(f'   {t:8s}: mean corr(location, image mass) = {r.mean():.3f}  (|r|>0.5 at {np.mean(np.abs(r) > 0.5):.0%} of locations)')

# Molmo: same head under both prompts on NSD images
ids_m = np.load(NSD / 'attn' / 'Molmo_describe_cvbehav.npz')['nsdId']
d27 = np.load(NSD / 'attn' / 'Molmo_describe_cvbehav.npz')['alphas']
s24 = np.load(NSD / 'attn' / 'Molmo_social_cvbehav.npz')['alphas']
print('\nMolmo NSD, same images:')
for lab, a, b in (('social L10H24 vs describe L10H27', s24, d27), ('social L10H27 vs describe L10H27', mol_s27, d27),
                  ('social L10H24 vs social L10H27', s24, mol_s27)):
    sp, loc = compare(a, b)
    print(f'   {lab:36s} within-image spatial r = {sp:.3f}; per-location across-image r = {loc:.3f}; '
          f'mass corr = {np.corrcoef(a.sum(1), b.sum(1))[0, 1]:.3f}')
