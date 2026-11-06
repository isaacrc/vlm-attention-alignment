"""
entropy_transformer_test.py
---------------------------
Is CNN-Transformer's attention reliably more concentrated than ViT-Transformer's?
Per-image normalized entropy (same maps, heads and entropy_norm as the paper's entropy
values, regen_fig5_entropy_cvheads.py), paired across the 200 images, per task.
Also reports two checks closer to what the figure shows: entropy after min-max
normalization (the figure's colour scale) and the share of attention mass in the top 10%
of patches.
"""
import os
import numpy as np
from scipy import stats

SRC = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment/archive/ICLR/scripts/regen_fig5_entropy_cvheads.py'
src = open(SRC).read()
g = {'__file__': SRC, '__name__': 'ent_test'}
exec(compile(src[:src.index('# ── 1. entropy')], SRC, 'exec'), g)
to14, ent, norm = g['to_14x14'], g['entropy_norm'], g['normalize']
ids = [t.replace('.jpg', '') for t in g['target_imgs']]


def top10(v):
    v = np.maximum(v, 0); v = v / v.sum()
    return np.sort(v)[::-1][:20].sum()   # 20 of 196 patches


def paired(name, a, b):
    d = a - b
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    print(f'    {name:22s} CNN {a.mean():.3f}  ViT {b.mean():.3f}  diff {d.mean():+.3f}  '
          f'dz = {d.mean() / d.std(ddof=1):+.2f}  t({len(d) - 1}) = {t.statistic:.2f}  p = {t.pvalue:.2g}  '
          f'(Wilcoxon p = {w.pvalue:.2g}); ViT lower on {np.mean(d > 0):.0%} of images')


for task in ('social', 'describe'):
    m = {k: g['trans_maps'](k, task) for k in ('CNN-Transformer', 'ViT-Transformer')}
    common = [i for i in ids if all(i in m[k] for k in m)]
    v = {k: [to14(m[k][i]) for i in common] for k in m}
    print(f'{task} (n = {len(common)} images; heads ' +
          ', '.join(f'{k} L{g["HEADS"][k][task][0]}H{g["HEADS"][k][task][1]}' for k in m) + ')')
    paired('entropy (paper)', *(np.array([ent(x) for x in v[k]]) for k in m))
    paired('entropy, min-max', *(np.array([ent(norm(x)) for x in v[k]]) for k in m))
    paired('top-10% mass', *(np.array([top10(x) for x in v[k]]) for k in m))

# The three images shown in Figure 2
for task in ('social', 'describe'):
    for im in g['BLEND_IMGS']:
        print(task, im, '  '.join(f"{k}: H = {ent(to14(g['trans_maps'](k, task)[im])):.3f}"
                                   for k in ('CNN-Transformer', 'ViT-Transformer')))
