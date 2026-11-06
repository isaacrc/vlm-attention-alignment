"""
map_rsa.py -- exploratory, not in the manuscript.
How similar are the predictors' spatial maps to one another? For every pair of predictors
(six models, three saliency models, edge energy, luminance), the per-image correlation
between their 14x14 maps, Fisher-z averaged over images. Spearman and Pearson, Social,
Describe and their average (Fisher-z mean of the two tasks).

  --part nsd   the 196-dim maps that enter the NSD encoding model (1,098 NSD images)
  --part coco  the 200 behavioural images: the maps of the manuscript analyses (bilinear
               Molmo, thinking-off Qwen, cross-validated heads, no blur), each 256x256 map
               reduced to 14x14 with INTER_AREA as in the encoding analysis; edge energy and
               luminance computed as make_lowlevel_baselines.py does, on the 256x256 image

Writes final_figures/molmo_bilinear/nsd_saliency/map_rsa_{part}.png and
map_rsa_{part}_{spearman,pearson}_{social,describe,average}.csv
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_saliency'
LABELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-T', 'ViT-T', 'Molmo', 'Qwen3.5',
          'Itti-Koch', 'GBVS', 'Spec. res.', 'Edge', 'Luminance']
TASKS = ('social', 'describe')


# ── NSD maps: exactly what fit_encoding._load_cvbehav feeds the ridge ──────────────────
CVH = {'CNN_Transformer': {'social': (4, 4), 'describe': (3, 5)},
       'ViT_Transformer': {'social': (4, 8), 'describe': (3, 8)},
       'Qwen35': {'social': (15, 12), 'describe': (7, 9)}}
NSD_KEYS = ['CNN_LSTM', 'ViT_LSTM', 'CNN_Transformer', 'ViT_Transformer', 'Molmo', 'Qwen35',
            'SalItti', 'SalGBVS', 'SalSR', 'EdgeEnergy', 'Luminance']


def nsd_load(m, t):
    A = NSD / 'attn'
    if m in ('CNN_Transformer', 'ViT_Transformer'):
        d = np.load(A / f'{m}_{t}_alllayers.npz', allow_pickle=True); L, h = CVH[m][t]
        f = d['heads_all'][:, L - 1, h - 1]
    elif m == 'Molmo':
        d = np.load(A / f'Molmo_{t}_cvbehav.npz', allow_pickle=True); f = d['alphas']
    elif m == 'Qwen35':
        d = np.load(A / f'Qwen35_{t}.npz', allow_pickle=True); f = d['heads'][:, CVH[m][t][1] - 1]
    else:
        d = np.load(A / f'{m}_{t}.npz', allow_pickle=True); f = d['alphas']
    return dict(zip(d['nsdId'].astype(int), np.asarray(f, np.float64)))


def nsd_maps(t):
    return [nsd_load(m, t) for m in NSD_KEYS]


# ── 200 behavioural images ─────────────────────────────────────────────────────────────
def coco_setup():
    """Loaders of the manuscript figures (plot_fig2_best_blur.py), without the blur."""
    SRC = ROOT / 'archive' / 'ICLR' / 'scripts' / 'regen_fig5_entropy_cvheads.py'
    FF = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear_pairwise'
    SEL = json.load(open(FF / 'results' / 'selected_heads.json'))
    src = open(SRC).read()
    g = {'__file__': str(SRC), '__name__': 'map_rsa'}
    exec(compile(src[:src.index('# ── 1. entropy')], str(SRC), 'exec'), g)
    for m in ('CNN-Transformer', 'ViT-Transformer', 'Qwen3.5', 'Molmo'):
        g['HEADS'][m] = {t: (SEL[f'{m}_{t}']['layer'], SEL[f'{m}_{t}']['head']) for t in TASKS}
    P = g['PROBING_DIR']
    g['QCFG'] = {'social': (os.path.join(P, 'output', 'social_cap', 'qwen35_nothink'),),
                 'describe': (os.path.join(P, 'output', 'coco', 'qwen35_nothink'),)}
    torch = g['torch']

    def molmo_maps(task, only):
        L, H = g['HEADS']['Molmo'][task]
        act_dir = os.path.join(g['MOLMO_DIR'], 'activations', task)
        grid36 = g['_molmo_grid'](act_dir)[0]
        vidx = np.where(grid36 > 0); pos = grid36[vidx]
        rows, cols = np.any(grid36 > 0, axis=1), np.any(grid36 > 0, axis=0)
        out = {}
        for im in only:
            p = os.path.join(act_dir, f'activations_img_{im}.pt')
            if not os.path.exists(p):
                continue
            a = torch.load(p, map_location='cpu')[L - 1, H - 1].float().numpy()
            vol = np.zeros((a.shape[0],) + grid36.shape, np.float32)
            vol[:, vidx[0], vidx[1]] = a[:, pos]
            out[im] = cv2.resize(vol.sum(0)[rows][:, cols], (256, 256), interpolation=cv2.INTER_LINEAR)
        return out
    g['molmo_maps2'] = molmo_maps
    return g


def coco_maps(t, g):
    sys.path.insert(0, str(ROOT / 'rebuttal' / 'task3_baselines'))
    sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
    import tierB_common as T, alignment_core as A
    imgs = [i.replace('.jpg', '') for i in A.target_images()]
    to14 = lambda m: cv2.resize(np.asarray(m, np.float32), (14, 14), interpolation=cv2.INTER_AREA).ravel()
    out = [g['lstm_maps']('CNN-LSTM', t), g['lstm_maps']('ViT-LSTM', t),
           g['trans_maps']('CNN-Transformer', t), g['trans_maps']('ViT-Transformer', t),
           g['molmo_maps2'](t, imgs), g['qwen_maps'](t, only=imgs)]
    for name in ('itti_koch', 'gbvs', 'spectral_residual'):
        out.append({k.replace('.jpg', ''): v for k, v in T.load_maps(name).items()})
    edge, lum = {}, {}
    for im in imgs:
        gimg = cv2.cvtColor(T.load_rgb(im + '.jpg'), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gimg = cv2.resize(gimg, (256, 256), interpolation=cv2.INTER_AREA)
        lum[im] = gimg
        edge[im] = np.sqrt(cv2.Sobel(gimg, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                           cv2.Sobel(gimg, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    out += [edge, lum]
    return [{k: to14(v).astype(np.float64) for k, v in d.items()} for d in out]


# ── correlation matrices ───────────────────────────────────────────────────────────────
def corr_matrix(maps, rank):
    ids = sorted(set.intersection(*[set(d) for d in maps]))
    X = []
    for d in maps:
        a = np.stack([d[i] for i in ids])
        if rank:
            a = np.apply_along_axis(rankdata, 1, a)
        a = a - a.mean(1, keepdims=True)
        X.append(a / np.linalg.norm(a, axis=1, keepdims=True))
    n = len(maps); Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = np.clip((X[i] * X[j]).sum(1), -0.999999, 0.999999)
            Z[i, j] = np.nanmean(np.arctanh(r))
    return Z, len(ids)        # Fisher z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', choices=['nsd', 'coco'], required=True)
    part = ap.parse_args().part
    if part == 'coco':
        g = coco_setup()
        maps = {t: coco_maps(t, g) for t in TASKS}
    else:
        maps = {t: nsd_maps(t) for t in TASKS}

    res, nimg = {}, None
    for meth, rank in (('spearman', True), ('pearson', False)):
        z = {}
        for t in TASKS:
            z[t], nimg = corr_matrix(maps[t], rank)
        z['average'] = (z['social'] + z['describe']) / 2
        for k, v in z.items():
            r = np.tanh(v); np.fill_diagonal(r, 1.0)
            res[(meth, k)] = r
            pd.DataFrame(r, index=LABELS, columns=LABELS).round(3).to_csv(OUT / f'map_rsa_{part}_{meth}_{k}.csv')

    # ── figure: rows Spearman / Pearson, columns Social / Describe / Average ──
    n = len(LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(22, 14.5))
    lim = 0.8
    for r, meth in enumerate(('spearman', 'pearson')):
        for c, col in enumerate(('social', 'describe', 'average')):
            ax = axes[r, c]
            M = res[(meth, col)].copy(); np.fill_diagonal(M, np.nan)
            im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_facecolor('#e6e6e6')
            for i in range(n):
                for j in range(n):
                    if i != j:
                        v = M[i, j]
                        ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=8.5,
                                color='white' if abs(v) > 0.5 else '#222222')
            for b in (5.5, 8.5):     # models | saliency | low-level
                ax.axhline(b, color='black', lw=1.5); ax.axvline(b, color='black', lw=1.5)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(LABELS, rotation=55, ha='right', fontsize=11)
            ax.set_yticklabels(LABELS if c == 0 else [''] * n, fontsize=11)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col.capitalize(), fontsize=20, fontweight='bold', pad=12)
        axes[r, 0].set_ylabel(meth.capitalize() + ' $r$', fontsize=20, fontweight='bold', labelpad=12)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.94, bottom=0.08, wspace=0.06, hspace=0.32)
    cax = fig.add_axes([0.915, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=12)
    cb.set_label('Per-image map correlation', fontsize=14)
    where = '1,098 NSD images' if part == 'nsd' else '200 behavioral images'
    fig.suptitle(f'Map-to-map similarity ({where}; 14×14 maps)', fontsize=18, y=0.995)
    fig.savefig(OUT / f'map_rsa_{part}.png', dpi=200, bbox_inches='tight')
    print(f'{part}: {nimg} images')
    for meth in ('spearman', 'pearson'):
        print(f'\n{meth} average'); print(pd.DataFrame(res[(meth, "average")], index=LABELS, columns=LABELS).round(2).to_string())


if __name__ == '__main__':
    main()
