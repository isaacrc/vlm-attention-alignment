"""
map_rsa_priors.py -- exploratory, not in the manuscript.
As map_rsa.py --part coco (200 behavioural images, 14x14 maps, no blur), Spearman only,
averaged over the two tasks (Fisher z), with the three semantic/spatial priors of the
Baselines appendix added:
  Center  leave-one-image-out mean human fixation map (run_center_controlled.py)
  Face    Gaussian blob on every annotated face (tierB_maps/coco_face.npz)
  Person  union of annotated person masks (tierB_maps/coco_person.npz)
Each pair is averaged over the images on which both maps vary (the face prior is flat
on the images without a visible face, so its correlations use only the images with one).
Writes final_figures/molmo_bilinear/nsd_saliency/map_rsa_coco_priors.png + .csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import map_rsa as R

ROOT = R.ROOT
sys.path.insert(0, str(ROOT / 'rebuttal' / 'task3_baselines'))
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import tierB_common as T, alignment_core as A
import run_center_controlled as CC

LABELS = R.LABELS + ['Center', 'Face', 'Person']
to14 = lambda m: cv2.resize(np.asarray(m, np.float32), (14, 14), interpolation=cv2.INTER_AREA).ravel().astype(np.float64)


def prior_maps():
    imgs = A.target_images()
    _rank = A._rank
    A._rank = lambda x, axis=1: x            # take the raw center prior, not its ranks
    try:
        cen = CC.loio_prior_ranked(imgs)
    finally:
        A._rank = _rank
    s = int(round(np.sqrt(cen.shape[1])))
    center = {im.replace('.jpg', ''): to14(cen[k].reshape(s, s)) for k, im in enumerate(imgs)}
    face = {k.replace('.jpg', ''): to14(v) for k, v in T.load_maps('coco_face').items()}
    person = {k.replace('.jpg', ''): to14(v) for k, v in T.load_maps('coco_person').items()}
    return [center, face, person]


def pair_z(a, b):
    ids = sorted(set(a) & set(b))
    x = np.stack([rankdata(a[i]) for i in ids]); y = np.stack([rankdata(b[i]) for i in ids])
    x = x - x.mean(1, keepdims=True); y = y - y.mean(1, keepdims=True)
    den = np.sqrt((x * x).sum(1) * (y * y).sum(1))
    ok = den > 0                              # drop images where either map is flat
    r = np.clip((x[ok] * y[ok]).sum(1) / den[ok], -0.999999, 0.999999)
    return np.arctanh(r).mean(), int(ok.sum())


def main():
    g = R.coco_setup()
    pri = prior_maps()
    Z, N = {}, {}
    for t in R.TASKS:
        maps = R.coco_maps(t, g) + pri
        n = len(maps); z = np.zeros((n, n)); cnt = np.zeros((n, n), int)
        for i in range(n):
            for j in range(i + 1, n):
                z[i, j], cnt[i, j] = pair_z(maps[i], maps[j]); z[j, i], cnt[j, i] = z[i, j], cnt[i, j]
        Z[t], N[t] = z, cnt
    r = np.tanh((Z['social'] + Z['describe']) / 2); np.fill_diagonal(r, 1.0)
    df = pd.DataFrame(r, index=LABELS, columns=LABELS)
    df.round(3).to_csv(R.OUT / 'map_rsa_coco_priors_spearman_average.csv')
    print(df.round(2).to_string())
    print('\nimages per pair with the face prior:', sorted(set(N['social'][LABELS.index('Face')]) - {0}))

    n = len(LABELS)
    fig, ax = plt.subplots(figsize=(11.5, 10))
    M = r.copy(); np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    ax.set_facecolor('#e6e6e6')
    for i in range(n):
        for j in range(n):
            if i != j:
                v = M[i, j]
                ax.text(j, i, f'{v:.2f}'.replace('0.', '.'), ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > 0.5 else '#222222')
    for b in (5.5, 8.5, 10.5):   # models | saliency | low-level | priors
        ax.axhline(b, color='black', lw=1.5); ax.axvline(b, color='black', lw=1.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=55, ha='right', fontsize=12)
    ax.set_yticklabels(LABELS, fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Map-to-map Spearman $r$, average of Social and Describe\n(200 behavioral images; 14×14 maps)',
                 fontsize=15, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=11)
    cb.set_label('Per-image map correlation', fontsize=13)
    fig.savefig(R.OUT / 'map_rsa_coco_priors.png', dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()
