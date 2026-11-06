"""
Task 3, Tier-B scoring.

Pushes every Tier-B baseline through the SAME pipeline as the models, on both
metrics, so the numbers drop straight into the existing tables:

  raw      Spearman r vs each subject's map, Fisher-z averaged  (A.per_image_z)
  partial  Spearman r controlling the leave-one-image-out center prior
           (run_center_controlled.partial_rows)

Each baseline is reported at its own best Gaussian sigma, chosen separately per
metric and per task -- the same courtesy the Tier-A center Gaussian got. Reporting
a baseline at anything less than its best sigma would understate it, which is the
wrong direction for a baseline whose job is to be hard to beat.

Outputs: results/tierB_baselines.csv        (best-sigma summary)
         results/tierB_sigma_sweep.csv      (full sweep, for the record)
         results/tierB_contrasts.csv        (baselines vs models, partial metric)
Figure:  figures/fig_tierB_baselines.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A
import tierB_common as T
import run_center_controlled as CC

RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)

SIGMAS = [0, 4, 8, 16, 24, 32]
BASELINES = ['coco_person', 'coco_face', 'coco_objects', 'coco_person_letterbox',
             'itti_koch', 'gbvs', 'spectral_residual',
             'deepgaze2e', 'deepgaze2e_uniform']
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer',
          'Molmo', 'Qwen3.5']


def blur(maps, s):
    if s == 0:
        return maps
    return {k: gaussian_filter(v, s) for k, v in maps.items()}


def main():
    imgs = A.target_images()
    print('building leave-one-image-out center prior...', flush=True)
    prior_rk = CC.loio_prior_ranked(imgs)

    raw_ceil = {t: A.ceiling(t) for t in ['social', 'describe']}
    par_ceil = {t: CC.partial_ceiling(t, imgs, prior_rk) for t in ['social', 'describe']}
    for t in raw_ceil:
        print(f'[{t}] raw ceiling z={raw_ceil[t]:.4f} (r={np.tanh(raw_ceil[t]):.4f}) | '
              f'partial z={par_ceil[t]:.4f} (r={np.tanh(par_ceil[t]):.4f})', flush=True)

    available = [b for b in BASELINES if T.has_maps(b)]
    missing = [b for b in BASELINES if not T.has_maps(b)]
    if missing:
        print(f'WARNING: missing maps, skipping: {missing}', flush=True)

    sweep, partial_long = [], []
    for name in available:
        maps = T.load_maps(name)
        for s in SIGMAS:
            bm = blur(maps, s)
            for task in ['social', 'describe']:
                df = A.per_image_z(bm, task, imgs, name)
                zr = float(A.subject_means(df)['mean_z'].mean())
                pdf = CC.partial_rows(bm, task, imgs, prior_rk, name)
                zp = float(pdf.groupby('subject_id')['z'].mean().mean())
                sweep.append(dict(baseline=name, task=task, sigma=s,
                                  raw_r=float(np.tanh(zr)), raw_z=zr,
                                  pct_raw_ceiling=100 * zr / raw_ceil[task],
                                  partial_r=float(np.tanh(zp)), partial_z=zp,
                                  pct_partial_ceiling=100 * zp / par_ceil[task]))
                partial_long.append(pdf.assign(sigma=s))
        print(f'  {name} swept', flush=True)

    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(RES, 'tierB_sigma_sweep.csv'), index=False)

    # ── best sigma per baseline x task, chosen separately for each metric ──
    rows = []
    for (b, t), g in sw.groupby(['baseline', 'task']):
        br = g.loc[g['raw_z'].idxmax()]
        bp = g.loc[g['partial_z'].idxmax()]
        rows.append(dict(baseline=b, task=t,
                         raw_r=br['raw_r'], raw_sigma=int(br['sigma']),
                         pct_raw_ceiling=br['pct_raw_ceiling'],
                         partial_r=bp['partial_r'], partial_sigma=int(bp['sigma']),
                         pct_partial_ceiling=bp['pct_partial_ceiling']))
    best = pd.DataFrame(rows).sort_values(['task', 'partial_r'], ascending=[True, False])
    best.to_csv(os.path.join(RES, 'tierB_baselines.csv'), index=False)
    print('\n=== Tier-B baselines (each at its own best sigma) ===')
    print(best.to_string(index=False))

    # ── contrasts: each baseline vs each model, partial metric, image-wise ──
    cc_long = pd.read_csv(os.path.join(RES, 'center_controlled.csv'))
    pl = pd.concat(partial_long, ignore_index=True)
    rng = np.random.default_rng(0)
    tests = []
    for task in ['social', 'describe']:
        mod_pim = {m: cc_long[(cc_long.model == m) & (cc_long.task == task)]
                   .groupby('image_id')['z'].mean() for m in MODELS}
        for b in available:
            g = sw[(sw.baseline == b) & (sw.task == task)]
            s_best = int(g.loc[g['partial_z'].idxmax(), 'sigma'])
            bp = pl[(pl.model == b) & (pl.task == task) & (pl.sigma == s_best)]
            b_pim = bp.groupby('image_id')['z'].mean()
            for m in MODELS:
                x, y = mod_pim[m], b_pim
                c = x.index.intersection(y.index)
                if len(c) == 0:
                    continue
                d = (x.loc[c] - y.loc[c]).values
                obs = d.mean()
                null = (rng.choice([-1, 1], size=(10000, len(d))) * d).mean(axis=1)
                tests.append(dict(task=task, model=m, baseline=b, sigma=s_best,
                                  mean_diff=float(obs),
                                  dz=float(obs / (d.std(ddof=1) + 1e-12)),
                                  n_img=len(d),
                                  p_perm=float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001)))
    td = pd.DataFrame(tests)
    td.to_csv(os.path.join(RES, 'tierB_contrasts.csv'), index=False)
    print('\n=== model vs baseline (partial metric; positive = model wins) ===')
    print(td.to_string(index=False))

    # ── figure ──
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cc_sum = pd.read_csv(os.path.join(RES, 'center_controlled_summary.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, task in zip(axes, ['social', 'describe']):
        ms = cc_sum[(cc_sum.task == task) & (cc_sum.model.isin(MODELS))]
        bs = best[best.task == task]
        items = ([(r.model, r.r_partial, '#4c72b0') for r in ms.itertuples()] +
                 [(r.baseline, r.partial_r, '#c44e52') for r in bs.itertuples()])
        items.sort(key=lambda x: -x[1])
        xs = np.arange(len(items))
        ax.bar(xs, [i[1] for i in items], color=[i[2] for i in items])
        ax.axhline(np.tanh(par_ceil[task]), ls='--', color='gray',
                   label='human ceiling (partial)')
        ax.set_xticks(xs); ax.set_xticklabels([i[0] for i in items], rotation=60, ha='right')
        ax.set_title(f'{task.capitalize()} -- center-controlled alignment')
        ax.set_ylabel('partial Spearman r')
        ax.legend(fontsize=8)
    fig.suptitle('Models (blue) vs Tier-B baselines (red), center bias removed')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_tierB_baselines.png'), dpi=150)
    print('\nSaved figure.')


if __name__ == '__main__':
    main()
