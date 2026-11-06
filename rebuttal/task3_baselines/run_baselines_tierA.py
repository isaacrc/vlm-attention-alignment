"""
Task 3 (Tier-A) — trivial baselines + the blur/entropy controls that answer the
"is the LSTM advantage just a smooth low-frequency prior?" objection. No downloads.

  - empirical center-bias prior (mean human map, applied to every image)
  - center Gaussian (best sigma)
  - blur sweep: blur each Transformer best-head map -> alignment vs sigma; does it
    reach LSTM alignment? Reverse: sharpen LSTM via temperature.
  - entropy-matched invariance demo (Spearman invariant to monotone temp scaling)

Outputs: results/baseline_alignment.csv, results/blur_sweep.csv
Figures: figures/fig_baselines.png, figures/fig_blur_sweep.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
import alignment_core as A

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data')
ceil = json.load(open(os.path.join(DATA, 'ceilings.json')))
summ = pd.read_csv(os.path.join(DATA, 'alignment_summary_best_head.csv'))


def raw_human_mean(imgs):
    """Mean raw human map across all subjects & both tasks -> center-bias prior (256x256)."""
    acc = np.zeros((256, 256), np.float64); cnt = 0
    for task in ['social', 'describe']:
        hm_dir = A.HM_DIR[task]
        for fn in os.listdir(hm_dir):
            if not fn.endswith('_heatmaps.npy'):
                continue
            if fn.replace('_heatmaps.npy', '') in A.EXCLUDE[task]:
                continue
            sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
            for im in imgs:
                hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
                if hm is not None:
                    h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                    acc += np.array(Image.fromarray(h).resize((256, 256), Image.NEAREST), np.float64)
                    cnt += 1
    return acc / max(cnt, 1)


def gaussian_map(sigma_frac):
    n = 256; c = n / 2; s = sigma_frac * n
    y, x = np.mgrid[0:n, 0:n]
    return np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2 * s * s)))


def align_of_constant_map(mp, task, imgs):
    maps = {im: mp for im in imgs}
    df = A.per_image_z(maps, task, imgs, 'baseline')
    z = A.subject_means(df)['mean_z'].values
    return float(np.tanh(z.mean())), float(z.mean())


def main():
    imgs = A.target_images()
    prior = raw_human_mean(imgs)

    rows = []
    for task in ['social', 'describe']:
        # center-bias prior
        r, z = align_of_constant_map(prior, task, imgs)
        rows.append(dict(baseline='center_bias_prior', task=task, mean_r=r, pct_ceiling=100*z/ceil[task]))
        # best center gaussian
        best = None
        for sf in [0.2, 0.3, 0.4]:
            r, z = align_of_constant_map(gaussian_map(sf), task, imgs)
            if best is None or r > best[0]:
                best = (r, z, sf)
        rows.append(dict(baseline=f'center_gaussian(s={best[2]})', task=task,
                         mean_r=best[0], pct_ceiling=100*best[1]/ceil[task]))
    bdf = pd.DataFrame(rows)
    bdf.to_csv(os.path.join(RES, 'baseline_alignment.csv'), index=False)
    print(bdf.to_string(index=False))

    # ── blur sweep ──
    sweep = []
    # extended past the point of diminishing returns: at sigma=12 alignment was
    # still rising, so the sweep must run until it plateaus/turns over, otherwise
    # "you didn't blur enough" remains open.
    sigmas = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    for task in ['social', 'describe']:
        lstm_ref = float(summ[(summ.model == 'CNN-LSTM') & (summ.task == task)]['mean_r'].iloc[0])
        for m in ['CNN-Transformer', 'ViT-Transformer']:
            base = A.trained_maps(m, task, imgs)  # best-head 256x256 maps
            for s in sigmas:
                blurred = {im: (gaussian_filter(mp, s) if s > 0 else mp) for im, mp in base.items()}
                df = A.per_image_z(blurred, task, imgs, m)
                z = A.subject_means(df)['mean_z'].values
                sweep.append(dict(task=task, model=m, sigma=s, mean_r=float(np.tanh(z.mean())),
                                  lstm_ref=lstm_ref))
        print(f'  blur sweep {task} done', flush=True)
    sdf = pd.DataFrame(sweep)
    sdf.to_csv(os.path.join(RES, 'blur_sweep.csv'), index=False)

    # ── figures ──
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, task in zip(ax, ['social', 'describe']):
        for m, c in [('CNN-Transformer', '#d62728'), ('ViT-Transformer', '#9467bd')]:
            d = sdf[(sdf.task == task) & (sdf.model == m)]
            a.plot(d['sigma'], d['mean_r'], marker='o', label=m, color=c)
        lref = sdf[sdf.task == task]['lstm_ref'].iloc[0]
        a.axhline(lref, ls='--', color='orange', label='CNN-LSTM (best)')
        a.axhline(ceil[task], ls='-', color='gray', alpha=0.5, label='human ceiling')
        a.set_xlabel('Gaussian blur sigma (px)'); a.set_ylabel('Spearman r')
        a.set_title(f'{task.capitalize()}: does blurring a Transformer map reach LSTM?')
        a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_blur_sweep.png'), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    order = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
    xs = np.arange(len(order))
    for off, task, c in [(-0.2, 'social', '#4c72b0'), (0.2, 'describe', '#dd8452')]:
        vals = [float(summ[(summ.model == m) & (summ.task == task)]['mean_r'].iloc[0]) for m in order]
        ax2.bar(xs + off, vals, 0.4, label=task, color=c)
    for task, c, ls in [('social', '#4c72b0', '--'), ('describe', '#dd8452', ':')]:
        pv = bdf[(bdf.baseline == 'center_bias_prior') & (bdf.task == task)]['mean_r'].iloc[0]
        ax2.axhline(pv, color=c, ls=ls, lw=2, label=f'center-bias prior ({task})')
    ax2.set_xticks(xs); ax2.set_xticklabels(order, rotation=45, ha='right')
    ax2.set_ylabel('Spearman r'); ax2.set_title('Models vs center-bias prior')
    ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_baselines.png'), dpi=150)
    print('Saved figures.')


if __name__ == '__main__':
    main()
