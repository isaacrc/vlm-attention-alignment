"""
Task 4 — alignment metric suite (Bylinskii et al. 2019) beyond Spearman.
For every model, both tasks: CC, NSS, AUC-Judd, sAUC, KL, SIM, InfoGain vs
center-bias prior. Human 'G' = mean human map per image; fixations = top-5% pixels.
Ceiling-normalize per metric (human-human under each metric). Reports rank
stability (Kendall tau) across metrics.

Needs torch (VLM maps) -> run on compute node.
Outputs: results/metric_matrix.csv, results/rank_stability.json
Figure: figures/fig_metric_suite.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import kendalltau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
import alignment_core as A
import vlm_maps as V
import saliency_metrics as SM

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data')

MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
METRICS = ['CC', 'NSS', 'AUC', 'sAUC', 'KL', 'SIM', 'IG']


def human_mean_maps(task, imgs):
    """{img: mean human map (256x256)} on canonical subject set."""
    hm_dir = A.HM_DIR[task]
    files = [f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy')
             and f.replace('_heatmaps.npy', '') not in A.EXCLUDE[task]]
    acc = {im: np.zeros((256, 256), np.float64) for im in imgs}
    cnt = {im: 0 for im in imgs}
    for fn in files:
        sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
        for im in imgs:
            hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
            if hm is not None:
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                acc[im] += np.array(Image.fromarray(h).resize((256, 256), Image.NEAREST), np.float64)
                cnt[im] += 1
    return {im: acc[im] / cnt[im] for im in imgs if cnt[im] > 0}


def get_maps(model, task, imgs):
    if model in ('Molmo', 'Qwen3.5'):
        return V.vlm_maps(model, task)
    return A.trained_maps(model, task, imgs)


def eval_metrics(P, G, fixb, other_fix, prior):
    return dict(CC=SM.cc(P, G), NSS=SM.nss(P, fixb), AUC=SM.auc_judd(P, fixb),
                sAUC=SM.auc_shuffled(P, fixb, other_fix), KL=SM.kl(P, G),
                SIM=SM.sim(P, G), IG=SM.info_gain(P, fixb, prior))


def main():
    imgs = A.target_images()
    rows = []
    for task in ['social', 'describe']:
        H = human_mean_maps(task, imgs)
        keys = [im for im in imgs if im in H]
        prior = np.mean([H[im] for im in keys], axis=0)
        fixb = {im: SM.fixation_binary(H[im]) for im in keys}
        # a pooled "other" fixation map for sAUC (all fixations)
        other = np.clip(np.sum([fixb[im] for im in keys], axis=0), 0, 1)

        # human-human ceiling per metric: split subjects, map half vs other half's mean
        # (approx via: each image's map vs prior-free leave-one-out not available here;
        #  use split-half of subjects)
        ceil_vals = {mt: [] for mt in METRICS}
        # per-model metrics
        for model in MODELS:
            maps = get_maps(model, task, imgs)
            vals = {mt: [] for mt in METRICS}
            for im in keys:
                if im not in maps:
                    continue
                P = np.asarray(maps[im], np.float32)
                if P.shape != (256, 256):
                    P = np.array(Image.fromarray(P).resize((256, 256), Image.BILINEAR))
                mv = eval_metrics(P, H[im], fixb[im], other, prior)
                for mt in METRICS:
                    if not np.isnan(mv[mt]):
                        vals[mt].append(mv[mt])
            for mt in METRICS:
                rows.append(dict(model=model, task=task, metric=mt,
                                 value=float(np.mean(vals[mt])) if vals[mt] else np.nan))
            print(f'  {task} {model} done', flush=True)

        # human split-half ceiling per metric
        hm_dir = A.HM_DIR[task]
        files = sorted(f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy')
                       and f.replace('_heatmaps.npy', '') not in A.EXCLUDE[task])
        half = len(files) // 2
        def mean_over(fs):
            acc = {im: np.zeros((256, 256)) for im in keys}; c = {im: 0 for im in keys}
            for fn in fs:
                sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
                for im in keys:
                    hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
                    if hm is not None:
                        h = np.nan_to_num(hm, 0.0).astype(np.float32)
                        acc[im] += np.array(Image.fromarray(h).resize((256, 256), Image.NEAREST)); c[im] += 1
            return {im: acc[im] / c[im] for im in keys if c[im] > 0}
        A_half, B_half = mean_over(files[:half]), mean_over(files[half:])
        for im in keys:
            if im in A_half and im in B_half:
                fb = SM.fixation_binary(B_half[im])
                mv = eval_metrics(A_half[im], B_half[im], fb, other, prior)
                for mt in METRICS:
                    if not np.isnan(mv[mt]):
                        ceil_vals[mt].append(mv[mt])
        for mt in METRICS:
            rows.append(dict(model='HUMAN_CEILING', task=task, metric=mt,
                             value=float(np.mean(ceil_vals[mt])) if ceil_vals[mt] else np.nan))

    mm = pd.DataFrame(rows)
    mm.to_csv(os.path.join(RES, 'metric_matrix.csv'), index=False)

    # rank stability across metrics (higher-better; flip KL)
    tau = {}
    for task in ['social', 'describe']:
        sub = mm[(mm.task == task) & (mm.model != 'HUMAN_CEILING')]
        piv = sub.pivot_table(index='model', columns='metric', values='value')
        piv['KL'] = -piv['KL']
        ranks = piv.rank(ascending=False)
        cols = ranks.columns.tolist(); taus = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                t, _ = kendalltau(ranks[cols[i]], ranks[cols[j]]); taus.append(t)
        tau[task] = float(np.nanmean(taus))
    json.dump(tau, open(os.path.join(RES, 'rank_stability.json'), 'w'), indent=2)
    print('Rank stability (Kendall tau):', tau)

    # figure: ceiling-normalized heatmap
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, task in zip(axes, ['social', 'describe']):
        sub = mm[mm.task == task]
        piv = sub.pivot_table(index='model', columns='metric', values='value')
        ceilrow = piv.loc['HUMAN_CEILING']
        norm = piv.drop(index='HUMAN_CEILING').copy()
        for mt in METRICS:
            if mt == 'KL':
                norm[mt] = ceilrow[mt] / norm[mt]      # lower better -> invert
            else:
                norm[mt] = norm[mt] / ceilrow[mt]
        norm = norm.reindex(MODELS)[METRICS]
        im = ax.imshow(norm.values, aspect='auto', cmap='viridis', vmin=0, vmax=1.2)
        ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(METRICS)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
        ax.set_title(f'{task.capitalize()} — metric / human ceiling')
        for i in range(len(MODELS)):
            for j in range(len(METRICS)):
                ax.text(j, i, f'{norm.values[i,j]:.2f}', ha='center', va='center',
                        color='white', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_metric_suite.png'), dpi=150)
    print('Saved figure.')


if __name__ == '__main__':
    main()
