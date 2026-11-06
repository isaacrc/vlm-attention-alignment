"""
Task 3b — center-bias-controlled alignment.

Motivation: an image-independent center-bias prior reaches r=.209 (social) and
r=.237 (describe), i.e. 76% / 95% of the human noise ceiling, without using the
image at all. Raw Spearman against a single mean human map is therefore dominated
by center bias and cannot by itself establish image-specific alignment.

This script re-runs the model comparison with center bias removed, three ways:

  1. PARTIAL Spearman r(model, human | prior), per subject x image. The prior is
     built LEAVE-ONE-IMAGE-OUT (mean human map over all other images, pooled over
     subjects and tasks) so it never sees the image it is applied to.
  2. Human ceiling under the same partial metric (pairwise between subjects,
     controlling for the same LOIO prior) -> a like-for-like normalizer.
  3. The center prior's own partial score, which is ~0 by construction and serves
     as the sanity check.

Outputs: results/center_controlled.csv, results/center_controlled_summary.csv
Figure:  figures/fig_center_controlled.png
"""
import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
import alignment_core as A
import vlm_maps as V

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)

MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
D = A.OUT_SIZE * A.OUT_SIZE


def _nrm(x):
    """Center and unit-normalize along last axis."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def loio_prior_ranked(imgs):
    """Leave-one-image-out center prior, pre-ranked, shape (N, D).

    Prior for image i = mean raw human map over every image except i, pooled over
    subjects and BOTH tasks. Because it excludes image i it carries no
    image-specific information, only the global spatial bias.
    """
    acc = np.zeros(D, np.float64)          # grand sum over all subj x img
    per_img = np.zeros((len(imgs), D), np.float64)   # sum over subjects per image
    cnt_img = np.zeros(len(imgs))
    for task in ['social', 'describe']:
        hm_dir = A.HM_DIR[task]
        for fn in sorted(os.listdir(hm_dir)):
            if not fn.endswith('_heatmaps.npy'):
                continue
            if fn.replace('_heatmaps.npy', '') in A.EXCLUDE[task]:
                continue
            sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
            for k, im in enumerate(imgs):
                hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
                if hm is None:
                    continue
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                h = np.asarray(Image.fromarray(h).resize((A.OUT_SIZE, A.OUT_SIZE),
                                                         Image.NEAREST), np.float64).ravel()
                per_img[k] += h; cnt_img[k] += 1; acc += h
    total = cnt_img.sum()
    out = np.empty((len(imgs), D), np.float32)
    for k in range(len(imgs)):
        denom = max(total - cnt_img[k], 1)
        out[k] = ((acc - per_img[k]) / denom).astype(np.float32)
    return A._rank(out, axis=1)


def partial_rows(model_maps, task, imgs, prior_rk, model_name):
    """Per subject x image partial Spearman r(model, human | prior)."""
    ranked, valid, subs = A.load_human(task, imgs)
    idx = {im: i for i, im in enumerate(imgs)}
    order = [im for im in imgs if im in model_maps]
    mv = np.stack([np.asarray(model_maps[im], np.float32).reshape(-1) for im in order])
    keep = mv.std(axis=1) > 1e-12
    order = [order[i] for i in range(len(order)) if keep[i]]
    mv_n = _nrm(A._rank(mv[keep], axis=1))
    pr_n = _nrm(prior_rk.astype(np.float32))

    rows = []
    for k, im in enumerate(order):
        j = idx[im]
        r_mp = float(np.dot(mv_n[k], pr_n[j]))          # model vs prior
        for si, sid in enumerate(subs):
            if not valid[si, j]:
                continue
            hv = _nrm(ranked[si, j].astype(np.float32))
            r_mh = float(np.dot(mv_n[k], hv))
            r_hp = float(np.dot(hv, pr_n[j]))
            den = np.sqrt(max((1 - r_mp**2) * (1 - r_hp**2), 1e-12))
            rp = np.clip((r_mh - r_mp * r_hp) / den, -0.999999, 0.999999)
            rows.append(dict(model=model_name, task=task, subject_id=sid, image_id=im,
                             r_raw=r_mh, r_partial=float(rp), z=float(np.arctanh(rp))))
    return pd.DataFrame(rows)


def partial_ceiling(task, imgs, prior_rk):
    """Human ceiling under the partial metric: pairwise between-subject partial
    Spearman controlling the same LOIO prior. Returns mean Fisher-z."""
    ranked, valid, _ = A.load_human(task, imgs)
    pr_n = _nrm(prior_rk.astype(np.float32))
    per_img = []
    for j in range(len(imgs)):
        vi = np.where(valid[:, j])[0]
        if len(vi) < 2:
            continue
        Vn = _nrm(ranked[vi, j].astype(np.float32))
        C = Vn @ Vn.T                       # subject x subject raw corr
        rp_vec = Vn @ pr_n[j]               # each subject vs prior
        den = np.sqrt(np.clip(1 - rp_vec**2, 1e-12, None))
        P = (C - np.outer(rp_vec, rp_vec)) / np.outer(den, den)
        iu = np.triu_indices(len(vi), 1)
        per_img.append(np.arctanh(np.clip(P[iu], -0.999999, 0.999999)).mean())
    return float(np.mean(per_img))


def get_maps(model, task, imgs):
    if model in ('Molmo', 'Qwen3.5'):
        return V.vlm_maps(model, task)
    return A.trained_maps(model, task, imgs)


def main():
    imgs = A.target_images()
    print('building leave-one-image-out center prior...', flush=True)
    prior_rk = loio_prior_ranked(imgs)

    all_rows, summary = [], []
    for task in ['social', 'describe']:
        pc = partial_ceiling(task, imgs, prior_rk)
        raw_c = A.ceiling(task)
        print(f'[{task}] ceiling raw z={raw_c:.4f} (r={np.tanh(raw_c):.4f}) | '
              f'partial z={pc:.4f} (r={np.tanh(pc):.4f})', flush=True)

        # the prior itself, as a sanity check (should be ~0 partial)
        pr_maps = {im: prior_rk[i].reshape(A.OUT_SIZE, A.OUT_SIZE)
                   for i, im in enumerate(imgs)}
        for name, maps in [('CENTER_PRIOR', pr_maps)] + \
                          [(m, get_maps(m, task, imgs)) for m in MODELS]:
            df = partial_rows(maps, task, imgs, prior_rk, name)
            if df.empty:
                continue
            all_rows.append(df)
            sm = df.groupby('subject_id')['z'].mean()
            zbar = float(sm.mean())
            summary.append(dict(
                model=name, task=task,
                r_raw=float(np.tanh(df.groupby('subject_id')['r_raw']
                                    .apply(lambda s: np.arctanh(np.clip(s, -.999999, .999999)).mean()).mean())),
                r_partial=float(np.tanh(zbar)),
                pct_partial_ceiling=100 * zbar / pc,
                pct_raw_ceiling=100 * float(df.groupby('subject_id')['z'].mean().mean()) / pc,
                n_sub=len(sm)))
            print(f'  {name:16s} partial r={np.tanh(zbar):.4f} '
                  f'({100*zbar/pc:.1f}% of partial ceiling)', flush=True)
        summary.append(dict(model='HUMAN_CEILING', task=task, r_raw=float(np.tanh(raw_c)),
                            r_partial=float(np.tanh(pc)), pct_partial_ceiling=100.0,
                            pct_raw_ceiling=100.0, n_sub=np.nan))

    long = pd.concat(all_rows, ignore_index=True)
    long.to_csv(os.path.join(RES, 'center_controlled.csv'), index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(RES, 'center_controlled_summary.csv'), index=False)
    print('\n' + sdf.to_string(index=False))

    # paired permutation on the key LSTM-vs-Transformer contrasts, partial metric
    rng = np.random.default_rng(0)
    pim = long.groupby(['model', 'task', 'image_id'])['z'].mean().reset_index()
    tests = []
    for task in ['social', 'describe']:
        for a, b in [('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer'),
                     ('CNN-LSTM', 'CENTER_PRIOR'), ('CNN-Transformer', 'CENTER_PRIOR'),
                     ('Molmo', 'CENTER_PRIOR'), ('Qwen3.5', 'CENTER_PRIOR')]:
            x = pim[(pim.model == a) & (pim.task == task)].set_index('image_id')['z']
            y = pim[(pim.model == b) & (pim.task == task)].set_index('image_id')['z']
            c = x.index.intersection(y.index)
            if len(c) == 0:
                continue
            d = (x.loc[c] - y.loc[c]).values
            obs = d.mean()
            null = (rng.choice([-1, 1], size=(10000, len(d))) * d).mean(axis=1)
            tests.append(dict(a=a, b=b, task=task, mean_diff=float(obs),
                              dz=float(obs / (d.std(ddof=1) + 1e-12)), n_img=len(d),
                              p_perm=float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001)))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RES, 'center_controlled_contrasts.csv'), index=False)
    print('\n' + tdf.to_string(index=False))

    # ── figure ──
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    order = MODELS + ['CENTER_PRIOR']
    for ax, task in zip(axes, ['social', 'describe']):
        sub = sdf[sdf.task == task].set_index('model')
        raw = [sub.loc[m, 'r_raw'] if m in sub.index else np.nan for m in order]
        par = [sub.loc[m, 'r_partial'] if m in sub.index else np.nan for m in order]
        x = np.arange(len(order))
        ax.bar(x - 0.2, raw, 0.4, label='raw Spearman', color='#b0b0b0')
        ax.bar(x + 0.2, par, 0.4, label='partial (center removed)', color='#4c72b0')
        ax.axhline(sub.loc['HUMAN_CEILING', 'r_raw'], ls=':', color='#808080',
                   label='ceiling (raw)')
        ax.axhline(sub.loc['HUMAN_CEILING', 'r_partial'], ls='--', color='#4c72b0',
                   label='ceiling (partial)')
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_title(f'{task.capitalize()}')
        ax.set_ylabel('Spearman r')
    axes[0].legend(fontsize=8)
    fig.suptitle('Alignment before vs after removing the center-bias prior')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_center_controlled.png'), dpi=150)
    print('\nSaved figure.')


if __name__ == '__main__':
    main()
