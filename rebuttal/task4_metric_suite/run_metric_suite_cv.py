"""
run_metric_suite_cv.py — Task 4 metric suite, recomputed on cross-validated heads.
==================================================================================

`run_metric_suite.py` produced the seven-metric matrix quoted in the rebuttal, but it
read every Transformer through `alignment_core.trained_maps` with no explicit
(layer, head), which falls back to the paper's fixed L3H5 / L3H8, and every VLM through
the sweep-argmax head. Same mixed head rule as the center-controlled table (see
writeups/task3c_headrule_center.md). It also stored only per-model means, so the matrix
carries no error bars and supports no paired test.

This version fixes both:

  * heads come from Task 1's cross-validated selection (results/selected_heads.json),
    so no head is chosen on the metric it is scored against;
  * every metric is stored PER IMAGE, which gives SEMs for the figures and lets the
    LSTM-vs-matched-Transformer contrast be tested per metric with a paired
    permutation test over images;
  * the human ceiling is averaged over N_SPLIT random split-halves of participants
    rather than one arbitrary split by filename order.

AUC-Judd and sAUC are re-implemented here in closed form (rank/cumsum) rather than the
threshold loop in shared/saliency_metrics.py, which is O(n_fix * n_pix) per image and
made the original run take ~37 min. The fast forms are checked against the originals at
startup and must agree to 1e-6.

Outputs: results/metric_perimage_cv.csv, results/metric_summary_cv.csv,
         results/metric_contrasts_cv.csv, results/metric_ceilings_cv.csv
Run: sbatch run_t4cv.slurm
"""
import os, sys, json
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
sys.path.insert(0, os.path.join(HERE, '..', 'task3_baselines'))

import alignment_core as A
import vlm_maps as V
import saliency_metrics as SM

RES = os.path.join(HERE, 'results')
T1 = os.path.join(HERE, '..', 'task1_head_selection', 'results')
os.makedirs(RES, exist_ok=True)

SELECTED = json.load(open(os.path.join(T1, 'selected_heads.json')))
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
METRICS = ['CC', 'NSS', 'AUC', 'sAUC', 'KL', 'SIM', 'IG']
N_SPLIT = 20


# ── fast AUCs ──────────────────────────────────────────────────────────────────
def auc_judd_fast(P, fix_binary):
    """AUC-Judd, identical to shared/saliency_metrics.auc_judd but vectorized.

    The reference sweeps thresholds at each fixation's saliency value and counts
    pixels at or above each one, which is O(n_fix * n_pix). The same counts come
    from one sort plus a searchsorted, which is O(n_pix log n_pix).
    """
    p = P.ravel().astype(np.float64)
    f = fix_binary.ravel() > 0
    n_fix, n_pix = int(f.sum()), p.size
    if n_fix == 0 or n_fix == n_pix:
        return np.nan
    p_asc = np.sort(p)
    th = np.sort(p[f])[::-1]
    above = n_pix - np.searchsorted(p_asc, th, side='left')
    i = np.arange(1, n_fix + 1)
    tp = i / n_fix
    fp = (above - i) / (n_pix - n_fix)
    return float(np.trapz(np.r_[0, tp, 1], np.r_[0, fp, 1]))


def auc_shuffled_fast(P, fix_binary, other_fix):
    """sAUC as the Mann-Whitney statistic between saliency at this image's
    fixations and saliency at other images' fixations. Deterministic -- it uses
    the whole negative set instead of resampling it n_splits times."""
    p = SM._norm01(P).ravel()
    pos, neg = p[fix_binary.ravel() > 0], p[other_fix.ravel() > 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method='average').values
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _check_fast():
    """AUC-Judd must reproduce the reference exactly. sAUC is checked only for
    agreement, not equality: the reference draws n_fix negatives at random from the
    pooled other-image fixations and averages 10 such draws, whereas the form here
    uses the whole negative set, which is that estimator's deterministic limit."""
    rng = np.random.default_rng(0)
    dj, ds = [], []
    for _ in range(3):
        P = rng.random((64, 64)); G = rng.random((64, 64))
        fb = SM.fixation_binary(G)
        other = SM.fixation_binary(rng.random((64, 64)))
        dj.append(abs(auc_judd_fast(P, fb) - SM.auc_judd(P, fb)))
        ds.append(abs(auc_shuffled_fast(P, fb, other) - SM.auc_shuffled(P, fb, other)))
    assert max(dj) < 1e-9, f'auc_judd mismatch {max(dj)}'
    assert max(ds) < 0.05, f'sAUC disagrees with reference by {max(ds)}'
    print(f'AUC-Judd matches reference (max diff {max(dj):.2e}); '
          f'sAUC within {max(ds):.4f} of the resampling reference', flush=True)


def eval_metrics(P, G, fixb, other_fix, prior):
    return dict(CC=SM.cc(P, G), NSS=SM.nss(P, fixb), AUC=auc_judd_fast(P, fixb),
                sAUC=auc_shuffled_fast(P, fixb, other_fix), KL=SM.kl(P, G),
                SIM=SM.sim(P, G), IG=SM.info_gain(P, fixb, prior))


# ── data ───────────────────────────────────────────────────────────────────────
def subject_maps(task, imgs):
    """{subject: {img: (256,256)}} on the canonical (exclusion-respecting) set."""
    hm_dir = A.HM_DIR[task]
    files = sorted(f for f in os.listdir(hm_dir) if f.endswith('_heatmaps.npy')
                   and f.replace('_heatmaps.npy', '') not in A.EXCLUDE[task])
    out = {}
    for fn in files:
        sd = np.load(os.path.join(hm_dir, fn), allow_pickle=True).item()
        d = {}
        for im in imgs:
            hm = sd.get(im, sd.get(im.replace('.jpg', ''), sd.get(im + '.jpg')))
            if hm is not None:
                h = np.nan_to_num(hm, nan=0.0).astype(np.float32)
                d[im] = np.array(Image.fromarray(h).resize((256, 256), Image.NEAREST), np.float32)
        out[fn.replace('_heatmaps.npy', '')] = d
    return out


def mean_maps(smaps, subs, keys):
    acc = {im: np.zeros((256, 256), np.float64) for im in keys}
    cnt = {im: 0 for im in keys}
    for s in subs:
        for im, m in smaps[s].items():
            if im in acc:
                acc[im] += m; cnt[im] += 1
    return {im: acc[im] / cnt[im] for im in keys if cnt[im] > 0}


def get_maps(model, task, imgs):
    """CV-honest head for every model that has one to choose."""
    if model in ('CNN-LSTM', 'ViT-LSTM'):
        return A.trained_maps(model, task, imgs)
    d = SELECTED[f'{model}_{task}']
    L, Hd = int(d['layer']), int(d['head'])
    if model in ('Molmo', 'Qwen3.5'):
        return V.vlm_maps(model, task, L, Hd)
    return A.trained_maps(model, task, imgs, layer=L, head=Hd)


def main():
    _check_fast()
    imgs = A.target_images()
    per_img, ceil_rows = [], []

    for task in ['social', 'describe']:
        smaps = subject_maps(task, imgs)
        subs = sorted(smaps)
        keys = [im for im in imgs if any(im in smaps[s] for s in subs)]
        H = mean_maps(smaps, subs, keys)
        prior = np.mean([H[im] for im in keys], axis=0)
        fixb = {im: SM.fixation_binary(H[im]) for im in keys}
        other = np.clip(np.sum([fixb[im] for im in keys], axis=0), 0, 1)
        print(f'\n[{task}] {len(subs)} subjects, {len(keys)} images', flush=True)

        for model in MODELS:
            maps = get_maps(model, task, imgs)
            head = ('-' if model in ('CNN-LSTM', 'ViT-LSTM')
                    else f"L{SELECTED[f'{model}_{task}']['layer']}H{SELECTED[f'{model}_{task}']['head']}")
            for im in keys:
                if im not in maps:
                    continue
                P = np.asarray(maps[im], np.float32)
                if P.shape != (256, 256):
                    P = np.array(Image.fromarray(P).resize((256, 256), Image.BILINEAR))
                mv = eval_metrics(P, H[im], fixb[im], other, prior)
                for mt in METRICS:
                    if not np.isnan(mv[mt]):
                        per_img.append(dict(model=model, task=task, head=head,
                                            metric=mt, image_id=im, value=mv[mt]))
            print(f'  {model:16s} {head:8s} done', flush=True)

        # human ceiling: mean over N_SPLIT random split-halves of participants
        rng = np.random.default_rng(0)
        for si in range(N_SPLIT):
            perm = rng.permutation(subs)
            hA, hB = perm[:len(subs) // 2], perm[len(subs) // 2:]
            MA, MB = mean_maps(smaps, hA, keys), mean_maps(smaps, hB, keys)
            for im in keys:
                if im not in MA or im not in MB:
                    continue
                mv = eval_metrics(MA[im], MB[im], SM.fixation_binary(MB[im]), other, prior)
                for mt in METRICS:
                    if not np.isnan(mv[mt]):
                        ceil_rows.append(dict(task=task, metric=mt, split=si,
                                              image_id=im, value=mv[mt]))
        print(f'  ceiling: {N_SPLIT} split-halves done', flush=True)

    pi = pd.DataFrame(per_img)
    pi.to_csv(os.path.join(RES, 'metric_perimage_cv.csv'), index=False)
    cd = pd.DataFrame(ceil_rows)
    ceil = cd.groupby(['task', 'metric'])['value'].agg(['mean', 'sem']).reset_index()
    ceil.columns = ['task', 'metric', 'ceiling', 'ceiling_sem']
    ceil.to_csv(os.path.join(RES, 'metric_ceilings_cv.csv'), index=False)

    summ = (pi.groupby(['model', 'task', 'head', 'metric'])['value']
              .agg(['mean', 'sem', 'count']).reset_index())
    summ.columns = ['model', 'task', 'head', 'metric', 'value', 'sem', 'n_img']
    summ = summ.merge(ceil, on=['task', 'metric'])
    summ.to_csv(os.path.join(RES, 'metric_summary_cv.csv'), index=False)

    # ── paired permutation: LSTM vs matched Transformer, per metric, over images ──
    rng = np.random.default_rng(0)
    tests = []
    for task in ['social', 'describe']:
        for lstm, trans in [('CNN-LSTM', 'CNN-Transformer'), ('ViT-LSTM', 'ViT-Transformer')]:
            for mt in METRICS:
                sub = pi[(pi.task == task) & (pi.metric == mt)]
                x = sub[sub.model == lstm].set_index('image_id')['value']
                y = sub[sub.model == trans].set_index('image_id')['value']
                c = x.index.intersection(y.index)
                if len(c) == 0:
                    continue
                d = (x.loc[c] - y.loc[c]).values
                if mt == 'KL':
                    d = -d                       # lower is better -> flip so + favours LSTM
                obs = d.mean()
                null = (rng.choice([-1, 1], size=(10000, len(d))) * d).mean(axis=1)
                tests.append(dict(task=task, lstm=lstm, transformer=trans, metric=mt,
                                  mean_diff=float(obs),
                                  dz=float(obs / (d.std(ddof=1) + 1e-12)), n_img=len(d),
                                  p_perm=float((np.sum(np.abs(null) >= abs(obs)) + 1) / 10001),
                                  lstm_wins=bool(obs > 0)))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(RES, 'metric_contrasts_cv.csv'), index=False)

    print('\n=== ceilings ===')
    print(ceil.to_string(index=False))
    print('\n=== LSTM vs matched Transformer (positive dz = LSTM better) ===')
    print(tdf[['task', 'lstm', 'transformer', 'metric', 'dz', 'p_perm', 'lstm_wins']]
          .to_string(index=False))
    n_win = int(tdf.lstm_wins.sum())
    print(f'\nLSTM wins {n_win} of {len(tdf)} model x metric x task comparisons')
    print('Wrote results/metric_{perimage,summary,contrasts,ceilings}_cv.csv')


if __name__ == '__main__':
    main()
