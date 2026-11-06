"""
Per-image human between-task similarity on the SAME footing as the models.

Model between-task similarity correlates a model's Social attention map against
its own Describe attention map for the same image: one map versus one map, no
measurement noise. The paper's human reference instead correlated one Social
PARTICIPANT against one Describe PARTICIPANT, which is a between-subject
quantity and is attenuated by noise in both groups. That mismatch is the
reviewers' objection (AC comment 5; hmYo).

The like-for-like human quantity is the group-mean Social map versus the
group-mean Describe map, per image. That is what this computes. It needs no
attenuation correction, because both sides are group-level estimates and the
comparison is map-to-map exactly as it is for the models.

Also emits the group-size ceiling: the split-half within-task correlation,
stepped up to full group size by Spearman-Brown. Human group maps are built from
27 and 22 participants, so unlike the models they do carry sampling noise, and
the ceiling states how much. Reported alongside rather than divided out.

Outputs: results/human_group_between_task.csv  (per-image r)
         results/human_group_between_task.json (mean, SEM, ceiling, index)
"""
import os, sys, json, gc
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A

RES = os.path.join(HERE, 'results')


def group_mean(ranked, valid, sub_idx):
    N, D = ranked.shape[1], ranked.shape[2]
    out = np.zeros((N, D), np.float32)
    for i in range(N):
        vi = sub_idx[valid[sub_idx, i]]
        if len(vi):
            out[i] = ranked[vi, i, :].mean(axis=0)
    return out


def rank_rows(G):
    out = np.empty(G.shape, np.float32)
    for i in range(len(G)):
        out[i] = rankdata(G[i])
    return out


def corr_rows(R1, R2):
    out = np.full(len(R1), np.nan)
    for i in range(len(R1)):
        a = R1[i].astype(np.float64); b = R2[i].astype(np.float64)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
        out[i] = float(np.dot(a, b) / len(a))
    return out


def fisher_mean(r):
    r = r[~np.isnan(r)]
    return float(np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean()))


def fisher_sem(r):
    r = r[~np.isnan(r)]
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.std(z, ddof=1) / np.sqrt(len(z)) * (1 - fisher_mean(r) ** 2))


def main():
    imgs = A.target_images()

    R = {}
    for task in ['social', 'describe']:
        ranked, valid, subs = A.load_human(task, imgs)
        R[task] = rank_rows(group_mean(ranked, valid, np.arange(len(subs))))
        print(f'{task}: n={len(subs)} group map built', flush=True)
        del ranked, valid
        A._HUMAN_CACHE.clear(); gc.collect()

    per_img = corr_rows(R['social'], R['describe'])
    mean_r, sem_r = fisher_mean(per_img), fisher_sem(per_img)

    pd.DataFrame({'image_name': imgs, 'between_task_r': per_img}).to_csv(
        os.path.join(RES, 'human_group_between_task.csv'), index=False)

    # group-size ceiling from the companion split-half run
    ceil_f = os.path.join(RES, 'within_task_ceiling.json')
    ceiling = json.load(open(ceil_f))['ceiling_geomean_raw'] if os.path.exists(ceil_f) else None

    out = dict(between_r=mean_r, sem=sem_r, n_img=int(np.sum(~np.isnan(per_img))),
               n_social=27, n_describe=22, group_size_ceiling=ceiling)
    if ceiling:
        out['between_r_noise_corrected'] = mean_r / ceiling
        out['task_diff_index'] = 1 - mean_r / ceiling
    json.dump(out, open(os.path.join(RES, 'human_group_between_task.json'), 'w'), indent=2)

    print(f'\nHuman group-level between-task r = {mean_r:.4f} (SEM {sem_r:.4f}, '
          f'{out["n_img"]} images)')
    if ceiling:
        print(f'  group-size ceiling            = {ceiling:.4f}')
        print(f'  noise-corrected               = {out["between_r_noise_corrected"]:.4f}')
        print(f'  task-differentiation index    = {out["task_diff_index"]:.4f}')


if __name__ == '__main__':
    main()
