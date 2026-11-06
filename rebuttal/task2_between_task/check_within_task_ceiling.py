"""
Companion to check_group_level_between_task.py — the within-task ceiling only.

The between-task half of that script finished (raw 0.658, partial 0.525) but the
split-half loop exhausted memory holding both tasks' (S, 200, 65536) arrays at
once. This runs one task at a time, in float32, dropping the human cache between
tasks.

Purpose: a group-level between-task correlation of 0.658 is itself attenuated,
because a 27-subject and a 22-subject group mean are noisy estimates of their
respective true group maps. The within-task split-half correlation, stepped up
to full group size by Spearman-Brown, is the matching ceiling. Reading 0.658
against that ceiling is the like-for-like version of the comparison the
disattenuation formula tries to make in one step.

Outputs: results/within_task_ceiling.json
"""
import os, sys, json, gc, time
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A

RES = os.path.join(HERE, 'results')
RNG = np.random.default_rng(0)
N_SPLIT = 20


def rank_rows(G):
    out = np.empty(G.shape, np.float32)
    for i in range(len(G)):
        out[i] = rankdata(G[i])
    return out


def _z(v):
    v = v.astype(np.float64)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else None


def corr_and_partial(R1, R2, RP):
    """Per-image Spearman and center-partialled Spearman between two rank maps."""
    raw, par = [], []
    for i in range(len(R1)):
        a, b, c = _z(R1[i]), _z(R2[i]), _z(RP[i])
        if a is None or b is None or c is None:
            continue
        n = len(a)
        rab = float(np.dot(a, b) / n)
        rac = float(np.dot(a, c) / n)
        rbc = float(np.dot(b, c) / n)
        raw.append(rab)
        den = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
        if den > 1e-12:
            par.append((rab - rac * rbc) / den)
    fm = lambda v: float(np.tanh(np.arctanh(np.clip(np.array(v), -0.999999, 0.999999)).mean()))
    return fm(raw), fm(par)


def group_mean(ranked, valid, sub_idx):
    N, D = ranked.shape[1], ranked.shape[2]
    out = np.zeros((N, D), np.float32)
    for i in range(N):
        vi = sub_idx[valid[sub_idx, i]]
        if len(vi):
            out[i] = ranked[vi, i, :].mean(axis=0)
    return out


def full_group_mean(task, imgs):
    ranked, valid, subs = A.load_human(task, imgs)
    G = group_mean(ranked, valid, np.arange(len(subs)))
    return G


def main():
    t0 = time.time()
    imgs = A.target_images()

    # rebuild the same leave-one-image-out prior the between-task run used:
    # pooled over both tasks, so it is identical for both.
    G_soc = full_group_mean('social', imgs)
    A._HUMAN_CACHE.clear(); gc.collect()
    G_des = full_group_mean('describe', imgs)
    A._HUMAN_CACHE.clear(); gc.collect()

    G_pool = (G_soc.astype(np.float64) + G_des.astype(np.float64)) / 2.0
    tot = G_pool.sum(axis=0)
    P = ((tot[None, :] - G_pool) / (len(G_pool) - 1)).astype(np.float32)
    del G_soc, G_des, G_pool, tot; gc.collect()
    RP = rank_rows(P)
    del P; gc.collect()
    print(f'prior ranked ({time.time()-t0:.0f}s)', flush=True)

    res = {}
    for task in ['social', 'describe']:
        ranked, valid, subs = A.load_human(task, imgs)
        S = len(subs)
        raws, pars = [], []
        for k in range(N_SPLIT):
            perm = RNG.permutation(S)
            h1, h2 = perm[:S // 2], perm[S // 2:]
            R1 = rank_rows(group_mean(ranked, valid, h1))
            R2 = rank_rows(group_mean(ranked, valid, h2))
            r, p = corr_and_partial(R1, R2, RP)
            raws.append(r); pars.append(p)
            del R1, R2
            if k % 5 == 0:
                print(f'  {task} split {k}: raw={r:.4f} partial={p:.4f} '
                      f'({time.time()-t0:.0f}s)', flush=True)
        del ranked, valid; A._HUMAN_CACHE.clear(); gc.collect()

        rh_raw, rh_par = float(np.mean(raws)), float(np.mean(pars))
        res[f'{task}_splithalf_raw'] = rh_raw
        res[f'{task}_splithalf_partial'] = rh_par
        # Spearman-Brown: half-group -> full-group reliability
        res[f'{task}_fullgroup_raw'] = 2 * rh_raw / (1 + rh_raw)
        res[f'{task}_fullgroup_partial'] = 2 * rh_par / (1 + rh_par)
        print(f'  {task}: split-half raw={rh_raw:.4f} partial={rh_par:.4f} -> '
              f"full-group raw={res[f'{task}_fullgroup_raw']:.4f} "
              f"partial={res[f'{task}_fullgroup_partial']:.4f}", flush=True)

    # between-task values from the companion run
    bt = {'raw': 0.6584, 'partial': 0.5245}
    for m in ['raw', 'partial']:
        ceil = float(np.sqrt(res[f'social_fullgroup_{m}'] * res[f'describe_fullgroup_{m}']))
        res[f'ceiling_geomean_{m}'] = ceil
        res[f'between_task_{m}'] = bt[m]
        res[f'between_over_ceiling_{m}'] = bt[m] / ceil
        res[f'task_diff_index_{m}'] = 1 - bt[m] / ceil

    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'within_task_ceiling.json'), 'w'), indent=2)

    print('\n=== GROUP-LEVEL between-task vs its own ceiling ===')
    for m in ['raw', 'partial']:
        print(f'\n--- {m.upper()} ---')
        print(f"  social   full-group reliability : {res[f'social_fullgroup_{m}']:.4f}")
        print(f"  describe full-group reliability : {res[f'describe_fullgroup_{m}']:.4f}")
        print(f"  geometric mean (ceiling)        : {res[f'ceiling_geomean_{m}']:.4f}")
        print(f"  BETWEEN-TASK                    : {res[f'between_task_{m}']:.4f}")
        print(f"  between / ceiling               : {res[f'between_over_ceiling_{m}']:.4f}")
        print(f"  task-differentiation index      : {res[f'task_diff_index_{m}']:.4f}")


if __name__ == '__main__':
    main()
