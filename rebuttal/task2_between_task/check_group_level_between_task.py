"""
Task 2 sanity check — does the disattenuated 0.911 hold up on a direct estimate?

The reported 0.911 comes from dividing an observed between-subject, between-task
correlation (0.237) by the geometric mean of two within-task between-subject
reliabilities (0.272, 0.248). Two things about that are worth testing directly.

1. If the construct of interest is the GROUP-AVERAGE fixation map for each task,
   the correlation between the two group means can just be computed. No
   correction formula, no reliability estimate. This is the transparent version
   of the same quantity, and n=27/n=22 is enough to estimate a group mean well.

2. Task 3 showed the raw Spearman metric is dominated by center bias: an
   image-independent center prior scores r=0.209/0.237 against human maps, which
   is most of the 0.272/0.248 within-task reliability. Center bias is identical
   in both tasks and across all subjects, so it inflates the numerator AND the
   denominator of the attenuation correction, pushing the ratio toward 1
   regardless of whether image-specific attention differs by task. Re-running
   with the center prior partialled out is the test of whether 0.911 means
   "humans look at the same things in both tasks" or merely "center bias is
   center bias in both tasks".

Reference points computed alongside, all at the group level so they are
like-for-like:
  - split-half WITHIN task (group mean of half the subjects vs the other half),
    which is the ceiling for a group-level between-task correlation
  - between-task group correlation, raw and center-partialled

Everything is computed on rank-transformed maps; ranks for the fixed quantities
(the two group maps and the leave-one-image-out prior) are cached once, since
rankdata over 65536 elements is the dominant cost.

Outputs: results/group_level_between_task.json (printed as a table too)
"""
import os, sys, json, time
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A

RES = os.path.join(HERE, 'results')
RNG = np.random.default_rng(0)
N_SPLIT = 20   # split-half repetitions (20 is ample; SEM over splits is tiny)


def group_mean_ranked(task, imgs, sub_idx=None):
    """(N, D) group-mean rank map per image. sub_idx selects a subject subset."""
    ranked, valid, subs = A.load_human(task, imgs)
    if sub_idx is not None:
        ranked, valid = ranked[sub_idx], valid[sub_idx]
    N, D = ranked.shape[1], ranked.shape[2]
    out = np.zeros((N, D), np.float64)
    for i in range(N):
        vi = np.where(valid[:, i])[0]
        out[i] = ranked[vi, i, :].mean(axis=0) if len(vi) else np.nan
    return out, len(subs)


def rank_rows(G):
    """Rank-transform each row, once. NaN rows preserved."""
    out = np.empty_like(G)
    for i in range(len(G)):
        if np.isnan(G[i]).any():
            out[i] = np.nan
        else:
            out[i] = rankdata(G[i])
    return out


def _z(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else None


def corr_rows(RA, RB):
    """Per-row Pearson on pre-ranked rows == Spearman. -> (N,)"""
    out = np.full(len(RA), np.nan)
    for i in range(len(RA)):
        if np.isnan(RA[i]).any() or np.isnan(RB[i]).any():
            continue
        a, b = _z(RA[i]), _z(RB[i])
        if a is None or b is None:
            continue
        out[i] = float(np.dot(a, b) / len(a))
    return out


def partial_rows(RA, RB, RC):
    """Per-row partial correlation of A,B controlling C, all pre-ranked."""
    out = np.full(len(RA), np.nan)
    for i in range(len(RA)):
        if np.isnan(RA[i]).any() or np.isnan(RB[i]).any() or np.isnan(RC[i]).any():
            continue
        a, b, c = _z(RA[i]), _z(RB[i]), _z(RC[i])
        if a is None or b is None or c is None:
            continue
        n = len(a)
        rab = float(np.dot(a, b) / n)
        rac = float(np.dot(a, c) / n)
        rbc = float(np.dot(b, c) / n)
        den = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
        if den < 1e-12:
            continue
        out[i] = (rab - rac * rbc) / den
    return out


def fisher_mean(r):
    r = r[~np.isnan(r)]
    return float(np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean()))


def loio_prior(G):
    """Leave-one-image-out center prior: mean of all OTHER images' group maps."""
    tot = np.nansum(G, axis=0)
    n = np.sum(~np.isnan(G[:, 0]))
    return (tot[None, :] - G) / (n - 1)


def main():
    t0 = time.time()
    imgs = A.target_images()
    G_soc, n_soc = group_mean_ranked('social', imgs)
    G_des, n_des = group_mean_ranked('describe', imgs)
    print(f'group maps: social n={n_soc}, describe n={n_des}, {len(imgs)} images', flush=True)

    # center prior pooled over BOTH tasks, leave-one-image-out
    P = loio_prior((G_soc + G_des) / 2.0)

    R_soc, R_des, R_P = rank_rows(G_soc), rank_rows(G_des), rank_rows(P)
    print(f'  ranked fixed maps ({time.time()-t0:.0f}s)', flush=True)

    res = {}

    # ── between-task, group level ──
    res['between_task_group_raw'] = fisher_mean(corr_rows(R_soc, R_des))
    res['between_task_group_partial'] = fisher_mean(partial_rows(R_soc, R_des, R_P))
    print(f"\n  BETWEEN-TASK raw     : {res['between_task_group_raw']:.4f}")
    print(f"  BETWEEN-TASK partial : {res['between_task_group_partial']:.4f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)

    # ── within-task split-half ceiling, group level (the like-for-like reference) ──
    for task in ['social', 'describe']:
        ranked, valid, subs = A.load_human(task, imgs)
        S = len(subs)
        raws, pars = [], []
        for k in range(N_SPLIT):
            perm = RNG.permutation(S)
            h1, h2 = perm[:S // 2], perm[S // 2:]
            A1, _ = group_mean_ranked(task, imgs, h1)
            A2, _ = group_mean_ranked(task, imgs, h2)
            R1, R2 = rank_rows(A1), rank_rows(A2)
            raws.append(fisher_mean(corr_rows(R1, R2)))
            pars.append(fisher_mean(partial_rows(R1, R2, R_P)))
        res[f'within_{task}_splithalf_raw'] = float(np.mean(raws))
        res[f'within_{task}_splithalf_partial'] = float(np.mean(pars))
        print(f'  {task:9s} split-half raw={np.mean(raws):.4f} '
              f'partial={np.mean(pars):.4f}  ({time.time()-t0:.0f}s)', flush=True)

    # ── the comparison that matters ──
    # NOTE: split-half compares two half-groups; the between-task correlation
    # compares two FULL groups. Spearman-Brown steps the split-half value up to
    # full-group reliability so the two are on the same footing.
    for metric in ['raw', 'partial']:
        sb = {}
        for task in ['social', 'describe']:
            rh = res[f'within_{task}_splithalf_{metric}']
            sb[task] = 2 * rh / (1 + rh)          # Spearman-Brown, half -> full
            res[f'within_{task}_fullgroup_{metric}'] = float(sb[task])
        ceil = float(np.sqrt(sb['social'] * sb['describe']))
        res[f'ceiling_geomean_{metric}'] = ceil
        res[f'between_over_ceiling_{metric}'] = float(
            res[f'between_task_group_{metric}'] / ceil)

    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'group_level_between_task.json'), 'w'), indent=2)

    print('\n=== GROUP-LEVEL between-task (no disattenuation formula) ===')
    for metric in ['raw', 'partial']:
        print(f'\n--- {metric.upper()} ---')
        print(f"  within social   full-group : {res[f'within_social_fullgroup_{metric}']:.4f}")
        print(f"  within describe full-group : {res[f'within_describe_fullgroup_{metric}']:.4f}")
        print(f"  geometric mean (ceiling)   : {res[f'ceiling_geomean_{metric}']:.4f}")
        print(f"  BETWEEN-TASK               : {res[f'between_task_group_{metric}']:.4f}")
        print(f"  between / ceiling          : {res[f'between_over_ceiling_{metric}']:.4f}")
        print(f"  task-differentiation index : {1 - res[f'between_over_ceiling_{metric}']:.4f}")


if __name__ == '__main__':
    main()
