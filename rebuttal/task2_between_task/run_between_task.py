"""
Task 2 — Between-task human baseline correction.
Corrects the human between-task reference for between-subject noise (Social n=27 and
Describe n=22 were different participants), via attenuation correction. Recomputes
everything from raw per-subject maps (cached ranked tensors) on the canonical subject
sets. See EXECUTION_PLAN Task 2.

Outputs (task2_between_task/results/):
  corrected_between_task.csv, bootstrap_ci.json, model_between_task.csv
Figure: figures/fig_between_task_corrected.png
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
import alignment_core as A
import vlm_maps as V

RES = os.path.join(os.path.dirname(__file__), 'results'); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures'); os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(0)


def within_reliability(task, imgs):
    """Mean pairwise Fisher-z Spearman across subjects, per image then averaged.
    = human-human noise ceiling (mean-z)."""
    return A.ceiling(task, imgs)   # already pairwise mean-z on canonical set


def normed(ranked):
    c = ranked - ranked.mean(axis=-1, keepdims=True)
    return c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-10)


def human_between_task(imgs):
    """Cross social-subject x describe-subject Spearman per image, Fisher-z averaged.
    Returns (mean_z, per_image_z array, soc_norm, des_norm, valids) for bootstrap."""
    sr, sv, _ = A.load_human('social', imgs)
    dr, dv, _ = A.load_human('describe', imgs)
    sn, dn = normed(sr), normed(dr)
    per_img = []
    for j in range(len(imgs)):
        si = np.where(sv[:, j])[0]; di = np.where(dv[:, j])[0]
        if len(si) == 0 or len(di) == 0:
            per_img.append(np.nan); continue
        C = sn[si, j] @ dn[di, j].T           # (nsoc, ndes)
        z = np.arctanh(np.clip(C, -0.999999, 0.999999))
        per_img.append(z.mean())
    per_img = np.array(per_img)
    return np.nanmean(per_img), per_img, sn, sv, dn, dv


def model_between_task(imgs):
    """Per-image Spearman between each model's social and describe maps (deterministic)."""
    from scipy.stats import spearmanr
    rows = []
    prov = {m: ('trained', m) for m in ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer']}
    prov.update({'Molmo': ('vlm', 'Molmo'), 'Qwen3.5': ('vlm', 'Qwen3.5')})
    for m, (kind, name) in prov.items():
        soc = A.trained_maps(m, 'social', imgs) if kind == 'trained' else V.vlm_maps(m, 'social')
        des = A.trained_maps(m, 'describe', imgs) if kind == 'trained' else V.vlm_maps(m, 'describe')
        rs = []
        for im in imgs:
            if im in soc and im in des:
                a, b = soc[im].flatten(), des[im].flatten()
                if a.std() > 1e-12 and b.std() > 1e-12:
                    r, _ = spearmanr(a, b); rs.append(r)
        rs = np.array(rs)
        rows.append(dict(model=m, between_r=float(rs.mean()),
                         sem=float(rs.std(ddof=1) / np.sqrt(len(rs))), n_img=len(rs)))
    return pd.DataFrame(rows)


def main():
    imgs = A.target_images()
    rel_soc = np.tanh(within_reliability('social', imgs))
    rel_des = np.tanh(within_reliability('describe', imgs))
    obs_z, per_img, sn, sv, dn, dv = human_between_task(imgs)
    obs_r = float(np.tanh(obs_z))
    disatt = obs_r / np.sqrt(rel_soc * rel_des)
    print(f'human within-task reliability (r): social={rel_soc:.4f} describe={rel_des:.4f}')
    print(f'human between-task observed r={obs_r:.4f}  -> disattenuated={disatt:.4f}')

    # bootstrap over images for CI on disattenuated value
    n = len(imgs); boot = []
    for _ in range(2000):
        idx = RNG.integers(0, n, n)
        bz = np.nanmean(per_img[idx])
        br = np.tanh(bz)
        boot.append(br / np.sqrt(rel_soc * rel_des))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    mdf = model_between_task(imgs)
    mdf['task_diff_index'] = 1 - mdf['between_r'] / 1.0   # model reliability = 1.0
    mdf.to_csv(os.path.join(RES, 'model_between_task.csv'), index=False)

    # human task-diff index on corrected scale
    human_tdi = 1 - disatt   # disatt is between/reliability already
    out = pd.DataFrame([
        dict(agent='Human (observed)', between_r=obs_r, reliability=np.sqrt(rel_soc*rel_des),
             between_corrected=disatt, task_diff_index=human_tdi),
    ])
    for _, r in mdf.iterrows():
        out = pd.concat([out, pd.DataFrame([dict(
            agent=r['model'], between_r=r['between_r'], reliability=1.0,
            between_corrected=r['between_r'], task_diff_index=r['task_diff_index'])])],
            ignore_index=True)
    out.to_csv(os.path.join(RES, 'corrected_between_task.csv'), index=False)
    json.dump(dict(observed_r=obs_r, rel_soc=float(rel_soc), rel_des=float(rel_des),
                   disattenuated=float(disatt), ci95=ci), open(os.path.join(RES, 'bootstrap_ci.json'), 'w'), indent=2)
    print('\n=== corrected between-task ===')
    print(out.to_string(index=False))
    print(f'\ndisattenuated human between-task 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')

    # figure
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    order = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
    mmap = {r['model']: r for _, r in mdf.iterrows()}
    # panel A: raw between-task with human observed vs corrected
    ax[0].bar(range(len(order)), [mmap[m]['between_r'] for m in order], color='#4c72b0')
    ax[0].axhline(obs_r, ls='--', c='gray', label=f'Human observed ({obs_r:.2f})')
    ax[0].axhline(disatt, ls='-', c='crimson', label=f'Human corrected ({disatt:.2f})')
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=45, ha='right')
    ax[0].set_ylabel('Between-task similarity (Spearman r)'); ax[0].set_title('Between-task: raw vs corrected human')
    ax[0].legend(); ax[0].set_ylim(0, 1)
    # panel B: task differentiation index
    agents = ['Human'] + order
    tdi = [human_tdi] + [mmap[m]['task_diff_index'] for m in order]
    ax[1].bar(range(len(agents)), tdi, color=['crimson'] + ['#4c72b0']*len(order))
    ax[1].set_xticks(range(len(agents))); ax[1].set_xticklabels(agents, rotation=45, ha='right')
    ax[1].set_ylabel('Task differentiation index (1 - between/reliability)')
    ax[1].set_title('Task differentiation (higher = more task-adaptive)')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_between_task_corrected.png'), dpi=150)
    print('Saved figure.')


if __name__ == '__main__':
    main()
