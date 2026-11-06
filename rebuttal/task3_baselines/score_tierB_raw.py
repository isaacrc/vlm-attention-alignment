"""Raw-metric contrasts between models and the Tier-B baselines (ICLR revision, 2026-09-23).

score_tierB.py tested baselines against models on the center-controlled (partial) metric
only. Isaac's revision reports the RAW metric alone, so the same paired tests are needed on
raw Spearman. Each baseline is taken at its own best RAW sigma (tierB_sigma_sweep.csv),
matching how the raw column of tierB_baselines.csv was built.

Model per-image values come from headrule_center_long.parquet (r_raw), which already uses
the cross-validated per-task heads; the LSTMs and the center prior carry rule='na'.

Output: results/tierB_contrasts_raw.csv
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A
import tierB_common as T

RES = os.path.join(HERE, 'results')
BASELINES = ['coco_person', 'coco_face', 'itti_koch', 'gbvs', 'spectral_residual']
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
N_PERM = 10000


def blur(maps, s):
    return maps if s == 0 else {k: gaussian_filter(v, s) for k, v in maps.items()}


def perm_test(d, rng):
    """Paired sign-flip permutation on per-image differences."""
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d.values).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return obs, obs / d.std(ddof=1), p


def main():
    imgs = A.target_images()
    sweep = pd.read_csv(os.path.join(RES, 'tierB_sigma_sweep.csv'))
    best_raw = (sweep.loc[sweep.groupby(['baseline', 'task'])['raw_z'].idxmax()]
                     .set_index(['baseline', 'task'])['sigma'].to_dict())

    long = pd.read_parquet(os.path.join(RES, 'headrule_center_long.parquet'))
    long = long[long['rule'].isin(['na', 'cv_best'])]          # single map or CV head

    rng = np.random.default_rng(0)
    rows = []
    for name in BASELINES:
        maps = T.load_maps(name)
        if maps is None:
            print(f'  missing maps: {name}'); continue
        for task in ['social', 'describe']:
            s = int(best_raw[(name, task)])
            df = A.per_image_z(blur(maps, s), task, imgs, name)
            base_pim = df.groupby('image_id')['z'].mean()
            for m in MODELS:
                mod_pim = (long[(long.model == m) & (long.task == task)]
                           .groupby('image_id')['r_raw'].mean().apply(np.arctanh))
                common = mod_pim.index.intersection(base_pim.index)
                d = mod_pim.loc[common] - base_pim.loc[common]
                diff, dz, p = perm_test(d, rng)
                rows.append(dict(task=task, model=m, baseline=name, sigma=s,
                                 mean_diff=diff, dz=dz, n_img=len(common), p_perm=p,
                                 model_wins=bool(diff > 0)))
            print(f'  {name} / {task} (sigma={s}) done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'tierB_contrasts_raw.csv'), index=False)
    print(f'\nwrote {RES}/tierB_contrasts_raw.csv  ({len(out)} contrasts)')
    print(out[out.model == 'CNN-LSTM'].to_string(index=False))


if __name__ == '__main__':
    main()
