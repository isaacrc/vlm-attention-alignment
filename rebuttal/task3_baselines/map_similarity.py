"""Map-to-map similarity between model attention and the semantic priors.

Alignment scores tie when two maps predict fixation equally well; that does not mean the
maps resemble each other. This computes the direct per-image Spearman correlation between
each model's attention map and the face / person / center priors, which is what licenses a
claim that a model attends to faces (or does so more in one task than the other).

Priors are taken at the same best-raw sigma used in the alignment table.
Output: results/map_similarity.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A
import tierB_common as T

RES = os.path.join(HERE, 'results')
PRIORS = ['coco_face', 'coco_person']
CVHEAD = {'CNN-Transformer': {'social': (4, 4), 'describe': (3, 5)},
          'ViT-Transformer': {'social': (4, 8), 'describe': (3, 8)}}
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer']


def spearman(a, b):
    return np.corrcoef(rankdata(a.ravel()), rankdata(b.ravel()))[0, 1]


def main():
    imgs = A.target_images()
    sweep = pd.read_csv(os.path.join(RES, 'tierB_sigma_sweep.csv'))
    best_raw = (sweep.loc[sweep.groupby(['baseline', 'task'])['raw_z'].idxmax()]
                     .set_index(['baseline', 'task'])['sigma'].to_dict())
    rows = []
    for task in ['social', 'describe']:
        priors = {}
        for p in PRIORS:
            s = int(best_raw[(p, task)])
            m = T.load_maps(p)
            priors[p] = m if s == 0 else {k: gaussian_filter(v, s) for k, v in m.items()}
        for model in MODELS:
            lh = CVHEAD.get(model, {}).get(task)
            mm = (A.trained_maps(model, task, imgs, layer=lh[0], head=lh[1]) if lh
                  else A.trained_maps(model, task, imgs))
            for p in PRIORS:
                # 40 of 200 images have no annotated head, so the face prior is constant
                # there and the correlation is undefined; those images are dropped.
                vals = [spearman(mm[im], priors[p][im])
                        for im in imgs if im in mm and im in priors[p]
                        and np.ptp(priors[p][im]) > 0 and np.ptp(mm[im]) > 0]
                z = np.arctanh(np.clip(vals, -.999999, .999999))
                rows.append(dict(task=task, model=model, prior=p, n_img=len(vals),
                                 r=float(np.tanh(z.mean())),
                                 sem=float(np.tanh(z.std(ddof=1) / np.sqrt(len(z))))))
            print(f'  {model} / {task} done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'map_similarity.csv'), index=False)
    print(out.pivot_table(index=['model', 'prior'], columns='task', values='r').round(3).to_string())


if __name__ == '__main__':
    main()
