"""How much does each model's attention map resemble each attention prior?

The alignment analysis asks how well a model predicts human fixation; two maps can tie on
that while looking nothing alike. This asks the direct question instead: the per-image
Spearman correlation between a model's attention map and each prior's map. It is what
licenses a claim that a model "recovers" a bias, which an equal alignment score does not.

Models use the cross-validated per-task head (Task 1). Priors: the leave-one-image-out
center prior used in the center-controlled analysis, and the COCO face and person priors at
their best raw Gaussian blur. The face prior is empty on the 40 of 200 images with no
annotated head; those images are dropped for that prior only, and the count is reported.

Output: results/model_prior_similarity.csv  (per model x prior x task: r, SEM, n images)
"""
import os, sys
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
import alignment_core as A
import tierB_common as T
import vlm_maps as V
from run_center_controlled import loio_prior_ranked

RES = os.path.join(HERE, 'results')
CVHEAD = {'CNN-Transformer': {'social': (4, 4), 'describe': (3, 5)},
          'ViT-Transformer': {'social': (4, 8), 'describe': (3, 8)},
          'Molmo':           {'social': (10, 24), 'describe': (8, 7)},
          'Qwen3.5':         {'social': (15, 12), 'describe': (15, 12)}}
MODELS = ['CNN-LSTM', 'ViT-LSTM', 'CNN-Transformer', 'ViT-Transformer', 'Molmo', 'Qwen3.5']
PRIORS = ['center', 'coco_face', 'coco_person']
LABEL = {'center': 'Center prior', 'coco_face': 'Face prior', 'coco_person': 'Person prior'}


def model_maps(model, task, imgs):
    if model in ('Molmo', 'Qwen3.5'):
        L, H = CVHEAD[model][task]
        return V.vlm_maps(model, task, L, H)
    lh = CVHEAD.get(model, {}).get(task)
    return (A.trained_maps(model, task, imgs, layer=lh[0], head=lh[1]) if lh
            else A.trained_maps(model, task, imgs))


def spearman(a, b):
    return np.corrcoef(rankdata(np.asarray(a).ravel()),
                       rankdata(np.asarray(b).ravel()))[0, 1]


def main():
    imgs = A.target_images()
    sweep = pd.read_csv(os.path.join(RES, 'tierB_sigma_sweep.csv'))
    best_raw = (sweep.loc[sweep.groupby(['baseline', 'task'])['raw_z'].idxmax()]
                     .set_index(['baseline', 'task'])['sigma'].to_dict())
    print('building leave-one-image-out center prior...', flush=True)
    prior_rk = loio_prior_ranked(imgs)          # (N, D), already rank-transformed

    rows = []
    for task in ['social', 'describe']:
        pri = {'center': {im: prior_rk[i].reshape(A.OUT_SIZE, A.OUT_SIZE)
                          for i, im in enumerate(imgs)}}
        for b in ('coco_face', 'coco_person'):
            s = int(best_raw[(b, task)]); m = T.load_maps(b)
            pri[b] = m if s == 0 else {k: gaussian_filter(v, s) for k, v in m.items()}
        for model in MODELS:
            mm = model_maps(model, task, imgs)
            for p in PRIORS:
                vals = [spearman(mm[im], pri[p][im]) for im in imgs
                        if im in mm and im in pri[p]
                        and np.ptp(pri[p][im]) > 0 and np.ptp(mm[im]) > 0]
                z = np.arctanh(np.clip(vals, -.999999, .999999))
                rows.append(dict(model=model, prior=LABEL[p], task=task, n_img=len(vals),
                                 r=float(np.tanh(z.mean())),
                                 sem=float(np.tanh(z.std(ddof=1) / np.sqrt(len(z))))))
            print(f'  {model} / {task} done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'model_prior_similarity.csv'), index=False)
    print('\n' + out.pivot_table(index=['model'], columns=['prior', 'task'],
                                 values='r').round(3).to_string())


if __name__ == '__main__':
    main()
