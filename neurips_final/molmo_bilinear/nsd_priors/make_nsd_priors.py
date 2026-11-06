"""
make_nsd_priors.py -- exploratory, not in the manuscript.
Person and face priors for the NSD stimuli, built as the behavioural priors are
(rebuttal/task3_baselines/make_maps_coco.py), restricted to the NSD images that contain
at least one annotated head (the same rule that defines the behavioural face subset).

  Person  union of COCO person segmentation masks
  Face    Gaussian blob on every annotated head (nose/eyes/ears keypoints), sigma from
          the keypoint spread (or the person's bbox height when one keypoint is visible)

Each map is drawn on the full COCO image, cropped with NSD's cropBox
(top, bottom, left, right fractions; nsd_stim_info_merged.csv) so it lines up with the
stimulus participants saw, and reduced to the 14x14 encoding grid with INTER_AREA.
Permutation floors are the maps with image labels permuted (seed 42), as for every
other predictor.

Writes task8_nsd/attn/{PriorPerson,PriorFace,ShufPriorPerson,ShufPriorFace}_{task}.npz
(identical across tasks), task8_nsd/attn/face_subset_nsdIds.npy, and
final_figures/molmo_bilinear/nsd_priors/crop_check.png
"""
import ast, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
from pycocotools.coco import COCO
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
NSD = ROOT / 'rebuttal' / 'task8_nsd'
ANN = ROOT / 'rebuttal' / 'data' / 'coco_trainval2017_ann' / 'annotations'
OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'nsd_priors'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'rebuttal' / 'task3_baselines'))
from make_maps_coco import HEAD_KP, gaussian_blob

SEED = 42
GRID = 14


def crop(m, box):
    top, bottom, left, right = box
    H, W = m.shape
    return m[int(round(top * H)):H - int(round(bottom * H)), int(round(left * W)):W - int(round(right * W))]


def main():
    idx = pd.read_csv(NSD / 'stimuli_index.csv')
    info = pd.read_csv(ROOT / 'rebuttal' / 'data' / 'nsd_stim_info_merged.csv').set_index('nsdId')
    api = {}
    for split in ('train2017', 'val2017'):
        api[split] = (COCO(str(ANN / f'instances_{split}.json')),
                      COCO(str(ANN / f'person_keypoints_{split}.json')))
    person_cat = api['val2017'][0].getCatIds(catNms=['person'])[0]

    ids, P, F, full = [], [], [], {}
    n_person = 0
    for nsd_id in idx.nsdId:
        row = info.loc[nsd_id]
        coco, kps = api[row.cocoSplit]
        cid = int(row.cocoId); box = ast.literal_eval(row.cropBox)
        meta = coco.loadImgs(cid)[0]; H, W = meta['height'], meta['width']

        m_person = np.zeros((H, W), np.float32)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=cid, catIds=[person_cat], iscrowd=None)):
            m_person = np.maximum(m_person, coco.annToMask(ann).astype(np.float32))
        n_person += int(m_person.any())

        m_face = np.zeros((H, W), np.float32); got = False
        for ann in kps.loadAnns(kps.getAnnIds(imgIds=cid, iscrowd=None)):
            if ann.get('num_keypoints', 0) == 0:
                continue
            k = np.array(ann['keypoints'], np.float32).reshape(-1, 3)
            vis = k[HEAD_KP][k[HEAD_KP][:, 2] > 0]
            if len(vis) == 0:
                continue
            cx, cy = float(vis[:, 0].mean()), float(vis[:, 1].mean())
            if len(vis) >= 2:
                d = np.sqrt(((vis[:, None, :2] - vis[None, :, :2]) ** 2).sum(-1))
                sigma = max(float(d.max()) * 0.75, 4.0)
            else:
                sigma = max(float(ann['bbox'][3]) / 8.0, 4.0)
            m_face = np.maximum(m_face, gaussian_blob((H, W), cx, cy, sigma))
            got = True

        cp, cf = crop(m_person, box), crop(m_face, box)
        # a head can sit in the cropped-away margin: keep only images whose cropped maps carry signal
        if not got or cf.max() < 0.5 or cp.sum() == 0:
            continue
        ids.append(int(nsd_id))
        P.append(cv2.resize(cp, (GRID, GRID), interpolation=cv2.INTER_AREA).ravel())
        F.append(cv2.resize(cf, (GRID, GRID), interpolation=cv2.INTER_AREA).ravel())
        if len(full) < 6:
            full[int(nsd_id)] = (cp, cf)

    ids = np.array(ids); P = np.stack(P).astype(np.float32); F = np.stack(F).astype(np.float32)
    print(f'NSD images: {len(idx)}; with a person mask: {n_person}; '
          f'with a head inside the crop (subset): {len(ids)}', flush=True)
    np.save(NSD / 'attn' / 'face_subset_nsdIds.npy', ids)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    blank = np.array([''] * len(ids), dtype=object)
    for name, X in (('PriorPerson', P), ('PriorFace', F)):
        for task in ('social', 'describe'):
            np.savez_compressed(NSD / 'attn' / f'{name}_{task}.npz', nsdId=ids, alphas=X, caption=blank)
            np.savez_compressed(NSD / 'attn' / f'Shuf{name}_{task}.npz', nsdId=ids, alphas=X[perm], caption=blank)
        print(f'{name}: {X.shape}', flush=True)

    # crop check: priors over the NSD stimulus they should match
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
    for c, (nid, (cp, cf)) in enumerate(full.items()):
        img = np.asarray(Image.open(NSD / 'stimuli' / f'{nid:05d}.png').convert('RGB'))
        for r, (m, lab) in enumerate(((cp, 'person'), (cf, 'face'))):
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(cv2.resize(m, img.shape[1::-1]), cmap='magma', alpha=0.5)
            ax.set_title(f'{nid} {lab}', fontsize=10); ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT / 'crop_check.png', dpi=110)


if __name__ == '__main__':
    main()
