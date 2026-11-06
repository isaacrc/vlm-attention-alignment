"""
Tier-B, part 1 -- COCO semantic priors.

Answers the reviewer question the center prior does not: "are the models doing
anything beyond finding the people/objects?" Three ground-truth-annotation maps,
none of which involve a model at all:

  coco_person   union of person segmentation masks
  coco_objects  union of ALL instance masks (generic objectness)
  coco_face     Gaussian blobs on annotated head keypoints (nose/eyes/ears).
                This is the sharpest alternative explanation for the SOCIAL task.

  coco_person_letterbox  identical to coco_person but with aspect-preserving
                padding instead of a square stretch -- an empirical check on the
                pipeline's geometry assumption (see tierB_common docstring).

Outputs: data/tierB_maps/{name}.npz  (raw, unblurred; score_tierB sweeps sigma)
"""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
import tierB_common as T
import alignment_core as A

from pycocotools.coco import COCO

HEAD_KP = [0, 1, 2, 3, 4]   # nose, left_eye, right_eye, left_ear, right_ear


def gaussian_blob(shape, cx, cy, sigma):
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    s2 = max(sigma, 1.0) ** 2
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s2)))


def main():
    imgs = A.target_images()
    coco = COCO(os.path.join(T.COCODIR, 'instances_val2017.json'))
    kps = COCO(os.path.join(T.COCODIR, 'person_keypoints_val2017.json'))
    person_cat = coco.getCatIds(catNms=['person'])[0]

    person, objects, face, person_lb = {}, {}, {}, {}
    n_face_imgs = 0

    for name in imgs:
        img_id = int(os.path.splitext(name)[0])
        meta = coco.loadImgs(img_id)[0]
        H, W = meta['height'], meta['width']

        # ── person + all-object masks (native resolution, then square-resized) ──
        m_person = np.zeros((H, W), np.float32)
        m_all = np.zeros((H, W), np.float32)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None)):
            m = coco.annToMask(ann).astype(np.float32)
            m_all = np.maximum(m_all, m)
            if ann['category_id'] == person_cat:
                m_person = np.maximum(m_person, m)

        person[name] = T.to_square(m_person)
        objects[name] = T.to_square(m_all)
        person_lb[name] = T.to_square(m_person, letterbox=True)

        # ── face prior from head keypoints ──
        m_face = np.zeros((H, W), np.float32)
        got = False
        for ann in kps.loadAnns(kps.getAnnIds(imgIds=img_id, iscrowd=None)):
            if ann.get('num_keypoints', 0) == 0:
                continue
            k = np.array(ann['keypoints'], np.float32).reshape(-1, 3)
            head = k[HEAD_KP]
            vis = head[head[:, 2] > 0]
            if len(vis) == 0:
                continue
            cx, cy = float(vis[:, 0].mean()), float(vis[:, 1].mean())
            if len(vis) >= 2:
                d = np.sqrt(((vis[:, None, :2] - vis[None, :, :2]) ** 2).sum(-1))
                sigma = max(float(d.max()) * 0.75, 4.0)
            else:
                # single keypoint: scale off the person's bbox height
                sigma = max(float(ann['bbox'][3]) / 8.0, 4.0)
            m_face = np.maximum(m_face, gaussian_blob((H, W), cx, cy, sigma))
            got = True
        n_face_imgs += int(got)
        face[name] = T.to_square(m_face)

    print(f'images: {len(imgs)}; with >=1 annotated head: {n_face_imgs}', flush=True)
    frac_person = np.mean([person[n].mean() for n in imgs])
    print(f'mean person-mask coverage: {frac_person:.3f}', flush=True)

    T.save_maps('coco_person', person)
    T.save_maps('coco_objects', objects)
    T.save_maps('coco_face', face)
    T.save_maps('coco_person_letterbox', person_lb)


if __name__ == '__main__':
    main()
