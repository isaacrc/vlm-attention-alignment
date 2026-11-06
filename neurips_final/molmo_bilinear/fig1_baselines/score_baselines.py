"""
score_baselines.py -- exploratory, not in the manuscript.
Figure 1A/B alignment for eight baselines, scored exactly as the models are
(alignment_core.per_image_z: per participant and image, Spearman r between the 256x256 map
and the fixation heatmap, Fisher z, mean over images per participant), unblurred like
the models in Figure 1:
  Itti-Koch, GBVS, spectral residual   tierB_maps (256x256)
  Center prior   leave-one-image-out mean human fixation map, pooled over both tasks
                 (run_center_controlled.py, as in the Baselines appendix)
  Person, Face   tierB_maps; Face on the 160 images with an annotated head (flat elsewhere)
  Edge energy, Luminance   per 14x14 cell, as in the encoding analysis, bilinearly
                 upsampled to 256x256 the way the LSTM maps are
Output: final_figures/molmo_bilinear/fig1_baselines/baseline_subject_z.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = Path('/scratch/gpfs/GRAZIANO/vlm-attention-alignment')
sys.path.insert(0, str(ROOT / 'rebuttal' / 'task3_baselines'))
sys.path.insert(0, str(ROOT / 'rebuttal' / 'shared'))
import alignment_core as A, tierB_common as T
import run_center_controlled as CC

OUT = ROOT / 'neurips_final' / 'final_figures' / 'molmo_bilinear' / 'fig1_baselines'
OUT.mkdir(parents=True, exist_ok=True)


def cell_maps(imgs):
    edge, lum = {}, {}
    for im in imgs:
        g = cv2.cvtColor(T.load_rgb(im), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
        mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
        for d, m in ((lum, g), (edge, mag)):
            c = cv2.resize(m, (14, 14), interpolation=cv2.INTER_AREA)
            d[im] = np.asarray(Image.fromarray(c).resize((256, 256), resample=Image.BILINEAR), np.float32)
    return edge, lum


def main():
    imgs = A.target_images()
    cen = CC.loio_prior_ranked(imgs)
    maps = {'Center prior': {im: cen[k] for k, im in enumerate(imgs)},
            'Person prior': T.load_maps('coco_person'),
            'Face prior': T.load_maps('coco_face'),
            'Itti-Koch': T.load_maps('itti_koch'),
            'GBVS': T.load_maps('gbvs'),
            'Spectral residual': T.load_maps('spectral_residual')}
    maps['Edge energy'], maps['Luminance'] = cell_maps(imgs)

    out = []
    for task in ('social', 'describe'):
        for name, m in maps.items():
            df = A.per_image_z(m, task, imgs, name)       # flat maps (no face) are skipped inside
            s = A.subject_means(df); s['n_images'] = df.image_id.nunique()
            out.append(s)
            print(f'{task:8s} {name:18s} r = {np.tanh(s.mean_z.mean()):.3f}  '
                  f'({df.image_id.nunique()} images)', flush=True)
    pd.concat(out).to_csv(OUT / 'baseline_subject_z.csv', index=False)


if __name__ == '__main__':
    main()
