"""
score_edge_lum.py -- edge energy and luminance rows for the Appendix E baselines table of
main_2026-10-23.tex, scored exactly as the other table baselines
(rebuttal/task3_baselines/score_tierB.py): raw Spearman r (alignment_core.per_image_z) and
partial Spearman r controlling the leave-one-image-out center prior
(run_center_controlled.partial_rows), each at its own best Gaussian sigma from
{0, 4, 8, 16, 24, 32}, chosen separately per metric and task.
Maps: Sobel edge energy and mean luminance per 14x14 cell, bilinearly upsampled to 256x256
(fig1_baselines/score_baselines.cell_maps), the same predictors as the NSD low-level controls.
Writes novlm_2026-10-23/results/edge_lum_baselines.csv and edge_lum_sigma_sweep.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = '/scratch/gpfs/GRAZIANO/vlm-attention-alignment'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'task3_baselines'))
sys.path.insert(0, os.path.join(ROOT, 'rebuttal', 'shared'))
sys.path.insert(0, os.path.join(ROOT, 'neurips_final', 'molmo_bilinear', 'fig1_baselines'))
import alignment_core as A
import run_center_controlled as CC
from score_baselines import cell_maps

SIGMAS = [0, 4, 8, 16, 24, 32]


def main():
    imgs = A.target_images()
    prior_rk = CC.loio_prior_ranked(imgs)
    edge, lum = cell_maps(imgs)
    sweep = []
    for name, maps in (('edge_energy', edge), ('luminance', lum)):
        for s in SIGMAS:
            bm = maps if s == 0 else {k: gaussian_filter(v, s) for k, v in maps.items()}
            for task in ('social', 'describe'):
                zr = float(A.subject_means(A.per_image_z(bm, task, imgs, name))['mean_z'].mean())
                zp = float(CC.partial_rows(bm, task, imgs, prior_rk, name).groupby('subject_id')['z'].mean().mean())
                sweep.append(dict(baseline=name, task=task, sigma=s, raw_z=zr, raw_r=np.tanh(zr),
                                  partial_z=zp, partial_r=np.tanh(zp)))
                print(f'{name:12s} {task:8s} sigma={s:2d}  raw r={np.tanh(zr):.4f}  partial r={np.tanh(zp):.4f}', flush=True)
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(HERE, 'results', 'edge_lum_sigma_sweep.csv'), index=False)
    rows = []
    for (b, t), g in sw.groupby(['baseline', 'task']):
        br, bp = g.loc[g.raw_z.idxmax()], g.loc[g.partial_z.idxmax()]
        rows.append(dict(baseline=b, task=t, raw_r=br.raw_r, raw_sigma=int(br.sigma),
                         partial_r=bp.partial_r, partial_sigma=int(bp.sigma)))
    best = pd.DataFrame(rows)
    best.to_csv(os.path.join(HERE, 'results', 'edge_lum_baselines.csv'), index=False)
    print(best.to_string(index=False))


if __name__ == '__main__':
    main()
