"""
loo_ceilings.py
---------------
Leave-one-subject-out noise ceilings, and the participant exclusions the Molmo head sweep
was missing.

LOO_Z: raw-metric ceiling (Fisher z) from neurips_final/qwen35_nothink/loo_ceiling.py --
each participant's map against the rank-averaged map of all other participants in the
same task, Spearman, z-averaged over participants within image and then over images.

loo_partial_ceiling(): the same ceiling under the center-controlled metric (partial
Spearman controlling the leave-one-image-out center prior), replacing the pairwise
version in rebuttal/task3_baselines/run_center_controlled.partial_ceiling.

patch_sweep_exclusions(): extract_molmo_alllayers.load_prerank_human read every heatmap
file, so Molmo's Describe sweep averaged 25 participants instead of the canonical 22.
"""
import os, json
import numpy as np
from scipy.stats import rankdata

_F = ('/scratch/gpfs/GRAZIANO/vlm-attention-alignment/neurips_final/final_figures/'
      'qwen35_nothink/blur_sweep/ceilings_pairwise_vs_loo.json')
LOO_Z = {t: v['loo_z'] for t, v in json.load(open(_F)).items()}


def _nrm(x):
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def loo_partial_ceiling(task, imgs, prior_rk):
    """Mean Fisher z of partial Spearman(participant, rank-mean of the others | prior)."""
    import alignment_core as A
    ranked, valid, _ = A.load_human(task, imgs)
    pr_n = _nrm(prior_rk.astype(np.float32))
    per_img = []
    for j in range(len(imgs)):
        vi = [i for i in np.where(valid[:, j])[0] if ranked[i, j].std() > 0]
        if len(vi) < 3:
            continue
        V = ranked[vi, j].astype(np.float64)
        Vn = _nrm(V.astype(np.float32))
        tot = V.sum(0)
        zs = []
        for k in range(len(vi)):
            m = _nrm(rankdata((tot - V[k]) / (len(vi) - 1)).astype(np.float32))
            r_hm, r_hp, r_mp = float(Vn[k] @ m), float(Vn[k] @ pr_n[j]), float(m @ pr_n[j])
            den = np.sqrt(max((1 - r_hp ** 2) * (1 - r_mp ** 2), 1e-12))
            zs.append(np.arctanh(np.clip((r_hm - r_hp * r_mp) / den, -0.999999, 0.999999)))
        per_img.append(np.mean(zs))
    return float(np.mean(per_img))


def patch_sweep_exclusions(EM, A):
    orig = EM.load_prerank_human

    def load_prerank_human(task, imgs):
        real_listdir = os.listdir

        def listdir(path):
            out = real_listdir(path)
            if os.path.abspath(path) == os.path.abspath(EM.HM_DIR[task]):
                out = [f for f in out if f.replace('_heatmaps.npy', '') not in A.EXCLUDE[task]]
            return out
        os.listdir = listdir
        try:
            return orig(task, imgs)
        finally:
            os.listdir = real_listdir
    EM.load_prerank_human = load_prerank_human
