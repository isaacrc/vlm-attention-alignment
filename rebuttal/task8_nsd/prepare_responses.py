"""Turn downloaded trial betas into per-subject image-level response matrices.

Steps per subject:
  1. z-score each vertex WITHIN session (NSD's standard correction for session drift)
  2. average the repeats of each image
  3. split into the shared1000 set (primary) and the overlap set (exploratory)

Also computes the cross-subject noise ceiling on shared-1000, which is only possible
because all 8 subjects saw those same 1000 images.

Outputs (responses/):
  Y_shared_subj{NN}.npy   (n_img, 20484) float32, rows ordered by shared_nsdIds.npy
  Y_overlap_subj{NN}.npy  (n_img, 20484)
  shared_nsdIds.npy       (1000,) image order, identical across subjects
  overlap_nsdIds_subj{NN}.npy
  noise_ceiling.npz       per-vertex ceiling + ncsnr
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BETAS = HERE / "betas"
OUT = HERE / "responses"
OUT.mkdir(exist_ok=True)
SUBJECTS = list(range(1, 9))


def zscore_within_session(betas, sessions):
    """Vertex-wise z-scoring within each scan session."""
    out = np.empty_like(betas)
    for s in np.unique(sessions):
        m = sessions == s
        chunk = betas[m]
        mu = chunk.mean(axis=0, keepdims=True)
        sd = chunk.std(axis=0, keepdims=True)
        out[m] = (chunk - mu) / np.where(sd > 0, sd, 1.0)
    return out


def average_reps(betas, nsd_ids):
    """Mean over repeats. Returns (unique_ids, averaged) with ids sorted."""
    uniq = np.unique(nsd_ids)
    avg = np.empty((len(uniq), betas.shape[1]), dtype=np.float32)
    for i, u in enumerate(uniq):
        avg[i] = betas[nsd_ids == u].mean(axis=0)
    return uniq, avg


shared_ref = None
shared_all = {}

for subj in SUBJECTS:
    path = BETAS / f"betas_subj{subj:02d}.npz"
    if not path.exists():
        print(f"subj{subj:02d}: betas missing, skipping")
        continue
    d = np.load(path, allow_pickle=True)
    betas = zscore_within_session(d["betas"], d["session"])
    image_set, nsd_ids = d["image_set"], d["nsdId"]

    for tag, fname in (("shared1000", "shared"), ("overlap106", "overlap")):
        m = image_set == tag
        if m.sum() == 0:
            continue
        ids, avg = average_reps(betas[m], nsd_ids[m])
        # Subjects 3/4/6/8 stopped early, so their shared set is a subset of the 1000.
        # Keep each subject's own image list rather than assuming a common order.
        np.save(OUT / f"Y_{fname}_subj{subj:02d}.npy", avg)
        np.save(OUT / f"{fname}_nsdIds_subj{subj:02d}.npy", ids)
        if fname == "shared":
            shared_all[subj] = (ids, avg)
        print(f"subj{subj:02d} {fname:8s}: {avg.shape}  "
              f"({m.sum()} trials -> {len(ids)} images)")

# --- cross-subject noise ceiling on shared-1000 -------------------------------------
# All subjects saw the same 1000 images, so between-subject consistency per vertex is a
# genuine upper bound on what any stimulus-driven encoding model can predict.
if len(shared_all) >= 2:
    subs = sorted(shared_all)
    # Restrict to images every subject actually has, so the ceiling is computed on a
    # common stimulus set (subjects 3/4/6/8 are missing ~70-93 of the 1000).
    common = sorted(set.intersection(*(set(shared_all[s][0].tolist()) for s in subs)))
    np.save(OUT / "shared_nsdIds_common.npy", np.array(common))
    print(f"\ncommon shared images across {len(subs)} subjects: {len(common)}")
    stack = np.stack([
        shared_all[s][1][[{int(v): k for k, v in enumerate(shared_all[s][0])}[c] for c in common]]
        for s in subs
    ])  # (n_subj, n_common, n_vert)
    n = len(subs)
    # leave-one-subject-out correlation, Fisher-z averaged
    zs = []
    for i in range(n):
        others = stack[np.arange(n) != i].mean(axis=0)
        a = stack[i] - stack[i].mean(axis=0)
        b = others - others.mean(axis=0)
        denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        # fsaverage5 includes medial-wall vertices with no data; these have zero variance
        # and would produce NaN. Treat them as zero correlation.
        r = np.divide((a * b).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0)
        zs.append(np.arctanh(np.clip(np.nan_to_num(r), -0.999, 0.999)))
    ceiling = np.tanh(np.mean(zs, axis=0))
    ncsnr = np.stack([np.load(BETAS / f"ncsnr_subj{s:02d}.npy") for s in subs]).mean(axis=0)

    # A handful of vertices carry non-finite betas in some subjects (NSD leaves gaps where
    # a subject's surface does not map). NSD's own ncsnr file is finite there, so these
    # would survive an ncsnr threshold and then crash the ridge solver. Exclude any vertex
    # that is non-finite in ANY subject, so every subject is fit on an identical vertex set.
    valid = np.ones(stack.shape[2], dtype=bool)
    for s in subs:
        valid &= np.isfinite(shared_all[s][1]).all(axis=0)
    print(f"valid vertices (finite in all {n} subjects): {valid.sum()} / {len(valid)} "
          f"({(~valid).sum()} excluded)")

    np.savez(OUT / "noise_ceiling.npz", ceiling=ceiling, ncsnr=ncsnr,
             valid=valid, subjects=np.array(subs))
    print(f"\nnoise ceiling over {n} subjects: median r={np.median(ceiling):.3f}  "
          f"max={ceiling.max():.3f}  vertices with r>0.1: {(ceiling > 0.1).sum()}")
    print(f"mean ncsnr: median={np.median(ncsnr):.3f}  "
          f"vertices ncsnr>0.2: {(ncsnr > 0.2).sum()}")
