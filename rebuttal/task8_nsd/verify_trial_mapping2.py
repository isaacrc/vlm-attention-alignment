"""Stronger version of the trial->frame mapping test.

Two improvements over v1:
  (a) restrict to high-ncsnr vertices, so the comparison is driven by stimulus-responsive
      cortex rather than global signal;
  (b) run the same test under deliberately WRONG mappings (frame offset by -1 and +1).
      The correct mapping must beat both, otherwise we have no evidence of alignment.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
HDR, VERTS_FS7, VERTS_FS5, TPS = 284, 163842, 10242, 750
SUBJ, N_IMGS = 1, 10

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
CACHE = Path(__file__).parent / "_cache"
CACHE.mkdir(exist_ok=True)


def get(url, start=None, end=None):
    h = {"Range": f"bytes={start}-{end}"} if start is not None else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h)) as r:
        return r.read()


# --- ncsnr, to pick stimulus-responsive vertices -----------------------------------
nc_path = CACHE / f"subj{SUBJ:02d}_lh.ncsnr.mgh"
if not nc_path.exists():
    nc_path.write_bytes(get(f"{BUCKET}/nsddata_betas/ppdata/subj{SUBJ:02d}/fsaverage/"
                            f"betas_fithrf_GLMdenoise_RR/lh.ncsnr.mgh"))
ncsnr = np.frombuffer(nc_path.read_bytes()[HDR:], dtype=">f4").astype(np.float32)[:VERTS_FS5]
sel = np.argsort(ncsnr)[-1000:]
print(f"ncsnr over fsaverage5 lh: median {np.median(ncsnr):.3f}, "
      f"top-1000 vertex threshold {ncsnr[sel].min():.3f}, max {ncsnr.max():.3f}\n")


def fetch(subj, trial, offset=0):
    ses, frame = divmod(trial - 1 + offset, TPS)
    ses += 1
    if not 1 <= ses <= 40:
        return None
    url = (f"{BUCKET}/nsddata_betas/ppdata/subj{subj:02d}/fsaverage/"
           f"betas_fithrf_GLMdenoise_RR/lh.betas_session{ses:02d}.mgh")
    start = HDR + frame * VERTS_FS7 * 4
    b = get(url, start, start + VERTS_FS5 * 4 - 1)
    return np.frombuffer(b, dtype=">f4").astype(np.float32)[sel]


seen = nsd[nsd[f"subject{SUBJ}"] == 1]
rep_cols = [f"subject{SUBJ}_rep{k}" for k in range(3)]
imgs = seen[(seen[rep_cols] != 0).all(axis=1)].head(N_IMGS)

for offset, label in [(0, "correct mapping"), (-1, "off-by-one (-1)"), (1, "off-by-one (+1)")]:
    data = {}
    for _, row in imgs.iterrows():
        arrs = [fetch(SUBJ, int(row[c]), offset) for c in rep_cols]
        if any(a is None for a in arrs):
            continue
        data[int(row["nsdId"])] = np.stack(arrs)
    keys = list(data)
    within, between = [], []
    for i, k in enumerate(keys):
        r = np.corrcoef(data[k])
        within += [r[0, 1], r[0, 2], r[1, 2]]
        for k2 in keys[i + 1:]:
            between += [np.corrcoef(data[k][a], data[k2][b])[0, 1]
                        for a in range(3) for b in range(3)]
    w, b = np.mean(within), np.mean(between)
    print(f"{label:18s}  within {w:+.4f}   between {b:+.4f}   diff {w-b:+.4f}")
