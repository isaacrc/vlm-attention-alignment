"""Verify the trial -> (session, frame) mapping before committing to a download.

Assumption under test: subjectN_repK in nsd_stim_info_merged is a 1-indexed trial number
in the subject's full 30,000-trial sequence, and beta frame f of session s corresponds to
trial (s-1)*750 + f + 1.

Test: for images seen 3 times, the 3 repeats should correlate with each other far more
than repeats of different images do. If the mapping were wrong, both would look alike.
"""

import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
HDR, VERTS_FS7, VERTS_FS5, TRIALS_PER_SES = 284, 163842, 10242, 750
SUBJ = 1

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)


def fetch_trial(subj, trial, hemi="lh", n_verts=VERTS_FS5):
    """One trial's betas, restricted to the first n_verts (= fsaverage5 when 10242)."""
    ses, frame = divmod(trial - 1, TRIALS_PER_SES)
    ses += 1
    url = (f"{BUCKET}/nsddata_betas/ppdata/subj{subj:02d}/fsaverage/"
           f"betas_fithrf_GLMdenoise_RR/{hemi}.betas_session{ses:02d}.mgh")
    start = HDR + frame * VERTS_FS7 * 4
    end = start + n_verts * 4 - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return np.frombuffer(r.read(), dtype=">f4").astype(np.float32)


seen = nsd[nsd[f"subject{SUBJ}"] == 1]
rep_cols = [f"subject{SUBJ}_rep{k}" for k in range(3)]
imgs = seen[(seen[rep_cols] != 0).all(axis=1)].head(6)

print(f"subj{SUBJ}: pulling 3 repeats each for {len(imgs)} images (fsaverage5 lh)\n")
data = {}
for _, row in imgs.iterrows():
    trials = [int(row[c]) for c in rep_cols]
    data[int(row["nsdId"])] = np.stack([fetch_trial(SUBJ, t) for t in trials])
    print(f"  nsdId {int(row['nsdId']):5d}  trials {trials}")

within, between = [], []
keys = list(data)
for i, k in enumerate(keys):
    r = np.corrcoef(data[k])
    within += [r[0, 1], r[0, 2], r[1, 2]]
    for k2 in keys[i + 1:]:
        between += [np.corrcoef(data[k][a], data[k2][b])[0, 1] for a in range(3) for b in range(3)]

within, between = np.array(within), np.array(between)
print(f"\nwithin-image repeat correlation : {within.mean():+.4f}  (n={len(within)}, "
      f"range {within.min():+.4f} to {within.max():+.4f})")
print(f"between-image correlation      : {between.mean():+.4f}  (n={len(between)}, "
      f"range {between.min():+.4f} to {between.max():+.4f})")
print(f"difference                     : {within.mean() - between.mean():+.4f}")

ok = within.mean() > between.mean() + 0.02
print("\nVERDICT:", "mapping looks CORRECT" if ok else "MAPPING SUSPECT - do not proceed")
sys.exit(0 if ok else 1)
