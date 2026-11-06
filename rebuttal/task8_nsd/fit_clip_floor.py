"""Permutation floor for the CLIP band.

Every attention predictor is referenced to its own maps with the image labels
permuted; CLIP alone had no such floor, so its R^2 was not on the same footing.
This fits CLIP-only with the embedding rows permuted, using the same seed and the
same fit_single path as the real CLIP fit already stored in the encoding npz.

Output: encoding/r2_ShufCLIP_subj{NN}_cvbehav.npz   (task-independent, like CLIP itself)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fit_encoding import fit_single

SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, required=True)
a = ap.parse_args()

out = HERE / "encoding" / f"r2_ShufCLIP_subj{a.subject:02d}_cvbehav.npz"
if out.exists():
    print(f"exists: {out.name}")
    raise SystemExit(0)

resp = HERE / "responses"
Y_full = np.load(resp / f"Y_shared_subj{a.subject:02d}.npy")
order = np.load(resp / f"shared_nsdIds_subj{a.subject:02d}.npy")
nc = np.load(resp / "noise_ceiling.npz")
keep = np.where((nc["ncsnr"] > 0.2) & nc["valid"])[0]
Y = Y_full[:, keep]

X = np.load(HERE / "features" / "X_clip_nsd.npy")
stim = pd.read_csv(HERE / "stimuli_index.csv")
lut = {int(i): k for k, i in enumerate(stim["nsdId"])}
X = X[[lut[int(i)] for i in order]]

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(order))
print(f"subj{a.subject:02d}: Y {Y.shape}, CLIP {X.shape}, "
      f"{(perm != np.arange(len(order))).mean():.1%} of rows displaced")

r2 = fit_single(X[perm], Y)
np.savez(out, r2_clip_shuf=r2, vertices=keep)
print(f"saved {out.name}  mean={r2.mean():.4f}")
