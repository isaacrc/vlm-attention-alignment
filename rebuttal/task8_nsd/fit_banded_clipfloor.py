"""Banded fit with the CLIP band permuted, to floor the CLIP and shared components.

Unique CLIP  = R2(attn + CLIP) - R2(attn);   its floor = R2(attn + shufCLIP) - R2(attn)
Shared       = R2(attn) + R2(CLIP) - R2(attn + CLIP)
               floor   = R2(attn) + R2(shufCLIP) - R2(attn + shufCLIP)

Attention stays real; only CLIP's image labels are permuted, with the same seed as
fit_clip_floor.py so the two CLIP floors use the same draw. r2_attn and r2_clip_shuf
already exist, so this run supplies the one missing term: the banded fit.

Output: encoding/r2_bandedShufCLIP_{model}_{task}_subj{NN}_cvbehav.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from fit_encoding import fit_banded, load_attention

SEED = 42
ap = argparse.ArgumentParser()
ap.add_argument("--subject", type=int, required=True)
ap.add_argument("--model", default="CNN_Transformer")
ap.add_argument("--task", default="social")
a = ap.parse_args()

out = HERE / "encoding" / f"r2_bandedShufCLIP_{a.model}_{a.task}_subj{a.subject:02d}_cvbehav.npz"
if out.exists():
    print(f"exists: {out.name}")
    raise SystemExit(0)

resp = HERE / "responses"
Y_full = np.load(resp / f"Y_shared_subj{a.subject:02d}.npy")
order = np.load(resp / f"shared_nsdIds_subj{a.subject:02d}.npy")
nc = np.load(resp / "noise_ceiling.npz")
keep = np.where((nc["ncsnr"] > 0.2) & nc["valid"])[0]
Y = Y_full[:, keep]

X_clip = np.load(HERE / "features" / "X_clip_nsd.npy")
stim = pd.read_csv(HERE / "stimuli_index.csv")
lut = {int(i): k for k, i in enumerate(stim["nsdId"])}
X_clip = X_clip[[lut[int(i)] for i in order]]

X_attn, ok = load_attention(a.model, a.task, order, "cvbehav")
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(order))
print(f"subj{a.subject:02d} {a.model}/{a.task}: Y {Y.shape}, attn {X_attn.shape}, "
      f"CLIP permuted ({(perm != np.arange(len(order))).mean():.1%} displaced)")

r2, deltas = fit_banded(X_attn, X_clip[perm], Y)
np.savez(out, r2_banded_shufclip=r2, deltas=deltas, vertices=keep)
print(f"saved {out.name}  mean={r2.mean():.4f}")
