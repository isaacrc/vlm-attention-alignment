"""Baseline attention maps, written in the same format as the model maps so they flow
through the identical encoding pipeline.

Without these, "model X's attention best predicts the brain" is not interpretable: a
smooth low-frequency map might predict visual cortex for trivial retinotopic reasons. These
are the reference lines that make the model numbers mean something. Same logic as rebuttal
Task 3, applied to the encoding model rather than to fixation alignment.

  Center      - isotropic Gaussian at image centre, identical for every image.
  NOTE: Center is retained only as a pipeline sanity check. Because it is identical for
  every image it CANNOT explain across-image variance and must return ~0; a non-zero value
  would indicate a bug. The substantive low-level controls live in
  make_lowlevel_baselines.py (Luminance, EdgeEnergy), which vary per image.
  Shuffled    - CNN-LSTM's real maps with the image labels permuted. Preserves the exact
                spatial statistics of a real model but destroys image correspondence, so
                it estimates the noise floor of the whole procedure. Should give ~0.

Output: attn/{Center,Shuffled}_{describe,social}.npz
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ATTN = HERE / "attn"
GRID = 14
RNG = np.random.default_rng(42)

ref = np.load(ATTN / "CNN_LSTM_social.npz", allow_pickle=True)
nsd_ids = ref["nsdId"]
n = len(nsd_ids)
print(f"{n} images")


def center_gaussian(sigma_frac=0.3):
    y, x = np.mgrid[0:GRID, 0:GRID]
    c = (GRID - 1) / 2
    s = sigma_frac * GRID
    g = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * s ** 2))
    return (g / g.sum()).ravel().astype(np.float32)


for task in ("describe", "social"):
    cg = center_gaussian()
    np.savez_compressed(ATTN / f"Center_{task}.npz", nsdId=nsd_ids,
                        alphas=np.tile(cg, (n, 1)),
                        caption=np.array([""] * n, dtype=object))
    real = np.load(ATTN / f"CNN_LSTM_{task}.npz", allow_pickle=True)["alphas"]
    np.savez_compressed(ATTN / f"Shuffled_{task}.npz", nsdId=nsd_ids,
                        alphas=real[RNG.permutation(n)],
                        caption=np.array([""] * n, dtype=object))

    print(f"{task}: Center (sigma=0.3), Shuffled")

print("\nbaselines written to attn/")
