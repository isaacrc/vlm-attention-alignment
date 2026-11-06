"""Low-level image baselines for the ENCODING analysis.

Why these and not Center/MeanAttn: an encoding model explains variance ACROSS images, so a
map that is identical for every image contributes nothing by construction and returns ~0
regardless of how plausible it looks. Center-bias style baselines are the right control for
per-image fixation alignment (rebuttal Task 3), not for this analysis. The meaningful
control here must vary per image and carry no top-down information.

  Luminance  - mean pixel intensity in each of the 14x14 cells. The simplest thing that
               drives visual cortex. Any attention map must beat this.
  EdgeEnergy - Sobel gradient magnitude averaged per cell: local contrast/spatial
               frequency, the classic bottom-up saliency proxy.
  Shuffled   - (already written by make_baselines.py) real CNN-LSTM maps with image labels
               permuted: preserves map statistics, destroys image correspondence, so it
               estimates the noise floor of the procedure. Should be ~0.

Output: attn/{Luminance,EdgeEnergy}_{describe,social}.npz  (identical across tasks, since
these are properties of the image, but written per task so they flow through unchanged.)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ATTN = HERE / "attn"
GRID = 14

idx = pd.read_csv(HERE / "stimuli_index.csv")
nsd_ids = idx["nsdId"].to_numpy()
n = len(nsd_ids)

lum = np.zeros((n, GRID * GRID), dtype=np.float32)
edge = np.zeros((n, GRID * GRID), dtype=np.float32)

for i, nsd_id in enumerate(nsd_ids):
    img = cv2.imread(str(HERE / "stimuli" / f"{int(nsd_id):05d}.png"), cv2.IMREAD_GRAYSCALE)
    g = img.astype(np.float32) / 255.0
    lum[i] = cv2.resize(g, (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge[i] = cv2.resize(mag, (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()
    if (i + 1) % 250 == 0:
        print(f"  {i+1}/{n}")

blank = np.array([""] * n, dtype=object)
for task in ("describe", "social"):
    np.savez_compressed(ATTN / f"Luminance_{task}.npz",
                        nsdId=nsd_ids, alphas=lum, caption=blank)
    np.savez_compressed(ATTN / f"EdgeEnergy_{task}.npz",
                        nsdId=nsd_ids, alphas=edge, caption=blank)

print(f"\nLuminance:  cross-image std {lum.std(axis=0).mean():.4f}")
print(f"EdgeEnergy: cross-image std {edge.std(axis=0).mean():.4f}")
print("written to attn/")
