"""Verify that reconstructing NSD stimuli from raw COCO jpg + cropBox reproduces the
images NSD actually presented, so we can build stimuli from COCO downloads (~220 MB)
instead of pulling the 39.6 GB nsd_stimuli.hdf5.

Ground truth: a handful of images read out of nsd_stimuli.hdf5 over HTTP range requests.
Test images: shared-1000 members that already exist in the local COCO train2017 subset.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from http_h5 import HTTPRangeFile, NSD_STIMULI_URL

REPO = Path("/scratch/gpfs/GRAZIANO/vlm-attention-alignment")
COCO_TRAIN = Path("/scratch/gpfs/GRAZIANO/CANS/rnn_model/data/coco_imgs/train2017")

nsd = pd.read_csv(REPO / "rebuttal" / "data" / "nsd_stim_info_merged.csv", index_col=0)
shared = nsd[nsd["shared1000"]].copy()
shared["local"] = [(COCO_TRAIN / f"{c:012d}.jpg").exists() for c in shared["cocoId"]]
cands = shared[shared["local"]].head(4)
print(f"testing {len(cands)} shared-1000 images that exist locally\n")


def crop_from_coco(path, cropbox, out_size=425):
    """cropBox = (top, bottom, left, right) as fractions of the ORIGINAL dimension
    to remove. NSD then resized the square result to 425x425."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    top, bottom, left, right = cropbox
    box = (
        int(round(left * W)),
        int(round(top * H)),
        W - int(round(right * W)),
        H - int(round(bottom * H)),
    )
    return np.asarray(im.crop(box).resize((out_size, out_size), Image.LANCZOS))


f = HTTPRangeFile(NSD_STIMULI_URL)
print(f"remote hdf5: {f.size / 1e9:.2f} GB")
with h5py.File(f, "r", driver="fileobj") as h5:
    print("datasets:", {k: (h5[k].shape, h5[k].dtype, h5[k].chunks) for k in h5.keys()}, "\n")
    brick = h5["imgBrick"]
    for _, row in cands.iterrows():
        truth = brick[int(row["nsdId"])]  # (425, 425, 3) uint8
        mine = crop_from_coco(
            COCO_TRAIN / f"{int(row['cocoId']):012d}.jpg", eval(row["cropBox"])
        )
        d = np.abs(truth.astype(int) - mine.astype(int))
        corr = np.corrcoef(truth.ravel(), mine.ravel())[0, 1]
        print(
            f"coco {int(row['cocoId']):012d} nsdId {int(row['nsdId']):5d} | "
            f"shape {truth.shape} vs {mine.shape} | mean|diff| {d.mean():6.2f} | "
            f"max|diff| {d.max():3d} | r {corr:.6f}"
        )
