"""Build fsaverage5 ROI masks from NSD's `streams` atlas.

Motivation: selecting vertices by ncsnr alone concentrates the analysis in early visual
cortex, where the highest-SNR vertices live and where local contrast dominates responses
almost by definition. That biases the model comparison toward whichever map most resembles
a contrast map -- which is exactly what the EdgeEnergy baseline is designed to expose.
Splitting by stream separates "does this map look like local contrast" (early) from "does
this map track higher-order scene content" (ventral/lateral/parietal).

NSD's streams atlas: 1 early, 2 midventral, 3 midlateral, 4 midparietal, 5 ventral,
6 lateral, 7 parietal. Stored in fsaverage (163842/hemi); fsaverage5 is the first 10242
vertices of each hemisphere (verified to 1e-7), so the mask slices directly.

Output: rois/roi_masks_fs5.npz -- one boolean (20484,) mask per ROI, lh then rh.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).parent
ROIS = HERE / "rois"
VERTS_FS5 = 10242

STREAMS = {1: "early", 2: "midventral", 3: "midlateral", 4: "midparietal",
           5: "ventral", 6: "lateral", 7: "parietal"}
# Coarser groupings that carry the actual hypothesis
GROUPS = {
    "early": ["early"],
    "ventral": ["midventral", "ventral"],
    "lateral": ["midlateral", "lateral"],
    "parietal": ["midparietal", "parietal"],
}


def load_hemi(name):
    arr = np.asarray(nib.load(str(ROIS / name)).get_fdata()).squeeze()
    return arr[:VERTS_FS5]


lh = load_hemi("lh.streams.mgz")
rh = load_hemi("rh.streams.mgz")
streams = np.concatenate([lh, rh]).astype(int)

lh_g = load_hemi("lh.nsdgeneral.mgz")
rh_g = load_hemi("rh.nsdgeneral.mgz")
nsdgeneral = np.concatenate([lh_g, rh_g]) > 0

masks = {}
for code, name in STREAMS.items():
    masks[name] = streams == code
for gname, members in GROUPS.items():
    masks[f"grp_{gname}"] = np.any([streams == c for c, n in STREAMS.items()
                                    if n in members], axis=0)
masks["nsdgeneral"] = nsdgeneral

np.savez_compressed(ROIS / "roi_masks_fs5.npz", **{k: v for k, v in masks.items()})

nc = np.load(HERE / "responses" / "noise_ceiling.npz")
keep = (nc["ncsnr"] > 0.2) & nc["valid"]
print(f"{'ROI':<16} {'vertices':>9} {'in ncsnr mask':>14} {'mean ceiling':>13}")
for k, v in masks.items():
    both = v & keep
    ceil = nc["ceiling"][both].mean() if both.sum() else np.nan
    print(f"{k:<16} {v.sum():>9} {both.sum():>14} {ceil:>13.3f}")
print(f"\ntotal ncsnr-selected vertices: {keep.sum()}")
print(f"of which in 'early': {(masks['early'] & keep).sum()} "
      f"({100*(masks['early'] & keep).sum()/keep.sum():.0f}%)")
