"""
Shared helpers for the Tier-B baselines (Task 3).

Every Tier-B baseline produces one {image_name: (256,256) float32} dict, cached to
data/tierB_maps/{name}.npz. score_tierB.py then pushes each through the *identical*
pipeline used for the models, on both the raw and the center-controlled metric, so
the numbers drop straight into the existing tables.

Geometry note: the paper's pipeline square-resizes every image to 256x256 (see
ISAACS_NOTES/model_pipeline_reference.txt) and compares against 336x336 human maps
resized the same way, i.e. everything lives in a stretched square frame. COCO masks
must therefore be stretched identically. `coco_person_letterbox` is generated as an
empirical check on that assumption -- if the display had preserved aspect ratio, the
letterboxed version would align better.
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MAPDIR = os.path.join(HERE, '..', 'data', 'tierB_maps')
COCODIR = os.path.join(HERE, '..', 'data', 'coco')
IMGDIR = ('/scratch/gpfs/GRAZIANO/CANS/rnn_model/archive_analysis_emmet/'
          'analysis/data/val2017')
SIZE = 256


def save_maps(name, maps):
    os.makedirs(MAPDIR, exist_ok=True)
    path = os.path.join(MAPDIR, f'{name}.npz')
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in maps.items()})
    print(f'  wrote {path}  ({len(maps)} images)', flush=True)


def load_maps(name):
    path = os.path.join(MAPDIR, f'{name}.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def has_maps(name):
    return os.path.exists(os.path.join(MAPDIR, f'{name}.npz'))


def img_path(img_name):
    return os.path.join(IMGDIR, img_name)


def load_rgb(img_name):
    return np.asarray(Image.open(img_path(img_name)).convert('RGB'))


def to_square(arr, size=SIZE, letterbox=False, resample=Image.BILINEAR):
    """(H,W) or (H,W,3) -> (size,size). Stretch by default; letterbox pads to a
    square with zeros first, preserving aspect ratio."""
    a = np.asarray(arr)
    if letterbox:
        h, w = a.shape[:2]
        s = max(h, w)
        pad = ((( s - h) // 2, s - h - (s - h) // 2),
               ((s - w) // 2, s - w - (s - w) // 2))
        if a.ndim == 3:
            pad = pad + ((0, 0),)
        a = np.pad(a, pad, mode='constant')
    im = Image.fromarray(a.astype(np.float32) if a.ndim == 2 else a.astype(np.uint8))
    return np.asarray(im.resize((size, size), resample), dtype=np.float32)


def normalize(a):
    a = np.asarray(a, np.float32)
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn + 1e-10)
