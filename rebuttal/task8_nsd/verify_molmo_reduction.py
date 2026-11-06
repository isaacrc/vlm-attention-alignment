"""Verify the in-loop Molmo reduction reproduces the reference implementation.

The NSD Molmo run replaces a ~55 MB/image raw activation dump with a (28, 28, 196) map
computed in-loop. That reduction is the only new code in an otherwise validated path, and
it relies on numpy advanced indexing whose output axis order is easy to get wrong
(`attn[L, H, :, positions]` puts the broadcast axis FIRST because the advanced indices are
separated by a slice). So compare it against tribe/scripts/2_feature_extraction.py's
_extract_molmo_map on real activations from the existing 200-image run.

Passing here means the NSD maps are computed the same way as the maps already in the paper.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ACT = Path("/scratch/gpfs/GRAZIANO/CANS/molmo/molmo/activations")
TASK = "social"
MOLMO_LAYER, MOLMO_HEAD = 9, 23  # L9 H24 1-indexed, the paper's best social head


def setup_grid(act_dir):
    img_input_idx = torch.load(act_dir / "image_input_idx.pt", map_location="cpu")[0].numpy()
    img_idx_all = (img_input_idx[1:].reshape(3, 3, 12, 12)
                   .transpose(0, 2, 1, 3).reshape(36, 36))
    return img_idx_all, np.where(img_idx_all > 0)


def reference_map(attn, layer_idx, head_idx, img_idx_all, valid_idx, out=14):
    """Verbatim logic from tribe/scripts/2_feature_extraction.py:_extract_molmo_map."""
    n_tokens = attn.shape[2]
    H, W = img_idx_all.shape
    vol = np.zeros((n_tokens, H, W), dtype=np.float32)
    img_positions = img_idx_all[valid_idx]
    for tok in range(n_tokens):
        vol[tok, valid_idx[0], valid_idx[1]] = attn[layer_idx, head_idx, tok, img_positions]
    head_map = vol.sum(axis=0)
    row_valid = np.any(img_idx_all > 0, axis=1)
    col_valid = np.any(img_idx_all > 0, axis=0)
    cropped = head_map[row_valid][:, col_valid]
    return cv2.resize(cropped, (out, out), interpolation=cv2.INTER_AREA)


def new_reduction(attn, img_idx_all, valid_idx, out=14):
    """The vectorised in-loop version used in batch_inference_nsd.py."""
    n_layers, n_heads = attn.shape[0], attn.shape[1]
    img_positions = img_idx_all[valid_idx]
    row_valid = np.any(img_idx_all > 0, axis=1)
    col_valid = np.any(img_idx_all > 0, axis=0)
    res = np.zeros((n_layers, n_heads, out * out), dtype=np.float32)
    for L in range(n_layers):
        for H in range(n_heads):
            vol = np.zeros(img_idx_all.shape, dtype=np.float32)
            vol[valid_idx] = attn[L, H, :, img_positions].sum(axis=-1)
            cropped = vol[row_valid][:, col_valid]
            res[L, H] = cv2.resize(cropped, (out, out),
                                   interpolation=cv2.INTER_AREA).flatten()
    return res


def main():
    act_dir = ACT / TASK
    files = sorted(act_dir.glob("activations_img_*.pt"))[:3]
    if not files:
        print(f"no activations in {act_dir}")
        return 1
    img_idx_all, valid_idx = setup_grid(act_dir)
    print(f"grid {img_idx_all.shape}, {len(valid_idx[0])} valid positions\n")

    worst = 0.0
    for f in files:
        attn = torch.load(f, map_location="cpu").numpy().astype(np.float32)
        ref = reference_map(attn, MOLMO_LAYER, MOLMO_HEAD, img_idx_all, valid_idx)
        new = new_reduction(attn, img_idx_all, valid_idx)[MOLMO_LAYER, MOLMO_HEAD]
        d = np.abs(ref.flatten() - new)
        rel = d.max() / (np.abs(ref).max() + 1e-12)
        worst = max(worst, rel)
        r = np.corrcoef(ref.flatten(), new)[0, 1]
        print(f"{f.name}: attn{attn.shape} | max|diff| {d.max():.3e} | "
              f"rel {rel:.3e} | r {r:.10f}")

    ok = worst < 1e-5
    print(f"\nVERDICT: {'MATCH' if ok else 'MISMATCH — do not use'} (worst rel {worst:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
