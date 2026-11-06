"""Convert raw Molmo / Qwen3.5 activations into the attn/{model}_{task}.npz format used
by the encoding pipeline.

Reduction logic mirrors tribe/scripts/2_feature_extraction.py so the NSD features are
computed exactly as the paper's are.

Head rule: `alphas` holds the MEAN over heads of the analysed layer -- a predefined rule,
consistent with how the Transformer models are treated here (rebuttal Task 1). The paper's
best heads are also stored, under `heads`, so `--head-rule best` remains available:
  Molmo   L9,  H24 social / H27 describe (1-indexed)
  Qwen3.5 L15, H12 both tasks

Output: attn/{Molmo,Qwen35}_{describe,social}.npz
    nsdId (n,), alphas (n, 196), heads (n, n_heads, 196)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).parent
ATTN = HERE / "attn"
ATTN.mkdir(exist_ok=True)
GRID, FLAT = 14, 196

MOLMO_LAYER = 9
MOLMO_HEAD = {"describe": 26, "social": 23}  # 0-based
QWEN_LAYER, QWEN_HEAD = 15, 11               # 0-based


def resize14(a):
    return cv2.resize(a.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)


def build_molmo(task, ids):
    """Maps are already reduced to (28 layers, 28 heads, 196) at inference time."""
    d = HERE / "vlm_out" / f"molmo_{task}"
    alphas, heads, keep = [], [], []
    for nsd_id in tqdm(ids, desc=f"Molmo {task}"):
        p = d / f"attnmap_img_{int(nsd_id):05d}.pt"
        if not p.exists():
            continue
        m = torch.load(p, map_location="cpu").float().numpy()   # (28, 28, 196)
        heads.append(m[MOLMO_LAYER])                            # (28, 196)
        alphas.append(m[MOLMO_LAYER].mean(axis=0))              # mean over heads
        keep.append(int(nsd_id))
    return np.array(keep), np.stack(alphas), np.stack(heads)


def build_qwen(task, ids):
    base = HERE / "vlm_out" / f"qwen35_{task}"
    act = base / "activations" / f"attn-layer-{QWEN_LAYER}"
    base_mask = torch.load(base / "image_mask.pt", map_location="cpu")
    if base_mask.dim() == 2:
        base_mask = base_mask.squeeze(0)

    alphas, heads, keep = [], [], []
    for nsd_id in tqdm(ids, desc=f"Qwen {task}"):
        p = act / f"{int(nsd_id):05d}.pt"
        if not p.exists():
            continue
        data = torch.load(p, map_location="cpu").squeeze(0).float()   # (n_heads, n_gen, seq)
        n_heads, _, seq_len = data.shape
        pad = seq_len - base_mask.shape[0]
        mask = F.pad(base_mask, (0, pad), value=False) if pad > 0 else base_mask[:seq_len]

        per_head = []
        for h in range(n_heads):
            avg = data[h].nanmean(dim=0)          # mean over generation steps -> (seq,)
            masked = avg[mask][1:-1]              # strip vision_start / vision_end
            n_tok = masked.shape[0]
            side = int(np.sqrt(n_tok))
            if side * side != n_tok:
                masked = masked[:side * side]
            per_head.append(resize14(masked.numpy().reshape(side, side)).ravel())
        per_head = np.stack(per_head)             # (n_heads, 196)
        heads.append(per_head)
        alphas.append(per_head.mean(axis=0))
        keep.append(int(nsd_id))
    return np.array(keep), np.stack(alphas), np.stack(heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Molmo", "Qwen35"])
    args = ap.parse_args()

    ids = pd.read_csv(HERE / "stimuli_index.csv")["nsdId"].to_numpy()
    for model in args.models:
        for task in ("describe", "social"):
            out = ATTN / f"{model}_{task}.npz"
            builder = build_molmo if model == "Molmo" else build_qwen
            try:
                kept, alphas, heads = builder(task, ids)
            except FileNotFoundError as e:
                print(f"  {model}/{task}: {e} -- skipping")
                continue
            if len(kept) < len(ids):
                print(f"  {model}/{task}: {len(kept)}/{len(ids)} images present")
            np.savez_compressed(out, nsdId=kept, alphas=alphas, heads=heads,
                                caption=np.array([""] * len(kept), dtype=object))
            print(f"saved {out.name}: alphas{alphas.shape} heads{heads.shape} "
                  f"| entropy={_entropy(alphas):.4f}")


def _entropy(a):
    p = a / (a.sum(axis=1, keepdims=True) + 1e-12)
    return float(np.mean(-(p * np.log(p + 1e-12)).sum(axis=1) / np.log(a.shape[1])))


if __name__ == "__main__":
    main()
