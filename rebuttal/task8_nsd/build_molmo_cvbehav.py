"""Build Molmo's behaviourally cross-validated attention band on the NSD stimuli.

ICLR revision (2026-09-23). build_vlm_features.py stored only layer 10's 28 heads, so
Molmo's Describe band in the encoding analysis was L10H27 -- the in-sample argmax -- while
rebuttal Task 1's cross-validated Describe head is L8H7. The per-image .pt maps under
vlm_out/molmo_{task}/ retain all 28 layers x 28 heads, so the correct band costs nothing
to recover; no model inference is re-run.

Heads (1-indexed, from task1_head_selection/results/selected_heads.json):
    social   L10H24   (5/5 folds; identical to what the encoding analysis already used)
    describe L8H7     (3/5 folds; replaces L10H27)

Output: attn/Molmo_{task}_cvbehav.npz  with nsdId and alphas (n, 196).
"""
import numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm

HERE = Path(__file__).parent
ATTN = HERE / "attn"
CV = {"social": (10, 24), "describe": (8, 7)}   # 1-indexed (layer, head)

ids = pd.read_csv(HERE / "stimuli_index.csv")["nsdId"].to_numpy()
for task, (L, H) in CV.items():
    d = HERE / "vlm_out" / f"molmo_{task}"
    keep, alphas = [], []
    for nsd_id in tqdm(ids, desc=f"Molmo {task} L{L}H{H}"):
        p = d / f"attnmap_img_{int(nsd_id):05d}.pt"
        if not p.exists():
            continue
        m = torch.load(p, map_location="cpu").float().numpy()   # (28, 28, 196)
        alphas.append(m[L - 1, H - 1])
        keep.append(int(nsd_id))
    out = ATTN / f"Molmo_{task}_cvbehav.npz"
    np.savez_compressed(out, nsdId=np.array(keep), alphas=np.stack(alphas).astype(np.float32))
    print(f"saved {out}  n={len(keep)}  L{L}H{H}")
