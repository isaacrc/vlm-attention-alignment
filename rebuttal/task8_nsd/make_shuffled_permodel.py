"""Per-model permutation floors for the NSD encoding model.

The original `Shuffled` baseline was CNN-LSTM's maps with the image labels permuted, and
that single floor was subtracted from every model (make_baselines.py, summarize_final.py).
That is not what the Methods and the Figure 3 caption describe, and it is not the right
reference: a permutation floor inherits the spatial statistics of the maps it is built
from, so applying CNN-LSTM's floor to Molmo or to edge energy tests each model against
another model's noise level rather than its own.

This writes one floor per model x task: that model's OWN cross-validated attention maps,
image labels permuted with a fixed seed. Features are baked out as `alphas`, so the fit
loads them through the ordinary path and no head selection is re-applied downstream.

Output: attn/Shuf{MODEL}_{task}.npz
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ATTN = HERE / "attn"
SEED = 42

from fit_encoding import CVBEHAV_HEADS

MODELS = ["CNN_LSTM", "ViT_LSTM", "CNN_Transformer", "ViT_Transformer",
          "Molmo", "Qwen35", "EdgeEnergy", "Luminance"]


def cvbehav_features(model, task):
    """(ids, feats) exactly as fit_encoding._load_cvbehav would select them."""
    if model in ("CNN_Transformer", "ViT_Transformer"):
        d = np.load(ATTN / f"{model}_{task}_alllayers.npz", allow_pickle=True)
        L, H = CVBEHAV_HEADS[model][task]
        return d["nsdId"], d["heads_all"][:, L - 1, H - 1, :]
    if model == "Molmo":
        d = np.load(ATTN / f"Molmo_{task}_cvbehav.npz", allow_pickle=True)
        return d["nsdId"], d["alphas"]
    d = np.load(ATTN / f"{model}_{task}.npz", allow_pickle=True)
    if model == "Qwen35" and "heads" in d:
        return d["nsdId"], d["heads"][:, CVBEHAV_HEADS[model][task][1] - 1, :]
    return d["nsdId"], d["alphas"]


if __name__ == "__main__":
    for task in ("social", "describe"):
        for model in MODELS:
            ids, feats = cvbehav_features(model, task)
            feats = np.asarray(feats, dtype=np.float32)
            # One permutation per model x task, seeded so the floor is reproducible.
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(ids))
            out = ATTN / f"Shuf{model}_{task}.npz"
            np.savez_compressed(out, nsdId=ids, alphas=feats[perm],
                                caption=np.array([""] * len(ids), dtype=object))
            moved = float((perm != np.arange(len(ids))).mean())
            print(f"{out.name}: {feats.shape}, {moved:.1%} of rows displaced")
