"""Extract attention maps for the NSD stimulus set from the four trained captioning models.

Reuses the beam-search extractors in architecture_manipulation/shared/extract_beam5.py so
the maps are produced exactly as in the paper; only the stimulus list changes.

Checkpoints are the canonical ones used for the manuscript (see tribe/scripts/
2_feature_extraction.py TRANS_CFG and rnn_model/analysis_scripts/get_rnn_h_c_alpha.py):
  describe -> phase2 (COCO finetune) ; social -> social_unfrozen

For the Transformer models we save ALL 8 heads of layer 3, not just the best head, so the
encoding analysis can use a CV-honest or mean-over-heads rule (rebuttal Task 1) rather than
inheriting the biased best-head selection.

Output: attn/{model}_{task}.npz
    nsdId  (n,) int
    alphas (n, 196) float32          - LSTM soft attention / Transformer L3 head-mean
    heads  (n, 8, 196) float32       - Transformer only, all layer-3 heads
    caption (n,) str

--all-layers (added 2026-09-23, ICLR revision): the layer-3 restriction above predates
Task 1's head sweep, which found the Social-task maxima in layer 4 (CNN-T L4H4, ViT-T L4H8),
so the encoding band could not be the behaviourally cross-validated head in that task.
With --all-layers the script writes every decoder layer instead:

Output: attn/{model}_{task}_alllayers.npz
    nsdId     (n,) int
    heads_all (n, num_layers, num_heads, 196) float32   - Transformer only
    alphas    (n, 196) float32                          - unchanged (word-mean map)
    caption   (n,) str
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).parent
ARCH = Path("/scratch/gpfs/GRAZIANO/CANS/architecture_manipulation")
RNN = Path("/scratch/gpfs/GRAZIANO/CANS/rnn_model")
sys.path.insert(0, str(ARCH / "shared"))
sys.path.insert(0, str(ARCH / "models"))
sys.path.insert(0, str(RNN))

from extract_beam5 import extract_lstm, extract_transformer  # noqa: E402

WORD_MAP = RNN / "data/splits/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
STIM = HERE / "stimuli"
OUT = HERE / "attn"
OUT.mkdir(exist_ok=True)

# model -> task -> checkpoint path. describe = phase2, social = social_unfrozen.
CKPT = {
    "CNN_LSTM": {
        "describe": RNN / "current_models/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq_final.pth.tar",
        "social": RNN / "current_models/BEST_checkpoint_CANS_1_cap_per_img_5_min_word_freq_final.pth.tar",
    },
    "ViT_LSTM": {
        "describe": ARCH / "ViT_LSTM/checkpoints/phase2/epoch003_bleu0.2457.pth.tar",
        "social": ARCH / "ViT_LSTM/checkpoints/social_unfrozen/epoch001_bleu0.1509.pth.tar",
    },
    "CNN_Transformer": {
        "describe": ARCH / "CNN_Transformer/checkpoints/phase2/epoch014_bleu0.2028.pth.tar",
        "social": ARCH / "CNN_Transformer/checkpoints/social_unfrozen/epoch003_bleu0.1383.pth.tar",
    },
    "ViT_Transformer": {
        "describe": ARCH / "ViT_Transformer/checkpoints/phase2/epoch011_bleu0.2035.pth.tar",
        "social": ARCH / "ViT_Transformer/checkpoints/social_unfrozen/epoch001_bleu0.1322.pth.tar",
    },
}
IS_TRANSFORMER = {"CNN_Transformer", "ViT_Transformer"}
IS_VIT = {"ViT_LSTM", "ViT_Transformer"}
LAYER3 = 3  # 1-indexed layer 3, matching the manuscript convention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPT))
    ap.add_argument("--task", required=True, choices=["describe", "social"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--all-layers", action="store_true",
                    help="save every decoder layer's heads to {model}_{task}_alllayers.npz "
                         "instead of only layer 3; leaves the existing files untouched")
    args = ap.parse_args()

    suffix = "_alllayers" if args.all_layers else ""
    out_path = OUT / f"{args.model}_{args.task}{suffix}.npz"
    if out_path.exists():
        print(f"exists, skipping: {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.model} / {args.task} on {device}")

    idx = pd.read_csv(HERE / "stimuli_index.csv")
    with open(WORD_MAP) as f:
        word_map = json.load(f)
    rev = {v: k for k, v in word_map.items()}

    ck = torch.load(CKPT[args.model][args.task], map_location="cpu", weights_only=False)
    encoder = ck["encoder"].to(device).eval()
    decoder = ck["decoder"].to(device).eval()
    print(f"  checkpoint: {CKPT[args.model][args.task].name}")

    is_tr = args.model in IS_TRANSFORMER
    img_size = 224 if args.model in IS_VIT else 256
    fn = extract_transformer if is_tr else extract_lstm

    nsd_ids, alphas, heads, caps = [], [], [], []
    t0 = time.time()
    for nsd_id in tqdm(idx["nsdId"], desc=f"{args.model}/{args.task}"):
        img_path = STIM / f"{int(nsd_id):05d}.png"
        try:
            seq, alpha, layer_alpha, head_layer = fn(
                encoder, decoder, str(img_path), word_map, args.beam_size, img_size)
        except Exception as e:
            print(f"  ERROR nsdId {nsd_id}: {e}")
            continue
        words = [rev[i] for i in seq if i not in
                 (word_map["<start>"], word_map["<end>"], word_map.get("<pad>", -1))]
        nsd_ids.append(int(nsd_id))
        alphas.append(np.asarray(alpha, dtype=np.float32).ravel())
        caps.append(" ".join(words))
        if is_tr and head_layer is not None:
            # head_layer: (num_layers, num_heads, 196)
            hl = np.asarray(head_layer, dtype=np.float32)
            heads.append(hl if args.all_layers else hl[LAYER3 - 1])

    payload = {
        "nsdId": np.array(nsd_ids),
        "alphas": np.stack(alphas),
        "caption": np.array(caps, dtype=object),
    }
    if heads:
        payload["heads_all" if args.all_layers else "heads"] = np.stack(heads)
    np.savez_compressed(out_path, **payload)
    print(f"saved {out_path}  n={len(nsd_ids)}  "
          f"alphas{payload['alphas'].shape}"
          + (f"  {'heads_all' if args.all_layers else 'heads'}"
             f"{payload['heads_all' if args.all_layers else 'heads'].shape}" if heads else "")
          + f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
